"""Global BA via InstantSfM's GPU bundle adjuster (third_party/InstantSfM, bae LM + PCG).

We keep our own data (poses, intrinsics, full-resolution tracks) and only
convert to InstantSfM's Cameras / Images / Tracks containers to call
TorchBA.Solve. Like their global mapper we alternate BA with track filtering,
but filter in pixels with a shrinking threshold.

License: InstantSfM is CC BY-NC 4.0 -> research baseline only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "InstantSfM"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from instantsfm.processors.bundle_adjustment import TorchBA          # noqa: E402
from instantsfm.scene.defs import CameraModelId, Cameras, Images, Tracks   # noqa: E402

from fgsfm.match.tracks import TrackSet                             # noqa: E402

DEFAULT_OPTIONS = dict(optimize_poses=True, optimize_points=True, optimize_intrinsics=True,
                       min_num_view_per_track=2, thres_loss_function=1.0, max_num_iterations=100,
                       function_tolerance=5e-4, depth_weight=0.1)


@dataclass
class BAOutput:
    w2c: np.ndarray            # (N, 4, 4)
    K: np.ndarray              # (3, 3) shared pinhole
    tracks: TrackSet
    reproj_px: list            # per stage: median reprojection error before / after
    num_tracks: list


def reprojection_errors(w2c, K, tracks: TrackSet, feats) -> tuple[np.ndarray, np.ndarray]:
    """All observation errors (px) and the owning track index."""
    errs, owner = [], []
    for t, (imgs, fts) in enumerate(zip(tracks.obs_img, tracks.obs_feat)):
        X = np.append(tracks.xyz[t], 1.0)
        x = np.einsum("nij,j->ni", w2c[imgs, :3, :], X)
        p = (x @ K.T)
        uv = p[:, :2] / np.maximum(p[:, 2:], 1e-9)
        obs = np.stack([feats[i]["xy"][f] for i, f in zip(imgs, fts)])
        e = np.linalg.norm(uv - obs, axis=1)
        e[x[:, 2] <= 0] = np.inf
        errs.append(e)
        owner.append(np.full(len(imgs), t))
    return np.concatenate(errs), np.concatenate(owner)


def filter_tracks(w2c, K, tracks: TrackSet, feats, max_px: float) -> TrackSet:
    """Drop observations above max_px; drop tracks left with < 2 observations."""
    errs, owner = reprojection_errors(w2c, K, tracks, feats)
    keep_obs = errs < max_px
    obs_img, obs_feat, xyz = [], [], []
    start = 0
    for t in range(len(tracks)):
        n = len(tracks.obs_img[t])
        k = keep_obs[start:start + n]
        start += n
        if k.sum() >= 2:
            obs_img.append(tracks.obs_img[t][k])
            obs_feat.append(tracks.obs_feat[t][k])
            xyz.append(tracks.xyz[t])
    return TrackSet(obs_img, obs_feat, np.array(xyz).reshape(-1, 3))


def _to_instantsfm(w2c, K, wh, tracks: TrackSet, feats):
    n = len(w2c)
    cams = Cameras(num_cameras=1)
    cams.widths[0], cams.heights[0] = wh
    cams.has_prior_focal_length[0] = True
    cams.set_params(0, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), CameraModelId.PINHOLE)
    imgs = Images(num_images=n)
    imgs.cam_ids[:] = 0
    imgs.is_registered[:] = True
    imgs.world2cams = w2c.copy()
    for i in range(n):
        imgs.features[i] = feats[i]["xy"].astype(np.float64)
    trk = Tracks(num_tracks=len(tracks))
    trk.xyzs = tracks.xyz.copy()
    trk.is_initialized[:] = True
    trk.ids = np.arange(len(tracks), dtype=np.int32)
    trk.observations = [np.stack([a, b], 1).astype(np.int32) for a, b in zip(tracks.obs_img, tracks.obs_feat)]
    return cams, imgs, trk


def run_global_ba(w2c0: np.ndarray, K0: np.ndarray, wh: tuple[int, int], tracks: TrackSet, feats: list[dict],
                  filter_px=(16.0, 8.0, 4.0), options: dict | None = None, device: str = "cuda:0",
                  intrinsics_mode: str = "shared_f+pp") -> BAOutput:
    """intrinsics_mode: "shared_f+pp" (default; one focal + principal point, fgsfm/ba/full_intrinsics.py)
    | "focal+pp" | "focal" (InstantSfM TorchBA: principal point fixed at the given value)."""
    opts = dict(DEFAULT_OPTIONS, **(options or {}))
    w2c, K = w2c0.copy(), K0.copy()
    reproj, ntracks = [], []
    ba = TorchBA(device=device)
    for px in filter_px:
        tracks = filter_tracks(w2c, K, tracks, feats, 4 * px)              # loose pre-filter of gross outliers
        e0, _ = reprojection_errors(w2c, K, tracks, feats)
        cams, imgs, trk = _to_instantsfm(w2c, K, wh, tracks, feats)
        if intrinsics_mode == "focal" or not opts["optimize_intrinsics"]:
            ba.Solve(cams, imgs, trk, opts, use_depths=False, optimize_intrinsics=opts["optimize_intrinsics"])
        else:
            from fgsfm.ba.full_intrinsics import solve_full_intrinsics
            solve_full_intrinsics(cams, imgs, trk, opts, mode=intrinsics_mode, device=device)
        # Solve filters tracks by min views and updates containers in place: read everything back
        w2c = imgs.world2cams.copy()
        fx, fy, cx, cy = cams.params[0, :4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        tracks = TrackSet([o[:, 0] for o in trk.observations], [o[:, 1] for o in trk.observations], trk.xyzs.copy())
        e1, _ = reprojection_errors(w2c, K, tracks, feats)
        tracks = filter_tracks(w2c, K, tracks, feats, px)
        reproj.append((float(np.median(e0)), float(np.median(e1))))
        ntracks.append(len(tracks))
        print(f"  BA stage (filter {px:.0f}px): reproj median {np.median(e0):.2f} -> {np.median(e1):.2f} px, "
              f"tracks kept {len(tracks)}, f=({fx:.1f}, {fy:.1f}) pp=({cx:.1f}, {cy:.1f})", flush=True)
    return BAOutput(w2c, K, tracks, reproj, ntracks)
