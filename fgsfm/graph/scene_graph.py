"""Incrementally built scene graph (README Stages 3-4, no GPS).

Cameras live in *frames*: each chain of consistently linked batches (a
segment) has its own Sim(3) frame. Growth merges segments into the seed
frame. Every camera stores its pose, intrinsics, sampled depth (for
depth-ratio scale estimation) and a retrieval descriptor; every healthy batch
adds overlap measurements; registered batches add 3D keypoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fgsfm.fm.keypoints import Keypoints
from fgsfm.geometry.transforms import Sim3, centers


@dataclass
class BatchOutput:
    """Light, CPU-side summary of one FM batch."""
    idx: np.ndarray            # (S,) global camera indices
    kind: str                  # "sweep" | "densify" | "link" | "verify"
    w2c: np.ndarray            # (S, 4, 4) in the batch frame
    K: np.ndarray              # (S, 3, 3) model resolution
    depth: np.ndarray          # (S, P) depth at the sampled grid, batch units
    valid: np.ndarray          # (S, P) confident samples
    overlap: np.ndarray        # (S, S) symmetric overlap
    cam_conf: np.ndarray       # (S,) median depth confidence per image
    health: float              # median depth confidence of the batch
    descriptors: np.ndarray    # (S, D)
    keypoints: Keypoints


@dataclass
class KeypointStore:
    X: list = field(default_factory=list)        # per block (K, 3) in its frame
    rgb: list = field(default_factory=list)
    conf: list = field(default_factory=list)
    frame: list = field(default_factory=list)    # per block frame id
    obs_kp: list = field(default_factory=list)   # per block (M,) local kp index
    obs_cam: list = field(default_factory=list)  # per block (M,) global camera index
    obs_uv: list = field(default_factory=list)   # per block (M, 2)

    def add(self, X, rgb, conf, frame, obs_kp, obs_cam, obs_uv):
        self.X.append(X)
        self.rgb.append(rgb)
        self.conf.append(conf)
        self.frame.append(frame)
        self.obs_kp.append(obs_kp)
        self.obs_cam.append(obs_cam)
        self.obs_uv.append(obs_uv)

    def blocks_in(self, frame: int) -> list[int]:
        return [b for b, f in enumerate(self.frame) if f == frame]

    def gather(self, frame: int):
        """Concatenate all keypoints of a frame: X, rgb, conf, obs_kp (global row), obs_cam, obs_uv, block id."""
        Xs, rgbs, confs, ok, oc, ou, blk = [], [], [], [], [], [], []
        off = 0
        for b in self.blocks_in(frame):
            Xs.append(self.X[b])
            rgbs.append(self.rgb[b])
            confs.append(self.conf[b])
            ok.append(self.obs_kp[b] + off)
            oc.append(self.obs_cam[b])
            ou.append(self.obs_uv[b])
            blk.append(np.full(len(self.X[b]), b))
            off += len(self.X[b])
        if not Xs:
            z = np.zeros((0, 3))
            return z, np.zeros((0, 3), np.uint8), np.zeros(0), np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)), np.zeros(0, int)
        return (np.concatenate(Xs), np.concatenate(rgbs), np.concatenate(confs), np.concatenate(ok),
                np.concatenate(oc), np.concatenate(ou), np.concatenate(blk))

    def scatter(self, frame: int, X: np.ndarray) -> None:
        """Write back optimized positions (same order as gather)."""
        off = 0
        for b in self.blocks_in(frame):
            k = len(self.X[b])
            self.X[b] = X[off:off + k]
            off += k


class SceneGraph:
    def __init__(self, names: list[str], image_hw: tuple[int, int]):
        n = len(names)
        self.names = names
        self.n = n
        self.image_hw = image_hw
        self.frame_of = np.full(n, -1)                  # -1 = not registered
        self.w2c = np.tile(np.eye(4), (n, 1, 1))
        self.K = np.zeros((n, 3, 3))
        self.depth = [None] * n                         # (P,) sampled depth in frame units
        self.depth_valid = [None] * n
        self.desc = None                                # (n, D)
        self.registered_by = np.full(n, -1)             # batch id that registered the camera
        self.ov_sum = np.zeros((n, n))
        self.ov_cnt = np.zeros((n, n), int)
        self.batches: list[dict] = []                   # bookkeeping only
        self.kp = KeypointStore()
        self._next_frame = 0
        self.seed_frame = -1
        self.origin = -1

    # ------------------------------------------------------------ measurements

    def add_measurement(self, b: BatchOutput) -> None:
        ii, jj = np.meshgrid(b.idx, b.idx, indexing="ij")
        self.ov_sum[ii, jj] += b.overlap
        self.ov_cnt[ii, jj] += 1

    def set_descriptors(self, b: BatchOutput) -> None:
        if self.desc is None:
            self.desc = np.full((self.n, b.descriptors.shape[1]), np.nan)
        new = np.isnan(self.desc[b.idx, 0])
        self.desc[b.idx[new]] = b.descriptors[new]

    @property
    def measured(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            M = np.where(self.ov_cnt > 0, self.ov_sum / np.maximum(self.ov_cnt, 1), np.nan)
        np.fill_diagonal(M, 1.0)
        return M

    @property
    def cobatched(self) -> np.ndarray:
        return self.ov_cnt > 0

    # ------------------------------------------------------------ frames

    def new_frame(self) -> int:
        f = self._next_frame
        self._next_frame += 1
        return f

    def cams_in(self, frame: int) -> np.ndarray:
        return np.flatnonzero(self.frame_of == frame)

    def frames(self) -> dict[int, np.ndarray]:
        return {int(f): self.cams_in(f) for f in np.unique(self.frame_of) if f >= 0}

    def register_batch(self, b: BatchOutput, batch_id: int, frame: int, sim: Sim3,
                       cams_local: np.ndarray, add_keypoints: bool = True) -> None:
        """Put the listed (local) cameras of batch b into `frame` via Sim(3) frame<-batch; add keypoints."""
        for l in cams_local:
            g = b.idx[l]
            self.frame_of[g] = frame
            self.w2c[g] = sim.apply_w2c(b.w2c[l])
            self.K[g] = b.K[l]
            self.depth[g] = sim.s * b.depth[l]
            self.depth_valid[g] = b.valid[l]
            self.registered_by[g] = batch_id
        if add_keypoints and len(b.keypoints):
            kp = b.keypoints
            in_frame = self.frame_of[b.idx[kp.obs_img]] == frame
            if in_frame.any():
                self.kp.add(sim.apply_points(kp.X), kp.rgb, kp.conf, frame,
                            kp.obs_kp[in_frame], b.idx[kp.obs_img[in_frame]], kp.obs_uv[in_frame])

    def transform_frame(self, frame: int, sim: Sim3, new_frame: int | None = None) -> None:
        """Apply sim to every camera / keypoint of `frame` (and relabel to new_frame)."""
        cams = self.cams_in(frame)
        if len(cams):
            self.w2c[cams] = sim.apply_w2c(self.w2c[cams])
            for c in cams:
                self.depth[c] = sim.s * self.depth[c]
        for blk in self.kp.blocks_in(frame):
            self.kp.X[blk] = sim.apply_points(self.kp.X[blk])
            if new_frame is not None:
                self.kp.frame[blk] = new_frame
        if new_frame is not None:
            self.frame_of[cams] = new_frame

    def log_depth_ratios(self, cams_global: np.ndarray, depth_other: np.ndarray, valid_other: np.ndarray) -> np.ndarray:
        """Per-pixel log(depth_in_graph / depth_other) on the given cameras (same sample grid)."""
        out = []
        for c, d, v in zip(cams_global, depth_other, valid_other):
            if self.depth[c] is None:
                continue
            m = v & self.depth_valid[c] & (d > 0) & (self.depth[c] > 0)
            if m.sum() > 50:
                out.append(np.log(self.depth[c][m] / d[m]))
        return np.concatenate(out) if out else np.zeros(0)

    # ------------------------------------------------------------ prediction

    def predicted_overlap(self, frame: int, max_pts_per_cam: int = 400, rng=None) -> tuple[np.ndarray, np.ndarray]:
        """Overlap predicted by projecting each camera's keypoints into every other camera of the frame.

        Returns (cams, O) with O (len(cams), len(cams)) = min of both directions.
        """
        rng = rng or np.random.default_rng(0)
        cams = self.cams_in(frame)
        X, _, _, obs_kp, obs_cam, _, _ = self.kp.gather(frame)
        if len(cams) == 0 or len(X) == 0:
            return cams, np.zeros((len(cams), len(cams)))
        H, W = self.image_hw
        R, t = self.w2c[cams, :3, :3], self.w2c[cams, :3, 3]
        K = self.K[cams]
        covis = np.zeros((len(cams), len(cams)))
        pos = {c: k for k, c in enumerate(cams)}
        order = np.argsort(obs_cam, kind="stable")
        bounds = np.searchsorted(obs_cam[order], cams)
        ends = np.searchsorted(obs_cam[order], cams, side="right")
        for a, c in enumerate(cams):
            kps = obs_kp[order[bounds[a]:ends[a]]]
            if len(kps) == 0:
                continue
            if len(kps) > max_pts_per_cam:
                kps = rng.choice(kps, max_pts_per_cam, replace=False)
            P = X[kps]
            Xc = np.einsum("nij,pj->npi", R, P) + t[:, None]
            z = Xc[..., 2]
            uvw = np.einsum("nij,npj->npi", K, Xc)
            u, v = uvw[..., 0] / np.maximum(z, 1e-9), uvw[..., 1] / np.maximum(z, 1e-9)
            inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            covis[a] = inside.mean(1)
        O = np.minimum(covis, covis.T)
        np.fill_diagonal(O, 1.0)
        return cams, O

    def camera_centers(self) -> np.ndarray:
        return centers(self.w2c)
