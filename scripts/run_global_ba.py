"""Stage 6-7: full-resolution matching on graph edges + global BA (InstantSfM / bae), from a graph checkpoint.

The VGGT graph (sweep + refinement, optionally GPS) is only the preconditioner: it gives
initial poses, intrinsics and which pairs to match. Here we
  1. map the seed frame to meters with the GPS (if given),
  2. extract RootSIFT on the original images and match only graph edges,
  3. build tracks, triangulate from the initial poses,
  4. run InstantSfM's GPU BA (alternating with pixel-threshold track filtering),
and evaluate initial vs refined poses against the COLMAP reference (eval only).

    python scripts/run_global_ba.py --scene /mnt/z/.../HAV/colmap_metrics \
        --ckpt runs/HAV_graph_v6_gps/ckpt/latest.pkl --out runs/HAV_ba_v1 --use-gps
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.ba.instantsfm_ba import reprojection_errors, run_global_ba
from fgsfm.eval.pose_metrics import pose_metrics
from fgsfm.geometry.transforms import centers, robust_umeyama
from fgsfm.io.gps import gps_enu_for
from fgsfm.io.scene import load_reference_scene
from fgsfm.match.matcher import match_pairs
from fgsfm.match.sift import extract_features
from fgsfm.match.tracks import build_tracks, subset, triangulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--use-gps", action="store_true", help="express the graph in meters via GPS before BA")
    ap.add_argument("--match-width", type=int, default=2736, help="SIFT image width (full res = 5472)")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--pair-overlap", type=float, default=0.1, help="graph edges (measured or predicted) to match")
    ap.add_argument("--topk", type=int, default=20, help="match top-K neighbours per image (0 = threshold mode)")
    ap.add_argument("--sift-cache", type=Path, default=Path("runs/cache/sift"))
    ap.add_argument("--topk-min-overlap", type=float, default=0.05)
    ap.add_argument("--init-reproj-px", type=float, default=64.0, help="triangulation filter with initial poses")
    ap.add_argument("--filter-px", type=str, default="16,8,4")
    ap.add_argument("--fix-intrinsics", action="store_true")
    ap.add_argument("--max-tracks", type=int, default=0, help="subsample tracks for BA (0 = all)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    T = {}
    t0 = time.time()

    # ---- reference (evaluation only) + graph checkpoint
    scene = load_reference_scene(args.scene)
    ref = scene.graph.cameras
    names = [c.name for c in ref]
    with open(args.ckpt, "rb") as f:
        state = pickle.load(f)
    g = state["graph"]
    assert g.names == names
    cams = g.cams_in(g.seed_frame)
    print(f"checkpoint {args.ckpt}: {len(cams)}/{len(names)} cameras in the seed frame", flush=True)

    # ---- initial poses in meters (GPS) and full-resolution intrinsics (VGGT, shared camera)
    w2c0 = g.w2c[cams].copy()
    if args.use_gps:
        scene_name = args.scene.parent.name if args.scene.name == "colmap_metrics" else args.scene.name
        gps, ok = gps_enu_for(names, args.scene / "AT-export.xml", cache=Path("runs/cache") / f"{scene_name}_gps.npz")
        m = ok[cams]
        S = robust_umeyama(centers(w2c0[m]), gps[cams][m])
        w2c0 = S.apply_w2c(w2c0)
        print(f"seed frame -> ENU meters: scale {S.s:.3f}", flush=True)
    W, H = ref[0].width, ref[0].height
    h_m, w_m = g.image_hw
    Kf = g.K[cams].copy()
    Kf[:, 0] *= W / w_m
    Kf[:, 1] *= H / h_m
    K0 = np.array([[np.median(Kf[:, 0, 0]), 0, W / 2], [0, np.median(Kf[:, 1, 1]), H / 2], [0, 0, 1.0]])
    print(f"initial intrinsics (VGGT, full res): fx={K0[0, 0]:.1f} fy={K0[1, 1]:.1f}  "
          f"[reference fx={ref[0].K[0, 0]:.1f}]", flush=True)

    # ---- features on the original images
    t = time.time()
    paths = [args.scene / "images" / names[c] for c in cams]
    feats = extract_features(paths, args.sift_cache, args.match_width, args.max_features)
    T["features_s"] = time.time() - t
    print(f"features: median {int(np.median([len(f['xy']) for f in feats]))} per image ({T['features_s']:.0f}s)", flush=True)

    # ---- pairs = graph edges (measured VGGT overlap or keypoint-predicted overlap)
    cams_p, O = g.predicted_overlap(g.seed_frame)
    assert (cams_p == cams).all()
    M = np.where(g.cobatched[np.ix_(cams, cams)], np.nan_to_num(g.measured[np.ix_(cams, cams)]), 0.0)
    Sc = np.maximum(O, M)
    np.fill_diagonal(Sc, 0.0)
    if args.topk > 0:
        # top-K predicted/measured neighbours per image (symmetrized), above a small floor
        E = np.zeros_like(Sc, dtype=bool)
        nn = np.argsort(-Sc, axis=1)[:, :args.topk]
        E[np.repeat(np.arange(len(Sc)), args.topk), nn.ravel()] = True
        E &= Sc >= args.topk_min_overlap
        E |= E.T
    else:
        E = Sc >= args.pair_overlap
    pairs = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(E)))]
    pair_tag = f"k{args.topk}" if args.topk > 0 else f"o{args.pair_overlap}"
    cache = args.out / f"tracks_w{args.match_width}_n{args.max_features}_{pair_tag}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            c = pickle.load(f)
        matches, tracks = c["matches"], c["tracks"]
        print(f"loaded cached matches/tracks from {cache}", flush=True)
    else:
        t = time.time()
        matches = match_pairs(feats, pairs)
        T["matching_s"] = time.time() - t
        print(f"pairs: {len(pairs)} graph edges -> {len(matches)} verified "
              f"(median {int(np.median([len(v) for v in matches.values()]))} inliers) ({T['matching_s']:.0f}s)", flush=True)
        # ---- tracks + triangulation from the initial poses
        t = time.time()
        tracks = build_tracks(matches, [len(f["xy"]) for f in feats])
        keep = triangulate(tracks, feats, w2c0, K0, args.init_reproj_px)
        tracks = subset(tracks, keep)
        T["tracks_s"] = time.time() - t
        with open(cache, "wb") as f:
            pickle.dump(dict(matches=matches, tracks=tracks), f, protocol=pickle.HIGHEST_PROTOCOL)
    lens = np.array([len(o) for o in tracks.obs_img])
    print(f"tracks: {len(tracks)} triangulated, mean length {lens.mean():.2f}", flush=True)
    if args.max_tracks and len(tracks) > args.max_tracks:
        sel = np.random.default_rng(0).choice(len(tracks), args.max_tracks, replace=False)
        m = np.zeros(len(tracks), bool)
        m[sel] = True
        tracks = subset(tracks, m)
        print(f"  subsampled to {len(tracks)} tracks", flush=True)

    # ---- global BA
    w2c_ref = np.stack([c.pose_w2c for c in ref])[cams]
    before = pose_metrics(w2c0, w2c_ref)
    e0, _ = reprojection_errors(w2c0, K0, tracks, feats)
    t = time.time()
    out = run_global_ba(w2c0, K0, (W, H), tracks, feats, tuple(float(x) for x in args.filter_px.split(",")),
                        options=dict(optimize_intrinsics=not args.fix_intrinsics))
    T["ba_s"] = time.time() - t
    after = pose_metrics(out.w2c, w2c_ref)
    T["total_s"] = time.time() - t0

    keys = ("num_cameras", "abs_rot_median_deg", "rel_rot_median_deg", "rel_trans_dir_median_deg",
            "RRA_5", "RTA_5", "AUC@3", "AUC@5", "AUC@15", "ATE_median", "ATE_rmse")
    print(f"\n{'metric':<26}{'initial (VGGT graph)':>22}{'after global BA':>18}")
    for k in keys:
        print(f"{k:<26}{before[k]:>22.4f}{after[k]:>18.4f}")
    print(f"{'reproj median px':<26}{np.median(e0):>22.2f}{out.reproj_px[-1][1]:>18.2f}")
    print(f"{'focal fx (ref ' + format(ref[0].K[0, 0], '.1f') + ')':<26}{K0[0, 0]:>22.1f}{out.K[0, 0]:>18.1f}")
    print(json.dumps(T))

    # final tracks: flat observation table (track, local image, full-res pixel) for export / coloring
    obs_trk = np.concatenate([np.full(len(o), t) for t, o in enumerate(out.tracks.obs_img)])
    obs_img = np.concatenate(out.tracks.obs_img)
    obs_xy = np.concatenate([feats[i]["xy"][f] for imgs, fts in zip(out.tracks.obs_img, out.tracks.obs_feat)
                             for i, f in zip(imgs, fts)]).reshape(-1, 2)
    np.savez_compressed(args.out / "ba_result.npz", cams=cams, names=np.array([names[c] for c in cams]),
                        w2c_init=w2c0, w2c=out.w2c, K_init=K0, K=out.K, image_wh=np.array([W, H]),
                        points=out.tracks.xyz.astype(np.float32), obs_track=obs_trk.astype(np.int32),
                        obs_image=obs_img.astype(np.int32), obs_xy=obs_xy.astype(np.float32),
                        obs_per_image=np.bincount(obs_img, minlength=len(cams)))
    (args.out / "metrics.json").write_text(json.dumps(dict(
        ckpt=str(args.ckpt), use_gps=args.use_gps, match_width=args.match_width, pairs=len(pairs),
        verified_pairs=len(matches), tracks_initial=len(tracks), ba_stages=dict(reproj=out.reproj_px, tracks=out.num_tracks),
        before=before, after=after, focal=dict(init=K0[0, 0], ba=out.K[0, 0], ref=ref[0].K[0, 0]), time=T),
        indent=1, default=float))


if __name__ == "__main__":
    main()
