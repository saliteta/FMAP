"""Remove isolated points from a dense PLY (scripts/export_gs_datasets.py output), scale-aware.

A point is an outlier when the mean distance to its k nearest neighbours exceeds max(--min-spacing, --rel * d),
d = distance to the nearest camera. VGGT samples get sparser with distance (one stride-4 sample spans ~0.008 d
at the DJI M3E field of view), so a fixed threshold would strip legitimate far geometry while missing flying points
near the cameras. Writes <out>_kept.ply and <out>_removed.ply (removed points in red, to check in view_ply.py).

    python scripts/filter_ply.py runs/feihong_e2e/feihong_20M.ply --ba runs/feihong_e2e/ba/ba_result.npz \
        --out runs/feihong_e2e/feihong_20M_filtered
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gs_datasets import write_points_ply  # noqa: E402
from view_point_clouds import read_ply  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--ba", type=Path, required=True, help="ba_result.npz with the camera poses")
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--min-spacing", type=float, default=0.8, help="metres")
    ap.add_argument("--rel", type=float, default=0.1, help="allowed mean neighbour distance per metre of range")
    args = ap.parse_args()

    t = time.time()
    xyz, rgb = read_ply(args.ply)
    d = np.load(args.ba)
    C = -np.einsum("nji,nj->ni", d["w2c"][:, :3, :3], d["w2c"][:, :3, 3])
    dc, _ = cKDTree(C).query(xyz, workers=-1)
    nn, _ = cKDTree(xyz).query(xyz, k=args.k + 1, workers=-1)
    spacing = nn[:, 1:].mean(1)
    out = spacing > np.maximum(args.min_spacing, args.rel * dc)
    write_points_ply(args.out.with_name(args.out.name + "_kept.ply"), xyz[~out], rgb[~out])
    write_points_ply(args.out.with_name(args.out.name + "_removed.ply"), xyz[out],
                     np.tile(np.array([[255, 0, 0]], np.uint8), (out.sum(), 1)))
    print(f"{len(xyz)} points: removed {out.sum()} ({100 * out.mean():.2f}%), kept {(~out).sum()}; removed points' "
          f"distance to the nearest camera median {np.median(dc[out]):.1f} m, max {dc[out].max():.0f} m; "
          f"kept max {dc[~out].max():.0f} m ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
