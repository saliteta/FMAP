"""Densify the 3DGS initial point cloud with VGGT depth, corrected per image by the BA tracks.

For every image the stored VGGT depth samples (stride-4 grid of the 518x350 model image, confident pixels only)
are distorted relative to the final BA geometry. The BA track observations in that image give exact depths at
sparse pixels, so per image we fit a smooth correction

    log z_BA = a * log z_VGGT + poly2(u, v)          (robust IRLS, Huber)

then lift every confident VGGT sample with the corrected depth and the BA pose / intrinsics. The fit is
validated on held-out observations (relative depth error). Points from all images are voxel-downsampled to a
target count and merged with the BA track points.

    python scripts/densify_vggt_points.py --ba runs/ab5_gps_prior/ba_result.npz \
        --ckpt runs/e2e_HAV/graph/ckpt/latest.pkl --base runs/gs_ab5/ours --out runs/gs_dense/ours --target 2000000
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_gs_datasets import write_points_ply  # noqa: E402

STRIDE = 4


def design(u, v, logz):
    """Columns of the correction model: a*log z_VGGT + 2nd-order polynomial in normalized pixel coords."""
    return np.stack([logz, np.ones_like(u), u, v, u * u, u * v, v * v], 1)


def irls(A, y, delta=0.05, iters=10):
    w = np.ones(len(y))
    for _ in range(iters):
        sw = np.sqrt(w)
        x = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(A @ x - y)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return x


def bilinear(grid, valid, gx, gy):
    """Sample grid at float indices; NaN where any of the 4 neighbours is invalid / outside."""
    ny, nx = grid.shape
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < nx) & (y0 + 1 < ny)
    out = np.full(len(gx), np.nan)
    x0, y0, fx, fy = x0[ok], y0[ok], gx[ok] - x0[ok], gy[ok] - y0[ok]
    v = valid[y0, x0] & valid[y0, x0 + 1] & valid[y0 + 1, x0] & valid[y0 + 1, x0 + 1]
    g = (grid[y0, x0] * (1 - fx) * (1 - fy) + grid[y0, x0 + 1] * fx * (1 - fy)
         + grid[y0 + 1, x0] * (1 - fx) * fy + grid[y0 + 1, x0 + 1] * fx * fy)
    out[np.flatnonzero(ok)] = np.where(v, g, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="graph builder checkpoint holding the VGGT depths")
    ap.add_argument("--base", type=Path, required=True, help="gsplat dataset whose cameras/images/points to reuse")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path("runs/gs_HAV_fast/split.json"))
    ap.add_argument("--images", type=Path, default=Path("runs/gs_HAV/images"), help="1600 px images for colors")
    ap.add_argument("--target", type=int, default=2_000_000, help="total initial points (tracks + VGGT)")
    ap.add_argument("--min-obs", type=int, default=50, help="track observations needed to fit an image")
    ap.add_argument("--max-fit-err", type=float, default=0.03, help="reject images with held-out median rel err above")
    args = ap.parse_args()
    t0 = time.time()

    d = dict(np.load(args.ba))
    names = [str(n) for n in d["names"]]
    W, H = (int(v) for v in d["image_wh"])
    w2c, K, pts = d["w2c"], d["K"], d["points"]
    g = pickle.load(open(args.ckpt, "rb"))["graph"]
    hm, wm = g.image_hw
    ny, nx = (hm + STRIDE - 1) // STRIDE, (wm + STRIDE - 1) // STRIDE
    split = json.loads(args.split.read_text())
    train = set(split["names"]) - set(split["test"])          # never lift test images (no test-view leakage)

    # full-res pixel (our center-at-0 convention) of every grid sample
    gys, gxs = np.meshgrid(np.arange(0, hm, STRIDE), np.arange(0, wm, STRIDE), indexing="ij")
    px = (gxs + 0.5) * W / wm - 0.5
    py = (gys + 0.5) * H / hm - 0.5
    Kinv = np.linalg.inv(K)
    rng = np.random.default_rng(0)

    order = np.argsort(d["obs_image"], kind="stable")
    bounds = np.searchsorted(d["obs_image"][order], np.arange(len(names) + 1))
    XYZ, RGB, report = [], [], {}
    for i, n in enumerate(names):
        c = int(d["cams"][i])
        if n not in train or g.depth[c] is None:
            continue
        D = g.depth[c].reshape(ny, nx)
        V = g.depth_valid[c].reshape(ny, nx) & (D > 0)
        o = order[bounds[i]:bounds[i + 1]]
        if len(o) < args.min_obs:
            continue
        xy = d["obs_xy"][o].astype(np.float64)
        Xc = pts[d["obs_track"][o]] @ w2c[i, :3, :3].T + w2c[i, :3, 3]
        zv = bilinear(np.log(np.where(V, D, 1.0)), V, ((xy[:, 0] + 0.5) * wm / W - 0.5) / STRIDE,
                      ((xy[:, 1] + 0.5) * hm / H - 0.5) / STRIDE)
        m = np.isfinite(zv) & (Xc[:, 2] > 0)
        if m.sum() < args.min_obs:
            continue
        u, v = xy[m, 0] / W - 0.5, xy[m, 1] / H - 0.5
        A, y = design(u, v, zv[m]), np.log(Xc[m, 2])
        hold = rng.random(len(y)) < 0.2
        x_cv = irls(A[~hold], y[~hold])
        err_cv = np.median(np.abs(np.expm1(A[hold] @ x_cv - y[hold])))
        x_scale = irls(A[~hold][:, 1:2], y[~hold] - A[~hold][:, 0])           # scale-only baseline
        err_scale = np.median(np.abs(np.expm1(A[hold][:, 1:2] @ x_scale + A[hold][:, 0] - y[hold])))
        report[n] = dict(obs=int(m.sum()), err_poly=float(err_cv), err_scale_only=float(err_scale))
        if err_cv > args.max_fit_err:
            report[n]["rejected"] = True
            continue
        x = irls(A, y)
        # lift confident samples; keep the correction inside the range seen at the tracks (no wild extrapolation)
        uu, vv = px[V] / W - 0.5, py[V] / H - 0.5
        corr = design(uu, vv, np.log(D[V])) @ x - np.log(D[V])
        seen = y - A[:, 0]
        lo, hi = np.percentile(seen, 1) - 0.05, np.percentile(seen, 99) + 0.05
        keep = (corr >= lo) & (corr <= hi)
        z = np.exp(corr[keep] + np.log(D[V][keep]))
        rays = np.c_[px[V][keep], py[V][keep], np.ones(keep.sum())] @ Kinv.T
        Pc = rays * z[:, None]
        R, t = w2c[i, :3, :3], w2c[i, :3, 3]
        XYZ.append((Pc - t) @ R)
        im = np.asarray(Image.open(args.images / n).convert("RGB"))
        s = im.shape[1] / W
        cx = np.clip(np.round((px[V][keep] + 0.5) * s - 0.5).astype(int), 0, im.shape[1] - 1)
        cy = np.clip(np.round((py[V][keep] + 0.5) * s - 0.5).astype(int), 0, im.shape[0] - 1)
        RGB.append(im[cy, cx])
    XYZ, RGB = np.concatenate(XYZ), np.concatenate(RGB)
    errs = np.array([r["err_poly"] for r in report.values()])
    errs_s = np.array([r["err_scale_only"] for r in report.values()])
    print(f"{len(report)} train images fitted, {sum('rejected' in r for r in report.values())} rejected; held-out "
          f"median rel depth err: poly {np.median(errs):.4f} vs scale-only {np.median(errs_s):.4f}; "
          f"{len(XYZ)} lifted VGGT points", flush=True)

    # voxel-downsample the VGGT points so that tracks + VGGT ~= target
    base_ply = args.base / "sparse/0/points3D.ply"
    raw = base_ply.read_bytes()
    k = raw.index(b"end_header\n") + len(b"end_header\n")
    tv = np.frombuffer(raw[k:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    want = args.target - len(tv)
    if len(XYZ) > want:
        lo, hi = 1e-4, 10.0
        for _ in range(30):
            vs = np.sqrt(lo * hi)
            nv = len(np.unique(np.floor(XYZ / vs).astype(np.int64), axis=0))
            lo, hi = (vs, hi) if nv > want else (lo, vs)
        perm = rng.permutation(len(XYZ))
        _, first = np.unique(np.floor(XYZ[perm] / hi).astype(np.int64), axis=0, return_index=True)
        sel = perm[first]
        print(f"voxel {hi * 100:.1f} cm -> {len(sel)} VGGT points", flush=True)
        XYZ, RGB = XYZ[sel], RGB[sel]
    allx = np.concatenate([np.c_[tv["x"], tv["y"], tv["z"]], XYZ])
    allc = np.concatenate([np.c_[tv["r"], tv["g"], tv["b"]], RGB])

    sp = args.out / "sparse/0"
    sp.mkdir(parents=True, exist_ok=True)
    for f in ("cameras.bin", "images.bin"):
        shutil.copy(args.base / "sparse/0" / f, sp / f)
    if not (args.out / "images").exists():
        (args.out / "images").symlink_to(args.images.resolve())
    write_points_ply(sp / "points3D.ply", allx, allc.astype(np.uint8))
    (args.out / "densify_report.json").write_text(json.dumps(report, indent=1))
    print(f"{args.out}: {len(tv)} track + {len(XYZ)} VGGT = {len(allx)} points ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
