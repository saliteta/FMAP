"""Dense 3DGS initial points from VGGT depth, corrected per image by the final BA tracks.

The stored VGGT depth samples of an image (stride-4 grid of the model image, confident pixels only) are
distorted relative to the BA geometry: wrong global scale, depth-dependent scale and a smooth image-space
warp. The BA track observations in that image give exact depths at sparse pixels, so per image we fit

    log z_BA = a * log z_VGGT + poly2(u, v)          (robust IRLS, Huber)

and lift every confident sample with the corrected depth and the BA pose / intrinsics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

STRIDE = 4                     # sample grid of the stored VGGT depth (fgsfm/fm/overlap.py)


def _design(u, v, logz):
    """Columns of the correction model: a*log z_VGGT + 2nd-order polynomial in normalized pixel coords."""
    return np.stack([logz, np.ones_like(u), u, v, u * u, u * v, v * v], 1)


def _irls(A, y, delta=0.05, iters=10):
    w = np.ones(len(y))
    for _ in range(iters):
        sw = np.sqrt(w)
        x = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(A @ x - y)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return x


def _bilinear(grid, valid, gx, gy):
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


def lift_vggt_points(ba: dict, graph, use_names: set[str], color_dir: Path, min_obs: int = 50,
                     max_fit_err: float = 0.03, seed: int = 0):
    """Corrected VGGT points of the images in use_names.

    ba: ba_result.npz contents (run_global_ba.py); graph: SceneGraph of the checkpoint the BA started from;
    color_dir: downscaled images to color from. Returns (xyz (N, 3), rgb (N, 3) uint8, per-image report).
    """
    names = [str(n) for n in ba["names"]]
    W, H = (int(v) for v in ba["image_wh"])
    w2c, K, pts = ba["w2c"], ba["K"], ba["points"]
    hm, wm = graph.image_hw
    ny, nx = (hm + STRIDE - 1) // STRIDE, (wm + STRIDE - 1) // STRIDE
    gys, gxs = np.meshgrid(np.arange(0, hm, STRIDE), np.arange(0, wm, STRIDE), indexing="ij")
    px = (gxs + 0.5) * W / wm - 0.5                  # full-res pixel, our center-at-0 convention
    py = (gys + 0.5) * H / hm - 0.5
    Kinv = np.linalg.inv(K)
    rng = np.random.default_rng(seed)
    order = np.argsort(ba["obs_image"], kind="stable")
    bounds = np.searchsorted(ba["obs_image"][order], np.arange(len(names) + 1))
    XYZ, RGB, report = [], [], {}
    for i, n in enumerate(names):
        c = int(ba["cams"][i])
        if n not in use_names or graph.depth[c] is None:
            continue
        D = graph.depth[c].reshape(ny, nx)
        V = graph.depth_valid[c].reshape(ny, nx) & (D > 0)
        o = order[bounds[i]:bounds[i + 1]]
        if len(o) < min_obs:
            continue
        xy = ba["obs_xy"][o].astype(np.float64)
        Xc = pts[ba["obs_track"][o]] @ w2c[i, :3, :3].T + w2c[i, :3, 3]
        zv = _bilinear(np.log(np.where(V, D, 1.0)), V, ((xy[:, 0] + 0.5) * wm / W - 0.5) / STRIDE,
                       ((xy[:, 1] + 0.5) * hm / H - 0.5) / STRIDE)
        m = np.isfinite(zv) & (Xc[:, 2] > 0)
        if m.sum() < min_obs:
            continue
        A = _design(xy[m, 0] / W - 0.5, xy[m, 1] / H - 0.5, zv[m])
        y = np.log(Xc[m, 2])
        hold = rng.random(len(y)) < 0.2            # held-out observations measure the fit
        err = np.median(np.abs(np.expm1(A[hold] @ _irls(A[~hold], y[~hold]) - y[hold])))
        report[n] = dict(obs=int(m.sum()), held_out_rel_depth_err=float(err))
        if err > max_fit_err:
            report[n]["rejected"] = True
            continue
        x = _irls(A, y)
        # lift confident samples; keep the correction inside the range seen at the tracks (no wild extrapolation)
        logd = np.log(D[V])
        corr = _design(px[V] / W - 0.5, py[V] / H - 0.5, logd) @ x - logd
        seen = y - A[:, 0]
        keep = (corr >= np.percentile(seen, 1) - 0.05) & (corr <= np.percentile(seen, 99) + 0.05)
        z = np.exp(corr[keep] + logd[keep])
        Pc = np.c_[px[V][keep], py[V][keep], np.ones(keep.sum())] @ Kinv.T * z[:, None]
        XYZ.append((Pc - w2c[i, :3, 3]) @ w2c[i, :3, :3])
        im = np.asarray(Image.open(color_dir / n).convert("RGB"))
        s = im.shape[1] / W
        cx = np.clip(np.round((px[V][keep] + 0.5) * s - 0.5).astype(int), 0, im.shape[1] - 1)
        cy = np.clip(np.round((py[V][keep] + 0.5) * s - 0.5).astype(int), 0, im.shape[0] - 1)
        RGB.append(im[cy, cx])
    if not XYZ:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), report
    return np.concatenate(XYZ), np.concatenate(RGB), report


def voxel_subsample(xyz: np.ndarray, n_max: int, seed: int = 0) -> np.ndarray:
    """Indices of at most n_max points, one per voxel; the voxel size is found by bisection."""
    if len(xyz) <= n_max:
        return np.arange(len(xyz))
    lo, hi = 1e-4, float(np.ptp(xyz, axis=0).max())
    for _ in range(30):
        vs = np.sqrt(lo * hi)
        nv = len(np.unique(np.floor(xyz / vs).astype(np.int64), axis=0))
        lo, hi = (vs, hi) if nv > n_max else (lo, vs)
    perm = np.random.default_rng(seed).permutation(len(xyz))
    _, first = np.unique(np.floor(xyz[perm] / hi).astype(np.int64), axis=0, return_index=True)
    return perm[first]
