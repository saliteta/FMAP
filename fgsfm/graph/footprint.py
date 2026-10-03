"""Track-free geometric overlap from poses: ground-plane footprint reprojection.

Rays from a pixel grid of camera i hit a horizontal plane at i's ground height
(median height of the points i observes); the hits are projected into j.
covis_{i->j} = fraction landing inside j in front of it; O_ij = min of both
directions. Ignores relief/occlusion, so it is a geometric upper-bound-ish
reference for nadir blocks, independent of tie-point density.
"""
from __future__ import annotations

import numpy as np


def footprint_overlap(w2c: np.ndarray, K: np.ndarray, hw: tuple[int, int], ground_z: np.ndarray,
                      up: np.ndarray, grid: int = 24) -> np.ndarray:
    n = len(w2c)
    H, W = hw
    up = up / np.linalg.norm(up)
    u, v = np.meshgrid((np.arange(grid) + 0.5) * W / grid, (np.arange(grid) + 0.5) * H / grid)
    pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], -1)
    covis = np.zeros((n, n))
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    C = -np.einsum("nji,nj->ni", R, t)
    Kinv = np.linalg.inv(K)
    for i in range(n):
        d = (pix @ Kinv.T) @ R[i]                          # world ray directions (R^T x)
        denom = d @ up
        s = (ground_z[i] - C[i] @ up) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
        X = C[i] + s[:, None] * d                          # (P, 3), NaN/negative s -> invalid
        ok = np.isfinite(s) & (s > 0)
        Xc = np.einsum("nij,pj->npi", R, X[ok]) + t[:, None]
        z = Xc[..., 2]
        uv = Xc @ K.T
        uu, vv = uv[..., 0] / z, uv[..., 1] / z
        inside = (z > 0) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        covis[i] = inside.sum(1) / max(ok.sum(), 1)
    return np.minimum(covis, covis.T)
