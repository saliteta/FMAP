"""Depth-guided matching: VGGT depth + initial poses predict where a keypoint lands in the other image.

For a pair (i, j), each keypoint of i with a VGGT depth is lifted to 3D and projected into j with the
initial (graph) poses. Descriptor comparison is restricted to j's keypoints within `radius` pixels of
that prediction (the k nearest in position); a ratio test among those local candidates, run in both
directions and kept mutual, gives the matches. Geometric verification is still MAGSAC on the image
matches (the prior only guides, README §3). Pairs with too few guided inliers fall back to global matching.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch



@torch.inference_mode()
def _guided_one_way(pred: torch.Tensor, ok: torch.Tensor, dq: torch.Tensor, xy_t: torch.Tensor,
                    dt: torch.Tensor, radius: float, ratio: float, k: int) -> torch.Tensor:
    """Best target index per query keypoint (-1 if none passes), searching near pred."""
    out = torch.full((len(pred),), -1, dtype=torch.long, device=pred.device)
    q = torch.nonzero(ok).squeeze(1)
    if len(q) == 0 or len(xy_t) < 2:
        return out
    for chunk in torch.split(q, 4096):
        dpos = torch.cdist(pred[chunk], xy_t)                            # (n, Nt) pixel distances
        kk = min(k, len(xy_t))
        dist_k, cand = dpos.topk(kk, dim=1, largest=False)               # nearest candidates by position
        sim = torch.einsum("nd,nkd->nk", dq[chunk], dt[cand])            # descriptor similarity
        dd = (2 - 2 * sim).clamp(min=0).sqrt()
        dd = torch.where(dist_k <= radius, dd, torch.full_like(dd, 9.0))
        v2, i2 = dd.topk(2, dim=1, largest=False)
        good = (v2[:, 0] < 9.0) & (v2[:, 0] < ratio * v2[:, 1])
        best = cand.gather(1, i2[:, :1]).squeeze(1)
        out[chunk[good]] = best[good]
    return out


def guided_match_pairs(feats: list[dict], pairs: list[tuple[int, int]], w2c: np.ndarray, K: np.ndarray,
                       depth_of, radius: float = 150.0, ratio: float = 0.85, k: int = 32,
                       ransac_px: float = 4.0, min_inliers: int = 30, fallback: bool = True,
                       device: str = "cuda", workers: int = 8):
    """depth_of(i, xy (N,2) full-res px) -> (N,) metric z-depth, NaN where unknown.
    Returns ({(i,j): (M,2) feature index pairs}, stats)."""
    W, H = feats[0]["wh"]
    Kinv = np.linalg.inv(K)
    xy = [torch.as_tensor(f["xy"].astype(np.float32), device=device) for f in feats]
    desc = [torch.as_tensor(f["desc"].astype(np.float32), device=device) for f in feats]
    rays, depth = {}, {}

    def lift(i):
        if i not in rays:
            d = depth_of(i, feats[i]["xy"])
            r = np.c_[feats[i]["xy"], np.ones(len(d))] @ Kinv.T
            rays[i] = (r * d[:, None]) if len(d) else np.zeros((0, 3))       # camera-frame points
            depth[i] = np.isfinite(d) & (d > 0)
        return rays[i], depth[i]

    def predict(i, j):
        Xc, ok = lift(i)
        Ri, ti, Rj, tj = w2c[i, :3, :3], w2c[i, :3, 3], w2c[j, :3, :3], w2c[j, :3, 3]
        Xj = (np.nan_to_num(Xc) - ti) @ Ri @ Rj.T + tj                      # camera i -> world -> camera j
        p = Xj @ K.T
        uv = p[:, :2] / np.maximum(p[:, 2:], 1e-9)
        ok = ok & (Xj[:, 2] > 0) & (uv[:, 0] > -radius) & (uv[:, 0] < W + radius) & \
            (uv[:, 1] > -radius) & (uv[:, 1] < H + radius)
        return torch.as_tensor(uv.astype(np.float32), device=device), torch.as_tensor(ok, device=device)

    raw, stats = {}, dict(guided=0, fallback=0)
    for i, j in pairs:
        pij, okij = predict(i, j)
        pji, okji = predict(j, i)
        a = _guided_one_way(pij, okij, desc[i], xy[j], desc[j], radius, ratio, k)
        b = _guided_one_way(pji, okji, desc[j], xy[i], desc[i], radius, ratio, k)
        ia = torch.nonzero(a >= 0).squeeze(1)
        mutual = ia[b[a[ia]] == ia]
        raw[(i, j)] = torch.stack([mutual, a[mutual]], 1).cpu().numpy()

    def verify(item):
        (i, j), m = item
        if len(m) < min_inliers:
            return (i, j), None
        F, mask = cv2.findFundamentalMat(feats[i]["xy"][m[:, 0]], feats[j]["xy"][m[:, 1]],
                                         cv2.USAC_MAGSAC, ransac_px, 0.999, 10000)
        if F is None or mask is None or mask.sum() < min_inliers:
            return (i, j), None
        return (i, j), m[mask.ravel().astype(bool)]

    with ThreadPoolExecutor(workers) as ex:
        out = dict(ex.map(verify, raw.items()))
    stats["guided"] = sum(v is not None for v in out.values())
    if fallback:
        failed = [p for p, v in out.items() if v is None]
        if failed:
            from fgsfm.match.matcher import match_pairs
            fb = match_pairs(feats, failed, ratio=ratio, ransac_px=ransac_px, min_inliers=min_inliers, device=device)
            stats["fallback"] = len(fb)
            out.update(fb)
    return {k: v for k, v in out.items() if v is not None}, stats
