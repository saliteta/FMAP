"""Match only graph edges (O(Nk) pairs): GPU mutual-NN + ratio test, then MAGSAC fundamental RANSAC."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch


@torch.inference_mode()
def _mnn_ratio(d1: torch.Tensor, d2: torch.Tensor, ratio: float) -> np.ndarray:
    sim = d1 @ d2.T                                  # RootSIFT: unit norm -> cosine
    dist = (2 - 2 * sim).clamp(min=0).sqrt()
    v12, i12 = dist.topk(2, dim=1, largest=False)
    i21 = dist.argmin(dim=0)
    a = torch.arange(len(d1), device=d1.device)
    keep = (i21[i12[:, 0]] == a) & (v12[:, 0] < ratio * v12[:, 1])
    return torch.stack([a[keep], i12[keep, 0]], 1).cpu().numpy()


def match_pairs(feats: list[dict], pairs: list[tuple[int, int]], ratio: float = 0.85,
                ransac_px: float = 4.0, min_inliers: int = 30, device: str = "cuda",
                workers: int = 8) -> dict[tuple[int, int], np.ndarray]:
    """Returns {(i, j): (M, 2) feature index pairs} for geometrically verified pairs."""
    desc = [torch.as_tensor(f["desc"].astype(np.float32), device=device) for f in feats]
    raw = {}
    for i, j in pairs:
        if len(desc[i]) < 2 or len(desc[j]) < 2:
            continue
        m = _mnn_ratio(desc[i], desc[j], ratio)
        if len(m) >= min_inliers:
            raw[(i, j)] = m

    def verify(item):
        (i, j), m = item
        p1, p2 = feats[i]["xy"][m[:, 0]], feats[j]["xy"][m[:, 1]]
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, ransac_px, 0.999, 10000)
        if F is None or mask is None:
            return (i, j), None
        inl = mask.ravel().astype(bool)
        return (i, j), (m[inl] if inl.sum() >= min_inliers else None)

    with ThreadPoolExecutor(workers) as ex:
        out = dict(ex.map(verify, raw.items()))
    return {k: v for k, v in out.items() if v is not None}
