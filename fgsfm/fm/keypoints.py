"""Select a sparse set of reliable 3D keypoints from one FM batch.

Candidates are the sampled pixels of BatchGeometry. A keypoint must be
  - confident (VGGT depth confidence),
  - multi-view consistent: reprojects with agreeing depth into >= min_views-1 other views,
  - textured (image gradient), so it is a "clear feature", not sky / flat roof,
and keypoints are spread by keeping the best candidate per (image, cell).
Observations (image, u, v) come from the same reprojection, i.e. they are
*derived* correspondences, used only for coarse BA (README §3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fgsfm.fm.overlap import BatchGeometry


@dataclass
class Keypoints:
    X: np.ndarray          # (K, 3) in the batch frame
    rgb: np.ndarray        # (K, 3) uint8
    conf: np.ndarray       # (K,)
    src: np.ndarray        # (K,) local image index of the source view
    obs_kp: np.ndarray     # (M,) keypoint index
    obs_img: np.ndarray    # (M,) local image index
    obs_uv: np.ndarray     # (M, 2) pixel coords at model resolution

    def __len__(self) -> int:
        return len(self.X)


def _texture(images: torch.Tensor) -> torch.Tensor:
    gray = images.mean(1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device)[None, None] / 8
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, kx.transpose(-1, -2), padding=1)
    return (gx ** 2 + gy ** 2).sqrt()[:, 0]                       # (S, H, W)


@torch.inference_mode()
def select_keypoints(geom: BatchGeometry, images: torch.Tensor, max_points: int = 1000,
                     min_views: int = 3, cell: int = 32) -> Keypoints:
    S, P = geom.valid.shape
    other = ~torch.eye(S, dtype=torch.bool, device=geom.hit.device)
    hits = geom.hit & other[:, :, None]                           # exclude self
    n_views = 1 + hits.sum(1)                                     # (S, P) incl. source

    tex = _texture(images)[:, geom.ys, geom.xs]                   # (S, P)
    tex_n = (tex / torch.quantile(tex.flatten(), 0.9).clamp(min=1e-6)).clamp(max=1.0)
    ok = geom.valid & (n_views >= min_views) & (tex >= torch.quantile(tex.flatten(), 0.5))
    score = torch.log1p(geom.conf) * n_views.float() * tex_n
    score = torch.where(ok, score, torch.full_like(score, -1.0))

    # best candidate per (image, cell)
    H = int(geom.ys.max()) + 1
    W = int(geom.xs.max()) + 1
    ncx = (W + cell - 1) // cell
    cell_id = (geom.ys // cell) * ncx + (geom.xs // cell)         # (P,)
    n_cells = int(cell_id.max()) + 1
    key = torch.arange(S, device=score.device)[:, None] * n_cells + cell_id[None]   # (S, P)
    flat_key, flat_score = key.flatten(), score.flatten()
    best = torch.full((S * n_cells,), -2.0, device=score.device).scatter_reduce(
        0, flat_key, flat_score, reduce="amax")
    is_best = (flat_score == best[flat_key]) & (flat_score >= 0)
    cand = torch.nonzero(is_best).squeeze(1)
    cand = cand[torch.argsort(flat_score[cand], descending=True)][:max_points]
    s_idx, p_idx = cand // P, cand % P

    X = geom.Xw[s_idx, p_idx]
    rgb = (images[s_idx, :, geom.ys[p_idx], geom.xs[p_idx]] * 255).clamp(0, 255).byte()
    obs_mask = hits[s_idx, :, p_idx]                              # (K, S)
    obs_mask[torch.arange(len(s_idx)), s_idx] = True              # source observation
    kk, ii = torch.nonzero(obs_mask, as_tuple=True)
    uv = geom.uv[s_idx[kk], ii, p_idx[kk]]
    src_px = torch.stack([geom.xs[p_idx] + 0.5, geom.ys[p_idx] + 0.5], -1).float()
    is_src = ii == s_idx[kk]
    uv[is_src] = src_px[kk[is_src]]
    return Keypoints(X.cpu().numpy().astype(np.float64), rgb.cpu().numpy(), geom.conf[s_idx, p_idx].cpu().numpy(),
                     s_idx.cpu().numpy(), kk.cpu().numpy(), ii.cpu().numpy(), uv.cpu().numpy().astype(np.float64))
