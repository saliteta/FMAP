"""Pairwise overlap from one FM batch by depth reprojection (README Stage 4 definition).

covis_{i->j} = fraction of confident pixels of i that, lifted with predicted
depth and moved into j with predicted poses, land inside j with a depth that
agrees with j's predicted depth. O_ij = min(covis_{i->j}, covis_{j->i}).

This is *derived* evidence (same prediction as the poses): good for scoring
and proposing edges, never sufficient to verify one.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BatchGeometry:
    ys: torch.Tensor          # (P,) sampled pixel rows (model resolution)
    xs: torch.Tensor          # (P,) sampled pixel cols
    Xw: torch.Tensor          # (S, P, 3) lifted points in the batch frame
    conf: torch.Tensor        # (S, P) depth confidence
    valid: torch.Tensor       # (S, P) confident source pixels
    hit: torch.Tensor         # (S_src, S_tgt, P) consistent reprojection into target
    uv: torch.Tensor          # (S_src, S_tgt, P, 2) target pixel coords
    covis: torch.Tensor       # (S, S) directional covisibility, row = source


@torch.inference_mode()
def batch_geometry(extrinsic: torch.Tensor, intrinsic: torch.Tensor, depth: torch.Tensor,
                   conf: torch.Tensor, stride: int = 4, conf_quantile: float = 0.3,
                   rel_depth_tol: float = 0.05) -> BatchGeometry:
    S, H, W = depth.shape
    dev = depth.device
    ys, xs = torch.meshgrid(torch.arange(0, H, stride, device=dev),
                            torch.arange(0, W, stride, device=dev), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs, dtype=torch.float32)], -1).float()  # (P, 3)

    d = depth[:, ys, xs]                                             # (S, P)
    c = conf[:, ys, xs]
    thr = torch.quantile(c, conf_quantile, dim=1, keepdim=True)
    valid = (c >= thr) & (d > 0)

    R, t = extrinsic[:, :, :3], extrinsic[:, :, 3]                    # w2c
    rays = pix @ torch.linalg.inv(intrinsic).transpose(1, 2)          # (S, P, 3) camera rays
    Xc = rays * d[..., None]
    Xw = torch.einsum("sij,spj->spi", R.transpose(1, 2), Xc - t[:, None])   # (S, P, 3)

    # project every source's points into every target: (S_src, S_tgt, P, 3)
    Xt = torch.einsum("tij,spj->stpi", R, Xw) + t[None, :, None]
    z = Xt[..., 2]
    uvw = torch.einsum("tij,stpj->stpi", intrinsic, Xt)
    u, v = uvw[..., 0] / z.clamp(min=1e-6), uvw[..., 1] / z.clamp(min=1e-6)
    inside = (z > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)

    ui = u.clamp(0, W - 1).round().long()
    vi = v.clamp(0, H - 1).round().long()
    tgt_idx = torch.arange(S, device=dev)[None, :, None].expand_as(ui)
    d_tgt = depth[tgt_idx, vi, ui]
    consistent = (z - d_tgt).abs() <= rel_depth_tol * d_tgt

    hit = inside & consistent & valid[:, None, :]
    covis = hit.sum(-1).float() / valid.sum(-1, keepdim=True).clamp(min=1).float()
    return BatchGeometry(ys, xs, Xw, c, valid, hit, torch.stack([u, v], -1), covis)


def reprojection_overlap(extrinsic, intrinsic, depth, conf, stride: int = 4, conf_quantile: float = 0.3,
                         rel_depth_tol: float = 0.05) -> torch.Tensor:
    """Returns (S, S) directional covisibility, row i = source image."""
    return batch_geometry(extrinsic, intrinsic, depth, conf, stride, conf_quantile, rel_depth_tol).covis


def symmetric_overlap(covis: torch.Tensor) -> torch.Tensor:
    return torch.minimum(covis, covis.T)
