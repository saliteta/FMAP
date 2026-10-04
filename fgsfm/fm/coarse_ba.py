"""Coarse BA over FM keypoints (README Stage 5, reprojection form).

Jointly refines camera poses and 3D keypoints of one frame from the derived
keypoint observations of all registered batches. Batches that share cameras
are thereby reconciled; refinement batches add the cross-strip constraints.

    min  Σ_obs  w ρ_huber( π(K_c, R_c, C_c, X_p) - uv )  +  λ Σ_c ||C_c - C_c^0||² / σ²
  over  R_c (axis-angle update), C_c, X_p;   origin camera fixed.

The weak center prior fixes the remaining scale gauge and keeps cameras with
few observations from drifting. Intrinsics stay at the FM prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fgsfm.geometry.transforms import centers, make_w2c


def _so3_exp(w: torch.Tensor) -> torch.Tensor:
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = w / theta
    Kx = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    Kx[..., 0, 1], Kx[..., 0, 2] = -k[..., 2], k[..., 1]
    Kx[..., 1, 0], Kx[..., 1, 2] = k[..., 2], -k[..., 0]
    Kx[..., 2, 0], Kx[..., 2, 1] = -k[..., 1], k[..., 0]
    th = theta[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


@dataclass
class CoarseBAResult:
    w2c: np.ndarray
    X: np.ndarray
    obs_keep: np.ndarray
    reproj_before_px: float
    reproj_after_px: float
    num_obs: int
    num_points: int


def coarse_ba(w2c0: np.ndarray, K: np.ndarray, X0: np.ndarray, obs_cam: np.ndarray, obs_pt: np.ndarray,
              obs_uv: np.ndarray, fixed_cam: int, iters: int = 1500, huber_px: float = 2.0,
              prior_weight: float = 1e-2, lr: float = 2e-3, outlier_px: float = 12.0,
              device: str = "cuda") -> CoarseBAResult:
    """w2c0/K indexed by local camera id; obs_cam / obs_pt are local indices."""
    dt = torch.float64
    tt = lambda a: torch.as_tensor(a, dtype=dt, device=device)
    R0, C0 = tt(w2c0[:, :3, :3]), tt(centers(w2c0))
    Kt, uv = tt(K), tt(obs_uv)
    ci, pi = torch.as_tensor(obs_cam, device=device), torch.as_tensor(obs_pt, device=device)

    spacing = np.median(np.linalg.norm(np.diff(centers(w2c0), axis=0), axis=1)) if len(w2c0) > 1 else 1.0
    sigma = max(float(spacing), 1e-6)
    scale = sigma                                   # parameter scaling: translations/points in units of spacing

    w = torch.zeros(len(w2c0), 3, dtype=dt, device=device, requires_grad=True)
    dC = torch.zeros(len(w2c0), 3, dtype=dt, device=device, requires_grad=True)
    dX = torch.zeros(len(X0), 3, dtype=dt, device=device, requires_grad=True)
    X0t = tt(X0)
    free = torch.ones(len(w2c0), 1, dtype=dt, device=device)
    free[fixed_cam] = 0.0

    def residuals():
        R = _so3_exp(w * free) @ R0
        C = C0 + dC * free * scale
        X = X0t + dX * scale
        Xc = torch.einsum("nij,nj->ni", R[ci], X[pi] - C[ci])
        z = Xc[:, 2:3].clamp(min=1e-6)
        proj = torch.einsum("nij,nj->ni", Kt[ci], Xc / z)[:, :2]
        return proj - uv, Xc[:, 2], C

    with torch.no_grad():
        r0, z0, _ = residuals()
        err0 = r0.norm(dim=1)
        keep = (z0 > 0) & (err0 < outlier_px * 4)
    opt = torch.optim.Adam([w, dC, dX], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        if it == iters // 2:                         # re-gate gross outliers once, mid-way
            with torch.no_grad():
                r, z, _ = residuals()
                keep = keep & (z > 0) & (r.norm(dim=1) < outlier_px)
        opt.zero_grad()
        r, _, C = residuals()
        e = r[keep].norm(dim=1)
        hub = torch.where(e < huber_px, 0.5 * e ** 2, huber_px * (e - 0.5 * huber_px))
        prior = ((C - C0) / sigma).pow(2).sum(1)
        loss = hub.mean() + prior_weight * prior.mean()
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        r, _, C = residuals()
        R = _so3_exp(w * free) @ R0
        X = X0t + dX * scale
        e = r.norm(dim=1)
        keep_final = keep & (e < outlier_px)
    return CoarseBAResult(make_w2c(R.cpu().numpy(), C.cpu().numpy()), X.cpu().numpy(), keep_final.cpu().numpy(),
                          float(err0[keep].median()), float(e[keep_final].median()),
                          int(keep_final.sum()), len(X0))
