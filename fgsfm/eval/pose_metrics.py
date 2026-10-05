"""Pose metrics of an estimated (sub)reconstruction against a reference (README §8)."""
from __future__ import annotations

import numpy as np

from fgsfm.geometry.transforms import Sim3, centers, robust_umeyama, rotation_angle_deg


def robust_sim3_align(C_est: np.ndarray, C_ref: np.ndarray, iters: int = 5, keep: float = 0.8) -> Sim3:
    """Umeyama on camera centers with iterative trimming of the worst (1-keep) fraction."""
    return robust_umeyama(C_est, C_ref, iters, keep)


def pose_metrics(w2c_est: np.ndarray, w2c_ref: np.ndarray, max_pairs: int = 50000, seed: int = 0) -> dict:
    """w2c_est, w2c_ref: (N, 4, 4) for the same N cameras (est in any frame)."""
    n = len(w2c_est)
    if n < 3:
        return dict(num_cameras=n)
    C_est, C_ref = centers(w2c_est), centers(w2c_ref)
    sim = robust_sim3_align(C_est, C_ref)
    aligned = sim.apply_w2c(w2c_est)
    ate = np.linalg.norm(centers(aligned) - C_ref, axis=1)
    abs_rot = rotation_angle_deg(np.einsum("nij,nkj->nik", aligned[:, :3, :3], w2c_ref[:, :3, :3]))

    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(n, k=1)
    if len(i) > max_pairs:
        s = rng.choice(len(i), max_pairs, replace=False)
        i, j = i[s], j[s]

    def rel(T, a, b):
        return T[b] @ np.linalg.inv(T[a])
    Rp, Rr = rel(w2c_est, i, j), rel(w2c_ref, i, j)
    rre = rotation_angle_deg(np.einsum("nji,njk->nik", Rp[:, :3, :3], Rr[:, :3, :3]))
    tp, tr = Rp[:, :3, 3], Rr[:, :3, 3]
    cos = np.sum(tp * tr, -1) / (np.linalg.norm(tp, axis=-1) * np.linalg.norm(tr, axis=-1) + 1e-12)
    rte = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    err = np.maximum(rre, rte)
    auc = {f"AUC@{d}": float(np.mean([(err <= x).mean() for x in np.linspace(0, d, 200)])) for d in (3, 5, 15, 30)}
    return dict(num_cameras=n, ATE_rmse=float(np.sqrt((ate ** 2).mean())), ATE_median=float(np.median(ate)),
                abs_rot_median_deg=float(np.median(abs_rot)), RRA_5=float((rre <= 5).mean()),
                RTA_5=float((rte <= 5).mean()), rel_rot_median_deg=float(np.median(rre)),
                rel_trans_dir_median_deg=float(np.median(rte)), **auc,
                cameras_ate_lt_5m=float((ate < 5.0).mean()))
