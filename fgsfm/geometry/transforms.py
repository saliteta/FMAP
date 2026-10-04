"""SE(3)/Sim(3) helpers. Poses are 4x4 world-to-camera (OpenCV), numpy float64.

A Sim(3) S = (s, R, t) maps points of frame B into frame A: X_A = s R X_B + t.
Applied to a camera:  R_A = R_B R^T,  C_A = s R C_B + t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sim3:
    s: float
    R: np.ndarray
    t: np.ndarray

    @classmethod
    def identity(cls) -> Sim3:
        return cls(1.0, np.eye(3), np.zeros(3))

    def apply_points(self, X: np.ndarray) -> np.ndarray:
        return self.s * X @ self.R.T + self.t

    def apply_w2c(self, w2c: np.ndarray) -> np.ndarray:
        """w2c in frame B (..., 4, 4) -> w2c in frame A."""
        R_b, C_b = w2c[..., :3, :3], centers(w2c)
        R_a = R_b @ self.R.T
        C_a = self.s * C_b @ self.R.T + self.t
        return make_w2c(R_a, C_a)

    def compose(self, other: Sim3) -> Sim3:
        """(self ∘ other)(X) = self(other(X))."""
        return Sim3(self.s * other.s, self.R @ other.R, self.s * self.R @ other.t + self.t)

    def inverse(self) -> Sim3:
        Ri = self.R.T
        return Sim3(1.0 / self.s, Ri, -(Ri @ self.t) / self.s)


def centers(w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[..., :3, :3], w2c[..., :3, 3]
    return -np.einsum("...ji,...j->...i", R, t)


def make_w2c(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    out = np.zeros(R.shape[:-2] + (4, 4))
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", R, C)
    out[..., 3, 3] = 1.0
    return out


def to4x4(w2c34: np.ndarray) -> np.ndarray:
    out = np.zeros(w2c34.shape[:-2] + (4, 4))
    out[..., :3, :] = w2c34[..., :3, :]
    out[..., 3, 3] = 1.0
    return out


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    cos = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def chordal_mean(Rs: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(Rs)) if w is None else w
    return project_to_so3(np.einsum("n,nij->ij", w, Rs))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim(3) with dst ≈ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / max((xs ** 2).sum() / len(src), 1e-12)) if with_scale else 1.0
    return Sim3(s, R, mu_d - s * R @ mu_s)


@dataclass
class Sim3Fit:
    sim3: Sim3
    inliers: np.ndarray          # bool per anchor
    rot_res_deg: np.ndarray      # per anchor
    center_res: np.ndarray       # per anchor, in target-frame units
    scale_spread: float          # robust spread of per-pixel log depth ratios

    @property
    def num_inliers(self) -> int:
        return int(self.inliers.sum())


def fit_sim3_from_cameras(w2c_dst: np.ndarray, w2c_src: np.ndarray, log_depth_ratios: np.ndarray | None,
                          rot_thresh_deg: float = 5.0, center_thresh: float | None = None) -> Sim3Fit:
    """Sim(3) taking frame `src` to frame `dst` from cameras known in both.

    rotation:    chordal mean of R_dst^T R_src (each camera gives one estimate)
    scale:       median per-pixel log depth ratio (dst/src) on the shared images when given,
                 otherwise from camera-center spread
    translation: median of C_dst - s R C_src
    One round of outlier rejection on rotation / center residuals.
    """
    n = len(w2c_dst)
    Rd, Rs = w2c_dst[:, :3, :3], w2c_src[:, :3, :3]
    Cd, Cs = centers(w2c_dst), centers(w2c_src)
    inl = np.ones(n, bool)
    for _ in range(2):
        R = chordal_mean(np.einsum("nji,njk->nik", Rd[inl], Rs[inl]))
        if log_depth_ratios is not None and len(log_depth_ratios) > 0:
            s = float(np.exp(np.median(log_depth_ratios)))
        elif inl.sum() >= 2:
            s = umeyama(Cs[inl], Cd[inl]).s
        else:
            s = 1.0
        t = np.median(Cd[inl] - s * Cs[inl] @ R.T, axis=0)
        sim = Sim3(s, R, t)
        rot_res = rotation_angle_deg(np.einsum("nij,nkj->nik", Rd, Rs @ R.T))
        c_res = np.linalg.norm(Cd - sim.apply_points(Cs), axis=1)
        thr_c = center_thresh if center_thresh is not None else 3.0 * max(np.median(c_res), 1e-9)
        new_inl = (rot_res <= rot_thresh_deg) & (c_res <= thr_c)
        if new_inl.sum() == 0 or (new_inl == inl).all():
            inl = new_inl if new_inl.sum() else inl
            break
        inl = new_inl
    spread = float(np.median(np.abs(log_depth_ratios - np.median(log_depth_ratios)))) \
        if log_depth_ratios is not None and len(log_depth_ratios) else float("nan")
    return Sim3Fit(sim, inl, rot_res, c_res, spread)
