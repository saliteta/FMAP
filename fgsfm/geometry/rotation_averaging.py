"""Global rotation averaging over the verified pairs, between matching and triangulation (scripts/run_global_ba.py).

Each stage-B cluster gets its rotation from its own fit to GPS (egr_graph_stage_b.merge_gps); nothing makes
neighbouring clusters agree, and BA cannot repair a block that starts ~10 deg off (its cross-cluster tracks fail
the triangulation filter). Here every verified pair contributes its two-view relative rotation R_ij (essential
matrix on the F-verified inliers), and all camera rotations are solved together (Chatterjee & Govindu 2013):

    c2w Q_i, measurement R_ij ~ R_j R_i^T = Q_j^T Q_i.  Residual m_ij = log(Q_j R_ij Q_i^T); with Q_i <- exp(phi_i) Q_i
    to first order m_ij + phi_j - phi_i, so each iteration solves  phi_i - phi_j = m_ij  (sparse weighted LSQ).

Weights: smoothed L1 first (a whole block off by 10 deg is outvoted by its many consistent boundary pairs, where
a tight robust loss would just drop those pairs as outliers), then Cauchy to sharpen. The result is rotated back
onto the initial rotations by a trimmed chordal mean, so the majority of cameras keep their (GPS-fitted) frame.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation


def relative_rotations(matches: dict, feats: list[dict], K: np.ndarray, thresh_px: float = 1.0,
                       min_inliers: int = 15, workers: int = 8):
    """(pairs (E, 2), R_ij (E, 3, 3), inliers (E,)) with x_j = R_ij x_i + t from the essential matrix."""
    Kinv = np.linalg.inv(K)
    f = float(K[0, 0])

    def one(item):
        (i, j), m = item
        x1 = (np.c_[feats[i]["xy"][m[:, 0]], np.ones(len(m))] @ Kinv.T)[:, :2]
        x2 = (np.c_[feats[j]["xy"][m[:, 1]], np.ones(len(m))] @ Kinv.T)[:, :2]
        try:
            E, mask = cv2.findEssentialMat(x1, x2, np.eye(3), cv2.RANSAC, 0.999, thresh_px / f)
            if E is None:
                return None
            n, R, _, _ = cv2.recoverPose(E[:3], x1, x2, np.eye(3), mask=mask)
        except cv2.error:
            return None
        return (i, j, R, int(n)) if n >= min_inliers else None

    with ThreadPoolExecutor(workers) as ex:
        out = [r for r in ex.map(one, matches.items()) if r is not None]
    pairs = np.array([(i, j) for i, j, _, _ in out])
    return pairs, np.stack([R for _, _, R, _ in out]), np.array([n for *_, n in out])


def _angle(R):
    return np.linalg.norm(Rotation.from_matrix(R).as_rotvec(), axis=-1)


def average_rotations(R0_w2c: np.ndarray, pairs: np.ndarray, Rij: np.ndarray, weight: np.ndarray,
                      l1_iters: int = 15, cauchy_iters: int = 10, cauchy_deg: float = 3.0, log=print):
    """Robust rotation averaging, initialised at R0 (w2c). Returns new w2c rotations (n, 3, 3)."""
    n = len(R0_w2c)
    Q = np.transpose(R0_w2c, (0, 2, 1)).copy()                  # c2w
    i, j = pairs[:, 0], pairs[:, 1]
    E = len(pairs)
    rows = np.repeat(np.arange(3 * E).reshape(E, 3), 2, axis=1).ravel()          # each row: +phi_i, -phi_j
    cols = np.stack([3 * i[:, None] + np.arange(3), 3 * j[:, None] + np.arange(3)], -1).reshape(-1)
    vals = np.tile([1.0, -1.0], 3 * E)
    for it in range(l1_iters + cauchy_iters):
        m = Rotation.from_matrix(Q[j] @ Rij @ np.transpose(Q[i], (0, 2, 1))).as_rotvec()
        r = np.linalg.norm(m, axis=1)
        if it < l1_iters:
            w = weight / np.sqrt(r ** 2 + np.radians(0.5) ** 2)
        else:
            c = np.radians(cauchy_deg)
            w = weight / (1 + (r / c) ** 2)
        sw = np.repeat(np.sqrt(w), 3)
        A = csr_matrix((vals * np.repeat(sw, 2), (rows, cols)), shape=(3 * E, 3 * n))
        phi = lsqr(A, (m.ravel() * sw), atol=1e-10, btol=1e-10, iter_lim=2000)[0].reshape(n, 3)
        Q = Rotation.from_rotvec(phi).as_matrix() @ Q
        step = np.degrees(np.linalg.norm(phi, axis=1).max())
        if it in (0, l1_iters - 1, l1_iters + cauchy_iters - 1) or step < 1e-3:
            log(f"    rotation averaging iter {it}: residual median {np.degrees(np.median(r)):.2f} deg, "
                f"p90 {np.degrees(np.percentile(r, 90)):.2f}, max step {step:.3f} deg")
        if step < 1e-3 and it >= l1_iters:
            break
    # back onto the initial frame: global rotation G with G Q_i ~ Q0_i for the majority (trimmed chordal mean)
    Q0 = np.transpose(R0_w2c, (0, 2, 1))
    D = Q0 @ np.transpose(Q, (0, 2, 1))
    sel = np.arange(n)
    for _ in range(3):
        U, _, Vt = np.linalg.svd(D[sel].sum(0))
        G = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        res = _angle(np.transpose(D, (0, 2, 1)) @ G)
        sel = np.argsort(res)[: int(0.7 * n)]
    return np.transpose(G @ Q, (0, 2, 1))
