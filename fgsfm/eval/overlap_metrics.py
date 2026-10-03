"""Metrics for predicted overlap vs reference covisibility, and relative-pose accuracy."""
from __future__ import annotations

import numpy as np


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    # average ties
    xs = x[order]
    bounds = np.flatnonzero(np.diff(xs)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(x)]):
        ranks[order[a:b]] = 0.5 * (a + b - 1)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def pr_curve(score: np.ndarray, label: np.ndarray):
    """Precision/recall at every threshold, descending score."""
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], label[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    keep = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]       # last index of each distinct score
    tp, fp, thr = tp[keep], fp[keep], s[keep]
    precision = tp / (tp + fp)
    recall = tp / max(y.sum(), 1)
    return precision, recall, thr, fp / max((1 - y).sum(), 1)


def binary_metrics(score: np.ndarray, label: np.ndarray) -> dict:
    p, r, thr, fpr = pr_curve(score, label)
    ap = float(np.sum(np.diff(np.r_[0, r]) * p))
    roc_auc = float(np.trapezoid(np.r_[0, r], np.r_[0, fpr]))
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    k = int(np.argmax(f1))
    out = dict(average_precision=ap, roc_auc=roc_auc, best_f1=float(f1[k]),
               best_f1_threshold=float(thr[k]), best_f1_precision=float(p[k]),
               best_f1_recall=float(r[k]), positives=int(label.sum()), negatives=int((~label).sum()))
    ok = r >= 0.90
    out["precision_at_recall_0.90"] = float(p[ok].max()) if ok.any() else 0.0
    ok = p >= 0.98
    out["recall_at_precision_0.98"] = float(r[ok].max()) if ok.any() else 0.0
    return out


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    cos = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def relative_pose_errors(w2c_pred: np.ndarray, w2c_ref: np.ndarray):
    """All ordered pairs i<j within one batch. Inputs (S, 3|4, 4). Returns (rot_err, trans_dir_err, i, j)."""
    def rel(T):
        T4 = np.tile(np.eye(4), (len(T), 1, 1))
        T4[:, :3, :] = T[:, :3, :]
        return T4[None, :] @ np.linalg.inv(T4)[:, None]          # [i, j] = T_j @ T_i^-1  (i -> j)
    Rp, Rr = rel(w2c_pred), rel(w2c_ref)
    i, j = np.triu_indices(len(w2c_pred), k=1)
    rot = rotation_angle_deg(Rp[i, j, :3, :3].transpose(0, 2, 1) @ Rr[i, j, :3, :3])
    tp, tr = Rp[i, j, :3, 3], Rr[i, j, :3, 3]
    cos = np.sum(tp * tr, -1) / (np.linalg.norm(tp, axis=-1) * np.linalg.norm(tr, axis=-1) + 1e-12)
    trans = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return rot, trans, i, j


def pose_auc(rot_err: np.ndarray, trans_err: np.ndarray, max_deg: float) -> float:
    """VGGT/RealEstate-style AUC of the pose-accuracy curve up to max_deg (error = max of rot / trans-dir)."""
    err = np.maximum(rot_err, trans_err)
    t = np.linspace(0, max_deg, 1000)
    return float(np.mean([(err <= x).mean() for x in t]))
