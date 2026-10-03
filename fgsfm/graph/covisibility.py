"""Reference covisibility from SfM tracks.

Given observations (point k seen by image i), the shared-point count is
C = A^T A with A the (points x images) incidence matrix. Normalized scores:

    overlap_ij = C_ij / min(C_ii, C_jj)        fraction of the sparser view that is shared
    jaccard_ij = C_ij / (C_ii + C_jj - C_ij)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

SCORE_TYPES = ("overlap", "jaccard", "shared")


def shared_point_counts(obs_point: np.ndarray, obs_image: np.ndarray,
                        num_points: int, num_images: int) -> np.ndarray:
    A = sp.csr_matrix((np.ones(len(obs_point), np.float32), (obs_point, obs_image)),
                      shape=(num_points, num_images))
    A.data[:] = 1.0                                  # duplicate observations count once
    return (A.T @ A).toarray()


def normalize(counts: np.ndarray, kind: str) -> np.ndarray:
    n = np.diag(counts).astype(np.float64)
    if kind == "shared":
        return counts.astype(np.float64)
    if kind == "overlap":
        denom = np.minimum(n[:, None], n[None, :])
    elif kind == "jaccard":
        denom = n[:, None] + n[None, :] - counts
    else:
        raise ValueError(kind)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, counts / denom, 0.0)
