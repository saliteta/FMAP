"""Batch proposal for FM inference.

spatial: k-nearest cameras by position (a stand-in for GPS) around seeds picked
         by farthest-point sampling until every camera is in >= 1 batch.
random:  uniformly random batches, mostly non-overlapping pairs (false-positive probe).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def spatial_knn_batches(centers: np.ndarray, batch_size: int, rng: np.random.Generator,
                        min_coverage: int = 1) -> list[np.ndarray]:
    n = len(centers)
    tree = cKDTree(centers)
    covered = np.zeros(n, int)
    batches = []
    dist_to_seeds = np.full(n, np.inf)
    seed = int(rng.integers(n))
    while covered.min() < min_coverage:
        _, nn = tree.query(centers[seed], k=min(batch_size, n))
        batches.append(np.sort(nn))
        covered[nn] += 1
        dist_to_seeds = np.minimum(dist_to_seeds, np.linalg.norm(centers - centers[seed], axis=1))
        # farthest uncovered camera becomes the next seed
        cand = np.where(covered < min_coverage, dist_to_seeds, -1)
        seed = int(np.argmax(cand))
    return batches


def random_batches(n: int, batch_size: int, num_batches: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [np.sort(rng.choice(n, batch_size, replace=False)) for _ in range(num_batches)]
