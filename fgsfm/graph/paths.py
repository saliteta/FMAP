"""Most reliable camera chains: max-product paths over edge correlations.

w(k) = max over paths s -> k of Π_edges corr(e), e.g. 0.8 * 0.7 beats 0.7 * 0.7;
search stops below min_weight. Used to build multi-hop (span) batches.
"""
from __future__ import annotations

import heapq

import numpy as np


def max_product_paths(corr: np.ndarray, sources: np.ndarray, min_weight: float = 0.2):
    """Multi-source max-product Dijkstra on a dense (n, n) correlation matrix.

    Returns weight (n,), source (n,) (-1 if unreached), parent (n,), hops (n,).
    """
    n = len(corr)
    weight = np.zeros(n)
    src = np.full(n, -1)
    parent = np.full(n, -1)
    hops = np.full(n, -1)
    heap = []
    for s in sources:
        weight[s], src[s], hops[s] = 1.0, s, 0
        heap.append((-1.0, int(s)))
    heapq.heapify(heap)
    done = np.zeros(n, bool)
    C = corr.copy()
    np.fill_diagonal(C, 0.0)
    while heap:
        negw, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        cand = -negw * C[u]
        better = (cand > weight) & (cand >= min_weight) & ~done
        for v in np.flatnonzero(better):
            weight[v], src[v], parent[v], hops[v] = cand[v], src[u], u, hops[u] + 1
            heapq.heappush(heap, (-cand[v], int(v)))
    return weight, src, parent, hops


def path_to(parent: np.ndarray, t: int) -> list[int]:
    path = [t]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    return path[::-1]
