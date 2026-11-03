"""No isolated islands: find image groups the pair graph does not connect to the rest, search for overlapping
images with a widening GPS neighbourhood (up to every image), and discard what still cannot be linked.

An island here is any connected component other than the largest one, so linking two islands to each other is
progress too. Two levels use the same loop:

  stage B (scripts/run_stage_ab.py): graph = VGGT-measured overlap >= a floor. A search step runs VGGT-Omega on
      one batch made of the island (up to half the batch) and its next-nearest candidates by GPS; overlap measured
      there links it. The batch is also a cluster for the merge, with the island images preferred for placement,
      so their pose comes from a batch that sees them together with the main block.
  matching (scripts/run_global_ba.py): graph = SIFT-verified pairs. A search step matches the island against its
      next candidates, with a lower inlier floor for these bridge pairs (ground-to-air pairs are weak).

Candidates are ranked by GPS distance from the nearest island image; each step takes the next batch of them, so
the search widens until it has tried every image (exhaustive). Islands that never link are discarded.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def components(adj: np.ndarray, active: np.ndarray) -> list[np.ndarray]:
    """Connected components (index arrays, largest first) of the active nodes of a boolean adjacency."""
    idx = np.flatnonzero(active)
    sub = adj[np.ix_(idx, idx)]
    _, lab = connected_components(csr_matrix(sub | sub.T), directed=False)
    comps = [idx[lab == c] for c in np.unique(lab)]
    return sorted(comps, key=len, reverse=True)


def _ranked_candidates(island: np.ndarray, pos: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Active images outside the island, nearest (to any island image) first."""
    out = np.flatnonzero(active)
    out = out[~np.isin(out, island)]
    d = np.linalg.norm(pos[out][:, None] - pos[island][None], axis=-1).min(1)
    return out[np.argsort(d, kind="stable")]


def repair(adj: np.ndarray, pos: np.ndarray, search: Callable[[np.ndarray, np.ndarray], np.ndarray],
           step: int, max_island_part: int | None = None, min_links: int = 3, log=print):
    """Link or discard every island of the graph `adj` (n x n bool, updated in place through `search`).

    search(island_part, candidates) -> updated adjacency after testing those candidates against the island
    (it must OR its new edges into the matrix it returns). step: candidates per search; max_island_part: island
    images per search (the ones nearest to the candidates). An island is linked once >= min_links images outside
    it have an edge to it. Returns (adj, discarded bool mask, report list)."""
    n = len(adj)
    active = np.ones(n, bool)
    report = []
    while True:
        comps = components(adj, active)
        if len(comps) == 1:
            break
        isl = comps[-1]                                    # smallest first: cheapest, and they may join bigger ones
        cand = _ranked_candidates(isl, pos, active)
        tried, linked = 0, False
        while tried < len(cand):
            nxt = cand[tried:tried + step]
            part = isl
            if max_island_part is not None and len(isl) > max_island_part:
                d = np.linalg.norm(pos[isl][:, None] - pos[nxt][None], axis=-1).min(1)
                part = isl[np.argsort(d)[:max_island_part]]
            adj = search(part, nxt)
            tried += len(nxt)
            outside = np.setdiff1d(np.flatnonzero(active), isl)
            links = int(adj[np.ix_(isl, outside)].any(0).sum() + 0)
            if links >= min_links:
                linked = True
                break
        report.append(dict(images=len(isl), searched=int(tried), of=len(cand), linked=linked))
        log(f"  island of {len(isl)} images: {'linked' if linked else 'DISCARDED'} after searching "
            f"{tried}/{len(cand)} candidates")
        if not linked:
            active[isl] = False
    return adj, ~active, report
