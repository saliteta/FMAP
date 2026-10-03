"""Load a GauUscene-style scene: COLMAP poses + AT tie-point tracks -> ViewGraph + covisibility."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fgsfm.graph.covisibility import normalize, shared_point_counts
from fgsfm.graph.view_graph import ViewGraph
from fgsfm.io.blocks_exchange import TiePoints, read_tie_points
from fgsfm.io.colmap_text import ColmapModel


@dataclass
class ReferenceScene:
    root: Path
    graph: ViewGraph
    points_xyz: np.ndarray            # (P, 3) in the COLMAP frame
    points_rgb: np.ndarray            # (P, 3) uint8
    shared: np.ndarray                # (N, N) shared tie-point counts; diag = points per image
    obs_point: np.ndarray             # (M,) point index per observation
    obs_image: np.ndarray             # (M,) graph index per observation

    def scores(self, kind: str = "overlap") -> np.ndarray:
        return normalize(self.shared, kind)


def load_reference_scene(root: str | Path, cache_dir: str | Path = "runs/cache",
                         score: str = "overlap") -> ReferenceScene:
    """root = .../<scene>/colmap_metrics with sparse/0, images/ and AT-export.xml."""
    root = Path(root)
    model = ColmapModel.load(root / "sparse" / "0")
    graph = ViewGraph.from_colmap(model, root / "images")

    scene_name = root.parent.name if root.name == "colmap_metrics" else root.name
    tp: TiePoints = read_tie_points(root / "AT-export.xml",
                                    cache=Path(cache_dir) / f"{scene_name}_tiepoints.npz")
    if model.points_xyz is not None and len(model.points_xyz) == len(tp.xyz):
        xyz, rgb = model.points_xyz, model.points_rgb   # same points, already in COLMAP frame
    else:
        raise ValueError("points3D.ply does not match AT tie points; frame alignment needed")

    photo_to_idx = {pid: graph.index_of(name) for pid, name in tp.photo_names.items()
                    if name in {c.name for c in graph.cameras}}
    keep = np.array([p in photo_to_idx for p in tp.obs_photo])
    obs_point = tp.obs_point[keep]
    obs_image = np.array([photo_to_idx[p] for p in tp.obs_photo[keep]], np.int64)

    shared = shared_point_counts(obs_point, obs_image, len(xyz), len(graph))
    graph.set_all(normalize(shared, score).astype(np.float32))
    return ReferenceScene(root, graph, xyz, rgb, shared, obs_point, obs_image)
