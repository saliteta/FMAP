"""Dense view graph used for initial construction.

Starts as an O(N^2) graph with no correlation (all scores 0, nothing
evaluated). Scores are filled pair by pair (or batch by batch) by a scorer,
or all at once from a reference model's tracks. Sparsification to O(Nk)
happens later (Stage 4, degree cap).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fgsfm.core.camera import CameraNode


class ViewGraph:
    def __init__(self, cameras: list[CameraNode]):
        n = len(cameras)
        self.cameras = cameras
        self.scores = np.zeros((n, n), dtype=np.float32)      # symmetric relation score
        self.evaluated = np.zeros((n, n), dtype=bool)         # has this pair been scored yet
        np.fill_diagonal(self.scores, 1.0)
        np.fill_diagonal(self.evaluated, True)
        for i, cam in enumerate(cameras):
            assert cam.image_id == i, "image_id must equal index in the graph"
            cam.connections = self.scores[i]                  # row view, stays in sync

    @classmethod
    def from_image_paths(cls, paths: list[str | Path]) -> ViewGraph:
        return cls([CameraNode(image_id=i, name=Path(p).name, image_path=Path(p))
                    for i, p in enumerate(paths)])

    @classmethod
    def from_colmap(cls, model, image_dir: str | Path | None = None) -> ViewGraph:
        """Cameras with reference poses from a COLMAP model, sorted by image name; no scores yet."""
        cams = []
        for i, im in enumerate(model.sorted_images()):
            cam = model.cameras[im.camera_id]
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = im.R, im.tvec
            cams.append(CameraNode(image_id=i, name=im.name,
                                   image_path=Path(image_dir) / im.name if image_dir else None,
                                   pose_w2c=w2c, K=cam.K, width=cam.width, height=cam.height))
        return cls(cams)

    def __len__(self) -> int:
        return len(self.cameras)

    def index_of(self, name: str) -> int:
        if not hasattr(self, "_name_to_idx"):
            self._name_to_idx = {c.name: c.image_id for c in self.cameras}
        return self._name_to_idx[name]

    def update_pair(self, i: int, j: int, score: float) -> None:
        self.scores[i, j] = self.scores[j, i] = score
        self.evaluated[i, j] = self.evaluated[j, i] = True

    def set_all(self, scores: np.ndarray) -> None:
        """Fill every pair at once (e.g. from reference covisibility)."""
        self.scores[:] = scores
        self.evaluated[:] = True

    def unevaluated_pairs(self) -> list[tuple[int, int]]:
        iu, ju = np.triu_indices(len(self), k=1)
        mask = ~self.evaluated[iu, ju]
        return list(zip(iu[mask].tolist(), ju[mask].tolist()))

    def edges(self, threshold: float) -> list[tuple[int, int, float]]:
        iu, ju = np.triu_indices(len(self), k=1)
        s = self.scores[iu, ju]
        keep = self.evaluated[iu, ju] & (s >= threshold)
        return [(int(a), int(b), float(c)) for a, b, c in zip(iu[keep], ju[keep], s[keep])]

    def save(self, path: str | Path) -> None:
        np.savez(path, scores=self.scores, evaluated=self.evaluated,
                 names=np.array([c.name for c in self.cameras]))
