"""Camera node: an image in the view graph.

The pose is optional. At graph-building time we only need the connection
scores to the other cameras; poses may come from a reference model (COLMAP)
or later from the FM module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraNode:
    image_id: int                           # index in the ViewGraph
    name: str = ""
    image_path: Path | None = None
    pose_w2c: np.ndarray | None = None      # 4x4 world-to-camera (COLMAP / OpenCV convention)
    K: np.ndarray | None = None             # 3x3 intrinsics
    width: int | None = None
    height: int | None = None
    connections: np.ndarray | None = None   # (N,) relation score to every camera; view into ViewGraph.scores

    @property
    def has_pose(self) -> bool:
        return self.pose_w2c is not None

    @property
    def pose_c2w(self) -> np.ndarray:
        return np.linalg.inv(self.pose_w2c)

    @property
    def center(self) -> np.ndarray:
        return self.pose_c2w[:3, 3]

    def score_to(self, other: int) -> float:
        return float(self.connections[other])

    def top_k(self, k: int) -> list[tuple[int, float]]:
        """Strongest k connections (excluding self), highest first."""
        s = self.connections.astype(np.float64)
        s[self.image_id] = -np.inf
        idx = np.argsort(-s)[:k]
        return [(int(j), float(s[j])) for j in idx if np.isfinite(s[j])]
