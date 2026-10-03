"""Minimal COLMAP text-model reader (cameras.txt, images.txt) + points3D.ply."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params[:4]
        elif self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = self.params[0]
            cx, cy = self.params[1:3]
        else:  # OPENCV and friends: fx, fy, cx, cy, ...
            fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray        # (w, x, y, z), world-to-camera rotation
    tvec: np.ndarray        # world-to-camera translation
    camera_id: int
    name: str

    @property
    def R(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.tvec


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _data_lines(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    cams = {}
    for line in _data_lines(path):
        e = line.split()
        cams[int(e[0])] = ColmapCamera(int(e[0]), e[1], int(e[2]), int(e[3]),
                                       np.array(e[4:], dtype=np.float64))
    return cams


def read_images_text(path: Path) -> dict[int, ColmapImage]:
    """images.txt has two lines per image; the 2D-point line may be empty, which
    _data_lines drops, so only the pose lines are parsed (tracks are not needed here)."""
    imgs = {}
    for line in _data_lines(path):
        e = line.split()
        if len(e) != 10 or not e[9].lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue  # a POINTS2D line
        imgs[int(e[0])] = ColmapImage(int(e[0]), np.array(e[1:5], float), np.array(e[5:8], float),
                                      int(e[8]), e[9])
    return imgs


def read_points_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


@dataclass
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points_xyz: np.ndarray | None
    points_rgb: np.ndarray | None

    @classmethod
    def load(cls, sparse_dir: str | Path) -> ColmapModel:
        d = Path(sparse_dir)
        xyz = rgb = None
        if (d / "points3D.ply").exists():
            xyz, rgb = read_points_ply(d / "points3D.ply")
        return cls(read_cameras_text(d / "cameras.txt"), read_images_text(d / "images.txt"), xyz, rgb)

    def sorted_images(self) -> list[ColmapImage]:
        return sorted(self.images.values(), key=lambda im: im.name)
