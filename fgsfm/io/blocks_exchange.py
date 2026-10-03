"""Reader for ContextCapture / iTwin "BlocksExchange" AT exports (AT-export.xml).

We only need photo id -> image name and, per tie point, the 3D position and
the list of photos that observe it. Those observations are the tracks that
COLMAP's points3D.txt would normally hold, and give ground-truth covisibility.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TiePoints:
    photo_names: dict[int, str]       # photo id -> image basename
    xyz: np.ndarray                   # (P, 3) in the AT's SRS (not the COLMAP frame)
    obs_point: np.ndarray             # (M,) tie-point index of each observation
    obs_photo: np.ndarray             # (M,) photo id of each observation

    def save(self, path: Path) -> None:
        ids = np.array(list(self.photo_names.keys()))
        names = np.array([self.photo_names[i] for i in ids])
        np.savez_compressed(path, ids=ids, names=names, xyz=self.xyz,
                            obs_point=self.obs_point, obs_photo=self.obs_photo)

    @classmethod
    def load_npz(cls, path: Path) -> TiePoints:
        d = np.load(path)
        return cls(dict(zip(d["ids"].tolist(), d["names"].tolist())), d["xyz"],
                   d["obs_point"], d["obs_photo"])


def read_tie_points(xml_path: str | Path, cache: str | Path | None = None) -> TiePoints:
    if cache is not None and Path(cache).exists():
        return TiePoints.load_npz(Path(cache))

    names: dict[int, str] = {}
    xyz, obs_point, obs_photo = [], [], []
    photo_id = None
    for event, el in ET.iterparse(str(xml_path), events=("end",)):
        tag = el.tag
        if tag == "Id" and photo_id is None:
            photo_id = el.text
        elif tag == "ImagePath":
            names[int(photo_id)] = Path(el.text.replace("\\", "/")).name
        elif tag == "Photo":
            photo_id = None
            el.clear()
        elif tag == "TiePoint":
            p = el.find("Position")
            xyz.append([float(p.find(c).text) for c in "xyz"])
            pid = len(xyz) - 1
            for m in el.iterfind("Measurement"):
                obs_point.append(pid)
                obs_photo.append(int(m.find("PhotoId").text))
            el.clear()
        elif tag in ("SRS", "Photogroup"):
            photo_id = None   # Ids outside photos must not leak into the next photo

    tp = TiePoints(names, np.asarray(xyz, np.float64),
                   np.asarray(obs_point, np.int64), np.asarray(obs_photo, np.int64))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        tp.save(Path(cache))
    return tp
