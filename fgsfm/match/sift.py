"""RootSIFT features at (near) full resolution, cached per image (README Stage 6)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _extract(path: Path, max_width: int, max_features: int):
    im = Image.open(path)
    W, H = im.size
    im.draft("L", (max_width, max_width))          # fast JPEG DCT downscale
    im = im.convert("L")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.BICUBIC)
    gray = np.asarray(im)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float16), (W, H)
    xy = np.array([k.pt for k in kps], np.float64)
    xy[:, 0] *= W / gray.shape[1]                   # back to full-resolution pixel coords
    xy[:, 1] *= H / gray.shape[0]
    desc = desc / np.maximum(desc.sum(1, keepdims=True), 1e-6)   # RootSIFT
    desc = np.sqrt(desc).astype(np.float16)
    return xy.astype(np.float32), desc, (W, H)


def extract_features(paths: list[Path], cache_dir: Path, max_width: int = 2736, max_features: int = 8192,
                     workers: int = 8) -> list[dict]:
    """Returns per image {xy (F,2) full-res pixels, desc (F,128) float16, wh}."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(p: Path):
        c = cache_dir / f"{p.stem}_w{max_width}_n{max_features}.npz"
        if c.exists():
            d = np.load(c)
            return dict(xy=d["xy"], desc=d["desc"], wh=tuple(d["wh"]))
        xy, desc, wh = _extract(p, max_width, max_features)
        np.savez(c, xy=xy, desc=desc, wh=np.array(wh))
        return dict(xy=xy, desc=desc, wh=wh)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, paths))
