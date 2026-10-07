"""GPU learned local features (SuperPoint / ALIKED via LightGlue), same output format as match/sift.py.

Images are decoded at the target width in CPU threads (JPEG DCT downscale) while the GPU extracts.
Keypoints are returned in full-resolution pixel coordinates; descriptors are L2-normalized float16.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _decode(path: Path, width: int):
    im = Image.open(path)
    W, H = im.size
    im.draft("RGB", (width, width * H // W))
    im = im.convert("RGB")
    if im.width != width:
        im = im.resize((width, round(im.height * width / im.width)), Image.BILINEAR)
    return np.asarray(im), (W, H)


@torch.inference_mode()
def extract_features_learned(paths: list[Path], cache_dir: Path, kind: str = "superpoint", width: int = 2736,
                             max_features: int = 8192, device: str = "cuda", workers: int = 8) -> list[dict]:
    from lightglue import ALIKED, SuperPoint
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfile = lambda p: cache_dir / f"{p.stem}_{kind}_w{width}_n{max_features}.npz"
    todo = [p for p in paths if not cfile(p).exists()]
    if todo:
        model = {"superpoint": SuperPoint, "aliked": ALIKED}[kind](max_num_keypoints=max_features).eval().to(device)
        with ThreadPoolExecutor(workers) as ex:
            for p, (img, (W, H)) in zip(todo, ex.map(lambda p: _decode(p, width), todo)):
                x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None].to(device)
                f = model({"image": x})
                xy = f["keypoints"][0].cpu().numpy().astype(np.float64)
                xy[:, 0] *= W / img.shape[1]
                xy[:, 1] *= H / img.shape[0]
                desc = torch.nn.functional.normalize(f["descriptors"][0], dim=-1).cpu().numpy().astype(np.float16)
                np.savez(cfile(p), xy=xy.astype(np.float32), desc=desc, wh=np.array([W, H]))
        del model
        torch.cuda.empty_cache()
    out = []
    for p in paths:
        d = np.load(cfile(p))
        out.append(dict(xy=d["xy"], desc=d["desc"], wh=tuple(d["wh"])))
    return out
