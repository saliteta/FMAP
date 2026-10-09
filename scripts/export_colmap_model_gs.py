"""Export a COLMAP TEXT model (e.g. GLOMAP / COLMAP / A/B outputs) as a gsplat dataset on a fixed image list.

    python scripts/export_colmap_model_gs.py --model runs/x/sparse/0_txt --split runs/gs_HAV_fast/split.json \
        --out runs/gs_x/model

Single shared camera exported as PINHOLE (f, f, cx, cy); a SIMPLE_RADIAL k1 is reported (HAV images are
already undistorted, |k1| ~ 1e-4 -> < 1 px). Points come from the model's points3D.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_gs_datasets import write_cameras_bin, write_images_bin, write_points_ply  # noqa: E402
from fgsfm.io.colmap_text import read_cameras_text, read_images_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="split.json with the image list to use")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", type=Path, default=Path("runs/gs_HAV/images"), help="shared 1600 px images")
    args = ap.parse_args()
    names = json.loads(args.split.read_text())["names"]
    cams = read_cameras_text(args.model / "cameras.txt")
    imgs = {im.name: im for im in read_images_text(args.model / "images.txt").values()}
    missing = [n for n in names if n not in imgs]
    assert not missing, f"{len(missing)} images missing in the model: {missing[:5]}"
    assert len(cams) == 1, "expects one shared camera"
    c = list(cams.values())[0]
    p = c.params
    if c.model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
        k1 = p[3] if c.model == "SIMPLE_RADIAL" else 0.0
    else:
        fx, fy, cx, cy = p[:4]
        k1 = 0.0
    xyz, rgb = [], []
    for line in open(args.model / "points3D.txt"):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        xyz.append([float(v) for v in e[1:4]])
        rgb.append([int(v) for v in e[4:7]])
    sp = args.out / "sparse/0"
    sp.mkdir(parents=True, exist_ok=True)
    if not (args.out / "images").exists():
        (args.out / "images").symlink_to(args.images.resolve())
    images = []
    for i, n in enumerate(names):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = imgs[n].R, imgs[n].tvec
        images.append((i + 1, T, 1, n))
    write_cameras_bin(sp / "cameras.bin", [(1, c.width, c.height, np.array([fx, fy, cx, cy]))])
    write_images_bin(sp / "images.bin", images)
    write_points_ply(sp / "points3D.ply", np.array(xyz), np.array(rgb, np.uint8))
    print(f"{args.out}: {len(images)} images, {len(xyz)} points, f=({fx:.1f},{fy:.1f}) pp=({cx:.1f},{cy:.1f}) k1={k1:.2e}")


if __name__ == "__main__":
    main()
