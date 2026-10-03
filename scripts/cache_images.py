"""Downscale a scene's images once to local disk (FM backbones run at ~518 px anyway).

    python scripts/cache_images.py --scene /mnt/z/.../HAV/colmap_metrics --width 518
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def resize_one(src: Path, dst: Path, width: int) -> None:
    if dst.exists():
        return
    im = Image.open(src)
    im.draft("RGB", (width, width))              # fast JPEG DCT downscale
    im = im.convert("RGB")
    h = round(im.height * width / im.width)
    im.resize((width, h), Image.BICUBIC).save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    scene_name = args.scene.parent.name if args.scene.name == "colmap_metrics" else args.scene.name
    out = args.out or Path("runs/cache") / f"{scene_name}_images_w{args.width}"
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted((args.scene / "images").glob("*.jpg"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: resize_one(p, out / p.name, args.width), srcs))
    print(f"cached {len(srcs)} images -> {out}")


if __name__ == "__main__":
    main()
