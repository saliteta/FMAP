"""Evaluate any COLMAP model (e.g. COLMAP 4.x global_mapper / GLOMAP) against the HAV reference.

    python scripts/eval_colmap_model.py --scene /mnt/d/.../HAV/colmap_metrics --model runs/colmap421_HAV/sparse/0

Binary models are converted to text with the given colmap binary first.
Same pose metrics as our pipeline (pose_metrics: Sim(3)-aligned ATE, abs/rel rotation, RRA/RTA, AUC).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.eval.pose_metrics import pose_metrics
from fgsfm.io.colmap_text import ColmapModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--colmap", type=str, default=str(Path.home() / "miniconda3/envs/colmap/bin/colmap"))
    args = ap.parse_args()

    txt = args.model.parent / f"{args.model.name}_txt"
    if not (args.model / "cameras.txt").exists():
        txt.mkdir(exist_ok=True)
        subprocess.run([args.colmap, "model_converter", "--input_path", str(args.model),
                        "--output_path", str(txt), "--output_type", "TXT"], check=True, capture_output=True)
        model_dir = txt
    else:
        model_dir = args.model
    est = ColmapModel.load(model_dir)
    ref = ColmapModel.load(args.scene / "sparse" / "0")
    ref_by = {im.name: im for im in ref.images.values()}
    est_by = {im.name: im for im in est.images.values()}
    common = sorted(set(ref_by) & set(est_by))

    def w2c(im):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = im.R, im.tvec
        return T

    m = pose_metrics(np.stack([w2c(est_by[n]) for n in common]), np.stack([w2c(ref_by[n]) for n in common]))
    n_pts = sum(1 for _ in open(model_dir / "points3D.txt") if not _.startswith("#"))
    cams = {c.camera_id: (c.model, c.params.round(2).tolist()) for c in est.cameras.values()}
    out = dict(model=str(args.model), registered=len(est_by), reference=len(ref_by), common=len(common),
               points=n_pts, cameras=cams, metrics=m)
    print(json.dumps(out, indent=1, default=float))
    (args.model.parent / f"{args.model.name}_eval.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
