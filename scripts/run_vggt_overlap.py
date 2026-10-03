"""Run VGGT on batches of a scene and build an FM overlap graph.

Each batch gives O_ij for all its pairs (depth reprojection). Pairs seen in
several batches are averaged. Pairs never co-batched stay NaN (unevaluated).

    python scripts/run_vggt_overlap.py --scene /mnt/z/.../HAV/colmap_metrics \
        --batch-size 24 --num-random 12 --out runs/HAV_vggt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.fm.backbones.vggt_adapter import VGGTAdapter
from fgsfm.fm.overlap import reprojection_overlap, symmetric_overlap
from fgsfm.io.scene import load_reference_scene
from fgsfm.proposal.batches import random_batches, spatial_knn_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, default=None, help="downscaled images (scripts/cache_images.py)")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-random", type=int, default=12)
    ap.add_argument("--coverage", type=int, default=1, help="min #spatial batches per camera")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scene = load_reference_scene(args.scene)
    g = scene.graph
    n = len(g)
    scene_name = args.scene.parent.name if args.scene.name == "colmap_metrics" else args.scene.name
    image_dir = args.image_dir or Path("runs/cache") / f"{scene_name}_images_w518"
    paths = [image_dir / c.name for c in g.cameras]
    assert all(p.exists() for p in paths), f"missing cached images in {image_dir}"

    rng = np.random.default_rng(args.seed)
    centers = np.stack([c.center for c in g.cameras])
    batches = [("spatial", b) for b in spatial_knn_batches(centers, args.batch_size, rng, args.coverage)]
    batches += [("random", b) for b in random_batches(n, args.batch_size, args.num_random, rng)]
    print(f"{n} cameras -> {len(batches)} batches of {args.batch_size} "
          f"({sum(k == 'spatial' for k, _ in batches)} spatial, {args.num_random} random)")

    model = VGGTAdapter()
    ov_sum = np.zeros((n, n))
    ov_cnt = np.zeros((n, n), int)
    covis_dir = np.full((n, n), np.nan)              # last directional covis, for inspection
    records = []
    args.out.mkdir(parents=True, exist_ok=True)
    for b, (kind, idx) in enumerate(tqdm(batches)):
        res = model.infer([paths[i] for i in idx])
        covis = reprojection_overlap(res.extrinsic, res.intrinsic, res.depth, res.depth_conf)
        O = symmetric_overlap(covis).cpu().numpy()
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        ov_sum[ii, jj] += O
        ov_cnt[ii, jj] += 1
        covis_dir[ii, jj] = covis.cpu().numpy()
        records.append(dict(batch=b, kind=kind, idx=idx.tolist(),
                            extrinsic=res.extrinsic.cpu().numpy().tolist(),
                            intrinsic=res.intrinsic.cpu().numpy().tolist(),
                            overlap=O.tolist(),
                            median_conf=float(res.depth_conf.median()),
                            time=res.inference_time, vram_gb=res.peak_vram_gb))

    with np.errstate(invalid="ignore"):
        vggt_overlap = np.where(ov_cnt > 0, ov_sum / np.maximum(ov_cnt, 1), np.nan)
    np.fill_diagonal(vggt_overlap, 1.0)
    np.savez_compressed(args.out / "vggt_overlap.npz", overlap=vggt_overlap, count=ov_cnt,
                        covis_dir=covis_dir, names=np.array([c.name for c in g.cameras]))
    (args.out / "batches.json").write_text(json.dumps(records))
    times = [r["time"] for r in records]
    summary = dict(num_batches=len(batches), batch_size=args.batch_size,
                   mean_batch_time_s=float(np.mean(times)), total_infer_time_s=float(np.sum(times)),
                   peak_vram_gb=float(max(r["vram_gb"] for r in records)),
                   pairs_evaluated=int((np.triu(ov_cnt, 1) > 0).sum()), pairs_total=n * (n - 1) // 2)
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
