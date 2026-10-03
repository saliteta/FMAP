"""Interactive reference view graph for a COLMAP scene.

Select a camera; the others turn red (high covisibility) to blue (none).
Covisibility = shared tie points from the AT export (the SfM tracks).

    conda activate fgsfm
    python scripts/view_covisibility.py \
        --scene /mnt/z/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
    # then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.graph.covisibility import SCORE_TYPES
from fgsfm.io.scene import load_reference_scene
from fgsfm.viz.covisibility_viewer import CovisibilityViewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path,
                    default=Path("/mnt/z/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--thumbnails", action="store_true", help="draw image thumbnails in frustums (slower start)")
    ap.add_argument("--frustum-scale", type=float, default=None)
    ap.add_argument("--save-graph", type=Path, default=None, help="write overlap scores to .npz")
    args = ap.parse_args()

    scene = load_reference_scene(args.scene)
    g = scene.graph
    n_pairs = len(g) * (len(g) - 1) // 2
    print(f"{len(g)} cameras, {len(scene.points_xyz)} points, "
          f"{int((np.triu(scene.shared, 1) > 0).sum())}/{n_pairs} covisible pairs")
    if args.save_graph:
        g.save(args.save_graph)

    order = np.argsort(scene.obs_image, kind="stable")
    bounds = np.searchsorted(scene.obs_image[order], np.arange(len(g) + 1))
    pts_by_img = [scene.obs_point[order[bounds[i]:bounds[i + 1]]] for i in range(len(g))]

    score_fns = {k: (lambda k=k: scene.scores(k)) for k in SCORE_TYPES}

    viewer = CovisibilityViewer(
        g, score_fns,
        points_xyz=scene.points_xyz, points_rgb=scene.points_rgb,
        points_of_image=lambda i: pts_by_img[i],
        port=args.port, frustum_scale=args.frustum_scale, thumbnails=args.thumbnails)
    viewer.run_forever()


if __name__ == "__main__":
    main()
