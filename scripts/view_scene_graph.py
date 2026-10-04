"""View a graph built by build_scene_graph.py on top of the COLMAP reference.

All cameras are shown (original index order):
  - seed (main graph): VGGT poses, Sim(3)-aligned to the reference      -> gray
  - isolated segments: their own VGGT poses, each aligned with its own
    Sim(3) (they live in separate frames); segments < 3 cameras are drawn
    at their COLMAP pose                                                  -> one color each
  - unregistered / abandoned: no VGGT pose, drawn at the COLMAP pose      -> black
"Color by: segment" shows this partition; "Color by: score" colors the others
by the graph's VGGT overlap (gray = never co-batched), COLMAP overlap or footprint.

    python scripts/view_scene_graph.py --scene /mnt/z/.../HAV/colmap_metrics --run runs/HAV_graph_v5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.core.camera import CameraNode
from fgsfm.eval.pose_metrics import robust_sim3_align
from fgsfm.geometry.transforms import centers
from fgsfm.graph.view_graph import ViewGraph
from fgsfm.io.scene import load_reference_scene
from fgsfm.viz.covisibility_viewer import CovisibilityViewer

# categorical palette (dataviz reference instance), fixed order
PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164),
           (0, 131, 0), (74, 58, 167), (227, 73, 72)]
SEED_RGB, UNREG_RGB = (175, 175, 175), (20, 20, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--footprint", type=Path, default=Path("runs/HAV_vggt/footprint_overlap.npy"))
    args = ap.parse_args()

    scene = load_reference_scene(args.scene)
    ref = scene.graph.cameras
    n = len(ref)
    w2c_ref = np.stack([c.pose_w2c for c in ref])
    d = np.load(args.run / "graph.npz")
    frame_of, seed = d["frame_of"], int(d["seed_frame"])

    w2c_show = w2c_ref.copy()
    cat = np.full(n, -1)
    names, colors = {0: "seed (main graph)", -1: "unregistered / abandoned"}, {0: SEED_RGB, -1: UNREG_RGB}
    seed_sim = None
    others = sorted((f for f in set(frame_of.tolist()) if f >= 0 and f != seed),
                    key=lambda f: -(frame_of == f).sum())
    print(f"{'segment':<22}{'cams':>5}  {'id range':<10} {'internal ATE med':>16}")
    for k, f in enumerate([seed] + others):
        cams = np.flatnonzero(frame_of == f)
        cid = 0 if f == seed else k
        cat[cams] = cid
        if f != seed:
            names[cid] = f"segment {k}"
            colors[cid] = PALETTE[(k - 1) % len(PALETTE)]
        if len(cams) >= 3:
            sim = robust_sim3_align(centers(d["w2c"][cams]), centers(w2c_ref[cams]))
            w2c_show[cams] = sim.apply_w2c(d["w2c"][cams])
            ate = np.median(np.linalg.norm(centers(w2c_show[cams]) - centers(w2c_ref[cams]), axis=1))
            seed_sim = sim if f == seed else seed_sim
            ate_s = f"{ate:.1f} m"
        else:
            names[cid] += " (COLMAP pose)"
            ate_s = "-"
        print(f"{names[cid]:<22}{len(cams):>5}  {cams.min():>3}-{cams.max():<6} {ate_s:>16}")
    unreg = np.flatnonzero(frame_of < 0)
    if len(unreg):
        print(f"{names[-1]:<22}{len(unreg):>5}  ids {unreg.tolist()}")

    nodes = [CameraNode(image_id=i, name=ref[i].name, image_path=ref[i].image_path, pose_w2c=w2c_show[i],
                        K=ref[i].K, width=ref[i].width, height=ref[i].height) for i in range(n)]
    g = ViewGraph(nodes)
    meas = np.where(d["cobatched"], np.nan_to_num(d["measured"]), np.nan)
    score_fns = {"vggt graph": lambda: meas, "colmap overlap": lambda: scene.scores("overlap")}
    if args.footprint.exists():
        score_fns["footprint"] = lambda: np.load(args.footprint)
    pts = seed_sim.apply_points(d["points"].astype(np.float64)) if seed_sim is not None else None
    viewer = CovisibilityViewer(g, score_fns, points_xyz=pts, points_rgb=d["points_rgb"], port=args.port,
                                categories=cat, category_names=names, category_colors=colors)
    viewer.select(int(d["origin"]))
    viewer.run_forever()


if __name__ == "__main__":
    main()
