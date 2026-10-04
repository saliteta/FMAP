"""Build a scene graph by sequential VGGT sweep + seeded refinement (no GPS), evaluate each stage.

    python scripts/build_scene_graph.py --scene /mnt/z/.../HAV/colmap_metrics --out runs/HAV_graph

The reference (COLMAP poses, tie points, footprint overlap) is used ONLY for evaluation.
Outputs: <out>/graph.npz, <out>/metrics.json, <out>/log.json, and checkpoints in <out>/ckpt/:
  sweep.pkl, origin.pkl, round_XX.pkl, latest.pkl (+ *_node_status.json per checkpoint).
Resume (skips finished stages; config flags given now apply):
    python scripts/build_scene_graph.py --scene ... --out runs/HAV_graph_v6 --resume runs/HAV_graph_v6/ckpt/latest.pkl
    python scripts/build_scene_graph.py --scene ... --out runs/try --resume runs/HAV_graph_v6/ckpt/origin.pkl --max-rounds 5
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.eval.pose_metrics import pose_metrics
from fgsfm.graph.footprint import footprint_overlap
from fgsfm.io.scene import load_reference_scene
from fgsfm.pipeline.graph_builder import BuilderConfig, GraphBuilder


def reference_footprint(scene) -> np.ndarray:
    g = scene.graph
    w2c = np.stack([c.pose_w2c for c in g.cameras])
    up = -np.stack([c.pose_c2w[:3, 2] for c in g.cameras]).mean(0)
    up /= np.linalg.norm(up)
    gz = np.array([np.median(scene.points_xyz[scene.obs_point[scene.obs_image == i]] @ up) for i in range(len(g))])
    c0 = g.cameras[0]
    return footprint_overlap(w2c, c0.K, (c0.height, c0.width), gz, up)


def make_eval(scene, footprint, min_overlap: float = 0.1):
    w2c_ref = np.stack([c.pose_w2c for c in scene.graph.cameras])
    n = len(w2c_ref)
    iu, ju = np.triu_indices(n, 1)
    pos_ref = footprint[iu, ju] >= min_overlap

    def evaluate(builder: GraphBuilder, tag: str) -> dict:
        g = builder.graph
        frames = g.frames()
        sizes = sorted((len(c) for c in frames.values()), reverse=True)
        out = dict(tag=tag, registered_any=int((g.frame_of >= 0).sum()), num_segments=len(frames),
                   segment_sizes=sizes[:12], num_batches=len(g.batches))
        if g.seed_frame >= 0:
            cams = g.cams_in(g.seed_frame)
            out["seed"] = dict(registration_rate=len(cams) / n, **pose_metrics(g.w2c[cams], w2c_ref[cams]))
        else:   # before origin: largest segment
            f = max(frames, key=lambda k: len(frames[k]))
            cams = frames[f]
            out["largest_segment"] = dict(registration_rate=len(cams) / n, **pose_metrics(g.w2c[cams], w2c_ref[cams]))
            # size-weighted per-segment metrics: are segments internally right?
            per = [(len(c), pose_metrics(g.w2c[c], w2c_ref[c])) for c in frames.values() if len(c) >= 4]
            if per:
                wsum = sum(k for k, _ in per)
                out["segments_weighted"] = {m: float(sum(k * p[m] for k, p in per) / wsum)
                                            for m in ("ATE_median", "rel_rot_median_deg", "AUC@5")}
        cob = g.cobatched[iu, ju]
        meas = np.nan_to_num(g.measured[iu, ju])
        out["graph"] = dict(
            pairs_cobatched=int(cob.sum()),
            footprint_pos_pairs=int(pos_ref.sum()),
            recall_cobatched=float((cob & pos_ref).sum() / max(pos_ref.sum(), 1)),
            recall_measured_edge=float((cob & (meas >= min_overlap) & pos_ref).sum() / max(pos_ref.sum(), 1)),
            precision_measured_edge=float((cob & (meas >= min_overlap) & pos_ref).sum()
                                          / max((cob & (meas >= min_overlap)).sum(), 1)))
        s = out.get("seed", out.get("largest_segment"))
        print(f"[{tag:>8}] segments={out['num_segments']:3d} | cams={s['num_cameras']:3d} "
              f"({s['registration_rate']:.0%}) ATE med={s.get('ATE_median', float('nan')):.2f}m "
              f"rel-rot med={s.get('rel_rot_median_deg', float('nan')):.2f}° AUC@5={s.get('AUC@5', float('nan')):.3f} "
              f"| edge recall={out['graph']['recall_measured_edge']:.3f} prec={out['graph']['precision_measured_edge']:.3f} "
              f"| batches={out['num_batches']}", flush=True)
        return out
    return evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=None, help="default: <out>/ckpt")
    ap.add_argument("--resume", type=Path, default=None,
                    help="checkpoint .pkl to continue from (sweep/origin/round_XX/latest); CLI config applies")
    ap.add_argument("--keep-round-ckpts", action=argparse.BooleanOptionalAction, default=True,
                    help="keep round_XX.pkl for every round (otherwise only latest.pkl)")
    for f in dataclasses.fields(BuilderConfig):
        if isinstance(f.default, tuple):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=lambda v: tuple(int(x) for x in v.split(",")),
                            default=f.default)
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    cfg = BuilderConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(BuilderConfig)})

    scene = load_reference_scene(args.scene)
    names = [c.name for c in scene.graph.cameras]          # alphabetical = capture order
    scene_name = args.scene.parent.name if args.scene.name == "colmap_metrics" else args.scene.name
    image_dir = args.image_dir or Path("runs/cache") / f"{scene_name}_images_w518"
    paths = [image_dir / n for n in names]
    assert all(p.exists() for p in paths), f"missing cached images in {image_dir} (run scripts/cache_images.py)"
    evaluate = make_eval(scene, reference_footprint(scene))

    t0 = time.time()
    b = GraphBuilder(paths, cfg)
    b.eval_stages = []
    ckpt_dir = args.ckpt_dir or args.out / "ckpt"
    stage = "init"
    if args.resume is not None:
        stage = b.load_checkpoint(args.resume)
        b.eval_stages = getattr(b, "eval_stages_saved", None) or []
        print(f"resumed from {args.resume}: stage={stage}, next_round={b.next_round}, relax={b.relax}, "
              f"seed={int((b.graph.frame_of == b.graph.seed_frame).sum())} cams", flush=True)

    def checkpoint(name: str, st: str) -> None:
        b.eval_stages_saved = b.eval_stages
        b.save_checkpoint(ckpt_dir / f"{name}.pkl", st)
        if name != "latest":
            b.save_checkpoint(ckpt_dir / "latest.pkl", st)

    if stage == "init":
        b.sweep()
        b.eval_stages.append(evaluate(b, "sweep"))
        checkpoint("sweep", "sweep")
        stage = "sweep"
    if stage == "sweep":
        b.choose_origin()
        b.eval_stages.append(evaluate(b, "origin"))
        checkpoint("origin", "origin")
        stage = "origin"
    if not b.finished:
        def on_round_end(builder, r):
            builder.eval_stages.append(builder.history[-1]["eval"])
            checkpoint(f"round_{r:02d}" if args.keep_round_ckpts else "latest", "round")
        b.grow(eval_fn=evaluate, on_round_end=on_round_end)

    metrics = dict(config=dataclasses.asdict(cfg), stages=b.eval_stages,
                   rounds=[{k: v for k, v in r.items() if k != "eval"} for r in b.history],
                   resumed_from=str(args.resume) if args.resume else None)
    metrics["time"] = dict(this_session_s=time.time() - t0, inference_s=b.inference_time, batches=len(b.graph.batches))
    print(json.dumps(metrics["time"]))

    args.out.mkdir(parents=True, exist_ok=True)
    g = b.graph
    X, rgb, conf, *_ = g.kp.gather(g.seed_frame)
    np.savez_compressed(args.out / "graph.npz", names=np.array(g.names), frame_of=g.frame_of, w2c=g.w2c, K=g.K,
                        measured=g.measured, cobatched=g.cobatched, origin=g.origin, seed_frame=g.seed_frame,
                        abandoned=b.abandoned, relax_level=b.relax,
                        points=X.astype(np.float32), points_rgb=rgb, points_conf=conf.astype(np.float32))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (args.out / "log.json").write_text(json.dumps(b.log.events, indent=1, default=float))
    (args.out / "batches.json").write_text(json.dumps(g.batches, default=float))


if __name__ == "__main__":
    main()
