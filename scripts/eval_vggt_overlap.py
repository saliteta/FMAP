"""Compare VGGT batch overlap against reference covisibility (tie-point tracks).

    python scripts/eval_vggt_overlap.py --scene /mnt/z/.../HAV/colmap_metrics --run runs/HAV_vggt

Writes <run>/metrics.json and <run>/figures/*.png.
Positives = pairs sharing >= --min-shared reference points (README: N = 30).
Baseline = camera-center distance (stand-in for GPS; uses reference poses, so optimistic).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.eval.overlap_metrics import (binary_metrics, pose_auc, pr_curve,
                                        relative_pose_errors, spearman)
from fgsfm.graph.footprint import footprint_overlap
from fgsfm.io.scene import load_reference_scene

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--min-shared", type=int, default=30)
    ap.add_argument("--min-footprint", type=float, default=0.1,
                    help="positive threshold for the track-free footprint reference")
    args = ap.parse_args()

    scene = load_reference_scene(args.scene)
    g = scene.graph
    n = len(g)
    ref_overlap = scene.scores("overlap")
    shared = scene.shared
    pred = np.load(args.run / "vggt_overlap.npz")
    vggt, count = pred["overlap"], pred["count"]
    records = json.loads((args.run / "batches.json").read_text())
    summary = json.loads((args.run / "run_summary.json").read_text())

    # which batch kinds evaluated each pair
    kind_mask = {k: np.zeros((n, n), bool) for k in ("spatial", "random")}
    for r in records:
        idx = np.array(r["idx"])
        kind_mask[r["kind"]][np.ix_(idx, idx)] = True

    iu, ju = np.triu_indices(n, k=1)
    ev = count[iu, ju] > 0
    pairs = {"all": ev, "spatial": ev & kind_mask["spatial"][iu, ju], "random": ev & kind_mask["random"][iu, ju]}
    centers = np.stack([c.center for c in g.cameras])
    dist = np.linalg.norm(centers[iu] - centers[ju], axis=1)

    # second, track-free reference: ground-plane footprint overlap from reference poses
    w2c_ref = np.stack([c.pose_w2c for c in g.cameras])
    up = -np.stack([c.pose_c2w[:3, 2] for c in g.cameras]).mean(0)
    up /= np.linalg.norm(up)
    ground_z = np.array([np.median(scene.points_xyz[scene.obs_point[scene.obs_image == i]] @ up)
                         for i in range(n)])
    cam0 = g.cameras[0]
    footprint = footprint_overlap(w2c_ref, cam0.K, (cam0.height, cam0.width), ground_z, up)
    np.save(args.run / "footprint_overlap.npy", footprint)

    metrics = dict(run=summary, min_shared=args.min_shared, min_footprint=args.min_footprint,
                   overlap={}, gps_baseline={}, overlap_vs_footprint={}, gps_baseline_vs_footprint={},
                   reference_disagreement={}, pose={})
    for name, m in pairs.items():
        y = shared[iu, ju][m] >= args.min_shared
        s = vggt[iu, ju][m]
        metrics["overlap"][name] = dict(
            pairs=int(m.sum()),
            spearman_vs_ref_overlap=spearman(s, ref_overlap[iu, ju][m]),
            spearman_vs_shared_points=spearman(s, shared[iu, ju][m]),
            mean_abs_error=float(np.abs(s - ref_overlap[iu, ju][m]).mean()),
            **binary_metrics(s, y))
        metrics["gps_baseline"][name] = binary_metrics(-dist[m], y)
        yf = footprint[iu, ju][m] >= args.min_footprint
        metrics["overlap_vs_footprint"][name] = dict(
            spearman_vs_footprint=spearman(s, footprint[iu, ju][m]), **binary_metrics(s, yf))
        metrics["gps_baseline_vs_footprint"][name] = binary_metrics(-dist[m], yf)

    # are "false positives" against tie points real overlap?
    m = pairs["all"]
    s, sh_, fp_ = vggt[iu, ju][m], shared[iu, ju][m], footprint[iu, ju][m]
    fp = (s >= 0.3) & (sh_ < args.min_shared)
    fn = (s < 0.05) & (sh_ >= args.min_shared)
    metrics["reference_disagreement"] = dict(
        vggt_high_but_few_tie_points=int(fp.sum()),
        of_which_footprint_ge_0p2=float((fp_[fp] >= 0.2).mean()) if fp.any() else None,
        their_median_footprint=float(np.median(fp_[fp])) if fp.any() else None,
        vggt_low_but_many_tie_points=int(fn.sum()),
        their_median_footprint_fn=float(np.median(fp_[fn])) if fn.any() else None,
        spearman_tiepoint_ref_vs_footprint=spearman(ref_overlap[iu, ju][m], fp_))

    # relative pose accuracy per batch kind
    pose_rows = []
    for r in records:
        idx = np.array(r["idx"])
        rot, tra, a, b = relative_pose_errors(np.array(r["extrinsic"]), w2c_ref[idx])
        for e1, e2, i, j in zip(rot, tra, idx[a], idx[b]):
            pose_rows.append((r["kind"], e1, e2, shared[i, j]))
    # batch health: does VGGT's own confidence flag batches whose poses failed?
    b_conf = np.array([r["median_conf"] for r in records])
    b_rot = np.array([np.median(relative_pose_errors(np.array(r["extrinsic"]), w2c_ref[np.array(r["idx"])])[0])
                      for r in records])
    failed = b_rot > 5.0
    metrics["batch_health"] = dict(
        batches=len(records), failed_batches_rot_gt_5deg=int(failed.sum()),
        failed_spatial=int((failed & np.array([r["kind"] == "spatial" for r in records])).sum()),
        spearman_conf_vs_rot_err=spearman(b_conf, b_rot),
        **{f"conf_detector_{k}": v for k, v in binary_metrics(-b_conf, failed).items()
           if k in ("roc_auc", "best_f1", "best_f1_threshold", "best_f1_precision", "best_f1_recall")})
    kinds = np.array([p[0] for p in pose_rows])
    rot = np.array([p[1] for p in pose_rows])
    tra = np.array([p[2] for p in pose_rows])
    sh = np.array([p[3] for p in pose_rows])
    for name, m in [("spatial", kinds == "spatial"), ("random", kinds == "random"),
                    ("spatial_covisible", (kinds == "spatial") & (sh >= args.min_shared))]:
        if m.any():
            metrics["pose"][name] = dict(
                pairs=int(m.sum()), median_rot_err_deg=float(np.median(rot[m])),
                median_trans_dir_err_deg=float(np.median(tra[m])),
                RRA_5=float((rot[m] <= 5).mean()), RTA_5=float((tra[m] <= 5).mean()),
                RRA_15=float((rot[m] <= 15).mean()), RTA_15=float((tra[m] <= 15).mean()),
                AUC_30=pose_auc(rot[m], tra[m], 30.0))

    (args.run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    figs = args.run / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "font.size": 10})

    # 1. scatter reference overlap vs VGGT overlap
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, col in (("spatial", BLUE), ("random", ORANGE)):
        m = pairs[name]
        ax.scatter(ref_overlap[iu, ju][m], vggt[iu, ju][m], s=8, c=col, alpha=0.35, lw=0,
                   label=f"{name} batches ({m.sum()} pairs, ρ={metrics['overlap'][name]['spearman_vs_ref_overlap']:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("reference overlap (shared tie points / min points)")
    ax.set_ylabel("VGGT overlap (depth reprojection)")
    style(ax, "Pairwise overlap: VGGT vs reference")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "scatter_overlap.png", dpi=150)
    plt.close(fig)

    # 1b. scatter footprint overlap vs VGGT overlap
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, col in (("spatial", BLUE), ("random", ORANGE)):
        m = pairs[name]
        ax.scatter(footprint[iu, ju][m], vggt[iu, ju][m], s=8, c=col, alpha=0.35, lw=0,
                   label=f"{name} batches (ρ={metrics['overlap_vs_footprint'][name]['spearman_vs_footprint']:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("footprint overlap (reference poses, ground plane)")
    ax.set_ylabel("VGGT overlap (depth reprojection)")
    style(ax, "Pairwise overlap: VGGT vs track-free footprint")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "scatter_footprint.png", dpi=150)
    plt.close(fig)

    # 2. precision-recall: VGGT vs GPS-distance baseline (all evaluated pairs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    m = pairs["all"]
    for ax, y, title in ((axes[0], shared[iu, ju][m] >= args.min_shared, f"positive = ≥{args.min_shared} shared tie points"),
                         (axes[1], footprint[iu, ju][m] >= args.min_footprint, f"positive = footprint overlap ≥{args.min_footprint}")):
        for label, score, col in (("VGGT overlap", vggt[iu, ju][m], BLUE),
                                  ("camera distance (GPS proxy)", -dist[m], ORANGE)):
            p, r, _, _ = pr_curve(score, y)
            ap_ = binary_metrics(score, y)["average_precision"]
            ax.plot(r, p, color=col, lw=2, label=f"{label}  AP={ap_:.3f}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.02)
        style(ax, title)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / "pr_curve.png", dpi=150)
    plt.close(fig)

    # 3. matrices: reference vs VGGT (gray = never co-batched), cameras in flight order
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    cmap.set_bad("#d9d8d4")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, M, title in ((axes[0], ref_overlap, "Reference overlap (all pairs)"),
                         (axes[1], vggt, "VGGT overlap (gray = not co-batched)")):
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.8, interpolation="nearest")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("camera (flight order)", color=INK2)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="overlap")
    fig.savefig(figs / "overlap_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. relative rotation error vs reference shared points
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for name, col in (("spatial", BLUE), ("random", ORANGE)):
        mk = kinds == name
        ax.scatter(np.maximum(sh[mk], 0.8), rot[mk], s=6, c=col, alpha=0.35, lw=0, label=f"{name} batches")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference shared points (0 shown at 0.8)")
    ax.set_ylabel("relative rotation error (deg)")
    style(ax, "VGGT relative-pose error vs true covisibility")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(figs / "rotation_error_vs_shared.png", dpi=150)
    plt.close(fig)

    # 5. gallery: where VGGT and the tie-point reference disagree most
    from PIL import Image
    image_dir = args.run.parent / "cache" / f"{scene.root.parent.name}_images_w518"
    m = pairs["all"]
    pi, pj = iu[m], ju[m]
    s, sh_ = vggt[pi, pj], shared[pi, pj]
    fp_idx = np.argsort(-(s * (sh_ < args.min_shared)))[:4]          # VGGT high, few tie points
    fn_idx = np.argsort(-(sh_ * (s < 0.05)))[:2]                     # many tie points, VGGT ~0
    rows = [("VGGT high / tie points low", k) for k in fp_idx] + [("tie points high / VGGT low", k) for k in fn_idx]
    if image_dir.exists():
        fig, axes = plt.subplots(len(rows), 2, figsize=(9, 2.9 * len(rows)))
        for r_, (tag, k) in enumerate(rows):
            a, b = pi[k], pj[k]
            for c_, cam in enumerate((a, b)):
                ax = axes[r_, c_]
                ax.imshow(Image.open(image_dir / g.cameras[cam].name))
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_title(f"cam {cam}", fontsize=9, color=INK2, loc="left")
            axes[r_, 0].set_ylabel(f"{tag}\nVGGT {s[k]:.2f} | shared {int(sh_[k])}\nfootprint {footprint[a, b]:.2f}",
                                   fontsize=8.5, color=INK, rotation=0, ha="right", va="center")
        fig.tight_layout()
        fig.savefig(figs / "disagreement_gallery.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({k: metrics[k] for k in metrics if k != "run"}, indent=1))


if __name__ == "__main__":
    main()
