"""Figures for scripts/egr_attention_mass.py: which images in one VGGT-Omega batch the attention probe relates.

  matrices_<kind>.png : frame-to-frame matrices of one batch (rows = query image, row-normalised, diagonal
                        dropped) - reference tie-point overlap, depth overlap, and attention probes
  queries_<kind>.png  : for a few query images, the images the probe ranks highest and lowest, with the
                        reference verdict (shares >= 30 tie points with the query or not)
  map_<kind>.png      : top-down camera positions of the batch, coloured by the query's attention share

    python scripts/egr_attention_viz.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics --batch 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_attention_mass import ap, row_norm  # noqa: E402
from fgsfm.geometry.transforms import centers  # noqa: E402
from fgsfm.io.scene import load_reference_scene  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
GOOD, BAD = "#0ca30c", "#d03b3b"
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
PROBE = (13, "patch", 15)                  # best probe: block 13, patch queries, head 15
PANELS = [("Reference: tie-point overlap", None),
          ("Current graph: depth overlap", "base"),
          ("Block 13 · patch queries · head 15", (13, "patch", 15)),
          ("Block 14 · register attention · head 15", (14, "reg", 15)),
          ("Block 14 · register attention · 16 heads summed", (14, "reg", None)),
          ("Block 9 · register attention · head 15", (9, "reg", 15))]

plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def matrix(d, spec):
    if spec is None:
        return row_norm(d["O"])
    if spec == "base":
        return row_norm(d["base"])
    b, qs, h = spec
    m = d["mass"][b][qs].astype(np.float32)
    return row_norm(m.sum(0) if h is None else m[h])


def spectral_order(O):
    """1-D ordering that puts mutually overlapping images next to each other (Fiedler vector of the overlap)."""
    W = 0.5 * (O + O.T)
    np.fill_diagonal(W, 0)
    L = np.diag(W.sum(1)) - W
    return np.argsort(np.linalg.eigh(L)[1][:, 1])


def thumb(path, w=260):
    im = Image.open(path)
    im.thumbnail((w, w))
    return np.asarray(im)


def frame_axis(ax, colour, lw=4):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour), s.set_linewidth(lw)


def fig_matrices(d, out):
    order = spectral_order(d["O"])
    G = d["G"][np.ix_(order, order)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4))
    for ax, (title, spec) in zip(axes.flat, PANELS):
        M = matrix(d, spec)
        ra = np.nan if spec is None else np.mean(
            [ap(M[i, np.arange(len(M)) != i], d["G"][i, np.arange(len(M)) != i])
             for i in range(len(M)) if 0 < d["G"][i].sum() < len(M) - 1])
        M = M[np.ix_(order, order)]
        im = ax.imshow(M, cmap=BLUES, vmin=0, vmax=np.percentile(M, 99), interpolation="nearest")
        yy, xx = np.nonzero(G)
        ax.scatter(xx, yy, s=2.5, c=GOOD, marker="s", linewidths=0, alpha=0.9)
        ax.set_title(title + ("" if spec is None else f"\nmean row AP {ra:.2f}"), fontsize=10, loc="left")
        ax.set_xlabel("key image j (ordered by overlap)")
        ax.set_ylabel("query image i")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("share of i's attention to j" if spec not in (None, "base") else
                     ("tie-point overlap (row share)" if spec is None else "depth covisibility (row share)"))
        cb.outline.set_visible(False)
    fig.suptitle(f"One {d['kind']} batch of 64 HAV images: which image relates to which "
                 "(green dots = reference edge, ≥30 shared tie points; chance row AP ≈ "
                 f"{np.mean([g.sum() / (len(g) - 1) for g in d['G'] if g.sum()]):.2f})",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def pick_queries(d, M):
    n = len(M)
    rows = [i for i in range(n) if 0 < d["G"][i].sum() < n - 1]
    aps = {i: ap(M[i, np.arange(n) != i], d["G"][i, np.arange(n) != i]) for i in rows}
    deg = d["G"].sum(1)
    many = sorted([i for i in rows if deg[i] >= np.median(deg[rows])], key=lambda i: aps[i])
    few = sorted([i for i in rows if deg[i] < np.median(deg[rows])], key=lambda i: aps[i])
    picks = [("well connected", many[len(many) // 2]),
             ("weakly connected", few[len(few) // 2]),
             ("hardest", min(rows, key=lambda i: aps[i]))]
    return picks, aps


def fig_queries(d, names, img_dir, out, k_top=6, k_low=3):
    M = matrix(d, PROBE)
    picks, aps = pick_queries(d, M)
    shared = d["shared"]
    n = len(M)
    ncol = 1 + k_top + k_low
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.15 * ncol, 2.35 * len(picks) + 0.6))
    for r, (label, q) in enumerate(picks):
        others = np.array([j for j in range(n) if j != q])
        rank = others[np.argsort(-M[q, others])]
        cols = [(q, None)] + [(j, "top") for j in rank[:k_top]] + [(j, "low") for j in rank[-k_low:]]
        for c, (j, kind) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(thumb(img_dir / names[d["idx"][j]]))
            if kind is None:
                frame_axis(ax, INK, 3)
                ax.set_title(f"QUERY: {label}\n{int(d['G'][q].sum())} neighbours · row AP {aps[q]:.2f}",
                             fontsize=8, loc="left")
                continue
            ok = bool(d["G"][q, j])
            frame_axis(ax, GOOD if ok else BAD)
            pos = int(np.flatnonzero(rank == j)[0]) + 1
            ax.set_title(f"#{pos}  {'✓ neighbour' if ok else '✗ not neighbour'}\n"
                         f"attn {M[q, j] * 100:.1f}% · {int(shared[q, j])} tie pts", fontsize=8,
                         color=INK, loc="left")
    fig.text(0.01, 0.995, "Block 13 · head 15 attention: for each query, the 6 images it attends to most and the "
             "3 it attends to least. Border: green ✓ = ≥30 shared tie points with the query (reference), "
             "red ✗ = fewer.", ha="left", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    # a gap marker between the top-k and the bottom-k columns
    x = 0.5 * (axes[0, k_top].get_position().x1 + axes[0, k_top + 1].get_position().x0)
    fig.add_artist(plt.Line2D([x, x], [0.02, 0.95], color="#c3c2b7", lw=1.5, ls=(0, (4, 3))))
    fig.text(x + 0.004, 0.955, "least attended →", fontsize=8, color=INK2, va="bottom")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return picks


def fig_map(d, C, picks, out):
    M = matrix(d, PROBE)
    xy = C[d["idx"], :2]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.0))
    for ax, (label, q) in zip(axes, picks):
        share = M[q].copy()
        others = np.arange(len(M)) != q
        sc = ax.scatter(xy[others, 0], xy[others, 1], c=share[others], cmap=BLUES, vmin=0,
                        vmax=share[others].max(), s=70, edgecolors=SURFACE, linewidths=1.5, zorder=2)
        nb = d["G"][q]
        ax.scatter(xy[nb, 0], xy[nb, 1], s=170, facecolors="none", edgecolors=GOOD, linewidths=1.8, zorder=3,
                   label="reference neighbour (≥30 tie pts)")
        ax.scatter(*xy[q], marker="*", s=420, c=INK, edgecolors=SURFACE, linewidths=1.5, zorder=4, label="query")
        ax.set_aspect("equal")
        ax.set_title(f"query: {label} image", fontsize=10, loc="left")
        ax.set_xlabel("x, reference frame (m)"), ax.set_ylabel("y, reference frame (m)")
        ax.grid(color="#e6e5e0", lw=0.6, zorder=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("query's attention share (block 13 · head 15)")
        cb.outline.set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(f"Camera positions of the {d['kind']} batch (top-down): darker = more attention from the query",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--scene", type=Path, required=True)
    ap_.add_argument("--mass", type=Path, default=Path("runs/egr_attention_mass/mass.pt"))
    ap_.add_argument("--batch", type=int, nargs="+", default=[0, 2, 4])
    ap_.add_argument("--out", type=Path, default=Path("runs/egr_attention_viz"))
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.mass, weights_only=False)
    ref = load_reference_scene(args.scene)
    C = centers(np.stack([c.pose_w2c for c in ref.graph.cameras]))
    C = C - C.mean(0)
    for b in args.batch:
        d = saved["data"][b]
        d["shared"] = ref.shared[np.ix_(d["idx"], d["idx"])]
        tag = f"{b}_{d['kind']}"
        fig_matrices(d, args.out / f"matrices_{tag}.png")
        picks = fig_queries(d, saved["names"], args.scene / "images", args.out / f"queries_{tag}.png")
        fig_map(d, C, picks, args.out / f"map_{tag}.png")
        print(f"batch {tag}: wrote matrices / queries / map", flush=True)


if __name__ == "__main__":
    main()
