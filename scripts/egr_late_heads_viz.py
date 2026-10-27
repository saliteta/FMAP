"""Per-head cross-frame attention in VGGT-Omega's late blocks (17-23): which key frames does each head read?

From runs/egr_attention_mass/mass.pt (register queries; mass on all of a key frame's tokens). For every block and
head, each query frame's row (self dropped, renormalised) is averaged over the query frames: the head's attention
profile over the key frames, shown as lift = share / uniform share (1/63). Key frames are ordered by their graph
distance from frame 0 (edge = >= 30 shared tie points): frame 0, hop 1, hop 2, hop 3+, other islands.
  late_heads_<batch>.png   one panel per block, rows = heads, columns = key frames (log colour)
  late_heads_summary.png   per block and head over all 6 batches: share on frame 0, and the share on the head's
                           3 most-read other frames

    python scripts/egr_late_heads_viz.py --batch 0 4
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
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from egr_attention_viz import BLUES, INK, INK2, SURFACE  # noqa: E402

BLOCKS = range(17, 24)
REGISTER_BLOCKS = (20,)
HOP_NAMES = ["f0", "hop 1", "hop 2", "hop 3+", "other island"]

plt.rcParams.update({"font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def profile(m):
    """(n, n) mass of one head -> mean over query frames 1..n-1 of the row share (self dropped), shape (n,)."""
    n = len(m)
    rows = []
    for i in range(1, n):
        p = m[i].astype(np.float64).copy()
        p[i] = 0
        rows.append(p / p.sum())
    return np.mean(rows, 0)


def hop_classes(G):
    D = shortest_path(G.astype(float), unweighted=True, directed=False)[0]
    cls = np.where(np.isinf(D), 4, np.minimum(D, 3)).astype(int)
    cls[0] = 0
    return cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=Path, default=Path("runs/egr_attention_mass/mass.pt"))
    ap.add_argument("--batch", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--out", type=Path, default=Path("runs/egr_late_heads"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.mass, weights_only=False)["data"]

    for bi in args.batch:
        d = data[bi]
        cls = hop_classes(d["G"])
        order = np.lexsort((np.arange(len(cls)), cls))           # frame 0, hop 1, hop 2, hop 3+, other islands
        bounds = [np.searchsorted(cls[order], k) - 0.5 for k in range(1, 5)]
        fig, axes = plt.subplots(len(BLOCKS), 1, figsize=(15, 2.05 * len(BLOCKS) + 1.0))
        norm = LogNorm(vmin=0.1, vmax=63)
        for ax, b in zip(axes, BLOCKS):
            P = np.stack([profile(d["mass"][b]["reg"][h]) for h in range(16)]) * 63    # lift
            im = ax.imshow(P[:, order], cmap=BLUES, norm=norm, aspect="auto", interpolation="nearest")
            for x in bounds:
                ax.axvline(x, color="#eb6834", lw=1)
            ax.set_yticks(range(0, 16, 3))
            ax.set_ylabel(f"block {b}{' (reg)' if b in REGISTER_BLOCKS else ''}\nhead", fontsize=8.5)
            ax.set_xticks([])
        centers = [(lo + hi) / 2 for lo, hi in zip([-0.5] + bounds, bounds + [len(cls) - 0.5]) if hi > lo]
        names = [n for (lo, hi), n in zip(zip([-0.5] + bounds, bounds + [len(cls) - 0.5]), HOP_NAMES) if hi > lo]
        axes[-1].set_xticks(centers)
        axes[-1].set_xticklabels(names)
        axes[-1].set_xlabel("key frames, ordered by graph distance from frame 0 (orange lines = class borders)")
        cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
        cb.set_label("attention to that frame ÷ uniform (log)")
        cb.outline.set_visible(False)
        fig.suptitle(f"{d['kind']} batch: each head's attention over the 63 other frames (register queries, mean over "
                     "query frames), blocks 17-23", x=0.01, ha="left", fontsize=11)
        fig.savefig(args.out / f"late_heads_batch{bi}_{d['kind']}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # summary over all batches
    f0 = np.zeros((len(BLOCKS), 16))
    top3 = np.zeros((len(BLOCKS), 16))
    for r, b in enumerate(BLOCKS):
        for h in range(16):
            a, t = [], []
            for d in data:
                p = profile(d["mass"][b]["reg"][h])
                a.append(p[0])
                rest = np.sort(p[1:])[::-1]
                t.append(rest[:3].sum() / max(1 - p[0], 1e-9))
            f0[r, h], top3[r, h] = np.mean(a), np.mean(t)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.2))
    for ax, M, title in ((a1, f0, "share of the head's attention on frame 0"),
                         (a2, top3, "share of its non-frame-0 attention on its 3 most-read frames (uniform: 5%)")):
        im = ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(BLOCKS)))
        ax.set_yticklabels([f"block {b}" for b in BLOCKS])
        ax.set_xticks(range(16))
        ax.set_xlabel("head")
        ax.set_title(title, loc="left", fontsize=10)
        for (r, h), v in np.ndenumerate(M):
            ax.text(h, r, f"{v * 100:.0f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if v > 0.5 else INK)
    fig.suptitle("Late blocks, per head, mean over 6 HAV batches (register queries; values in %)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out / "late_heads_summary.png", dpi=130)
    plt.close(fig)
    np.savez(args.out / "late_heads_summary.npz", f0=f0, top3=top3, blocks=np.array(list(BLOCKS)))
    print("share on frame 0 (%), rows = blocks 17..23, columns = heads 0..15")
    for r, b in enumerate(BLOCKS):
        print(f"  b{b}: " + " ".join(f"{v * 100:3.0f}" for v in f0[r]))
    print("share of non-frame-0 attention on the 3 most-read frames (%)")
    for r, b in enumerate(BLOCKS):
        print(f"  b{b}: " + " ".join(f"{v * 100:3.0f}" for v in top3[r]))
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
