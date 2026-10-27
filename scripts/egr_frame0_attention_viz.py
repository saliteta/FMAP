"""How much more attention does every image give frame 0 (the batch's reference frame) than any other image?

From runs/egr_attention_mass/mass.pt (register queries; mass on all of a key frame's tokens; self dropped, rows
renormalised; values shown as lift = attention / uniform share 1/63):
  frame0_matrices_<batch>.png : per block, heads summed, frame 0 as the first column, log colour scale
  frame0_lift.png             : per block, the lift each image gives frame 0 vs the other images (6 batches),
                                and the frame-0 lift of every head (heads x blocks)

    python scripts/egr_frame0_attention_viz.py --batch 4
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_attention_mass import row_norm  # noqa: E402
from egr_attention_viz import BLUES, INK, INK2, SURFACE, spectral_order  # noqa: E402

REGISTER_BLOCKS = (2, 6, 9, 14, 20)
FRAME0, OTHER = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": "#c3c2b7", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def lift_matrix(m):
    """(H, n, n) mass -> heads summed, self dropped, row-normalised, times (n - 1): 1 = uniform."""
    M = row_norm(m.astype(np.float64).sum(0))
    return M * (len(M) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=Path, default=Path("runs/egr_attention_mass/mass.pt"))
    ap.add_argument("--batch", type=int, nargs="+", default=[4, 0])
    ap.add_argument("--out", type=Path, default=Path("runs/egr_frame0_attention"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.mass, weights_only=False)["data"]
    n_blocks = len(data[0]["mass"])

    for bi in args.batch:
        d = data[bi]
        rest = 1 + spectral_order(d["O"][1:, 1:])                     # frame 0 first, others by overlap
        order = np.r_[0, rest]
        fig, axes = plt.subplots(4, 6, figsize=(17, 12.4))
        norm = LogNorm(vmin=0.1, vmax=100)
        for b, ax in zip(range(n_blocks), axes.flat):
            L = lift_matrix(d["mass"][b]["reg"])[np.ix_(order, order)]
            np.fill_diagonal(L, np.nan)
            im = ax.imshow(L, cmap=BLUES, norm=norm, interpolation="nearest")
            f0 = np.nanmedian(L[1:, 0])
            oth = np.nanmedian(L[1:, 1:])
            ax.set_title(f"block {b}{' (register)' if b in REGISTER_BLOCKS else ''}\n"
                         f"to frame 0: {f0:.1f}× · to others: {oth:.2f}×", fontsize=8.5, loc="left")
            ax.set_xticks([0]), ax.set_xticklabels(["f0"], fontsize=7)
            ax.set_yticks([])
        cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
        cb.set_label("attention from query image (row) to key image (column) ÷ uniform  (log scale)")
        cb.outline.set_visible(False)
        fig.suptitle(f"{d['kind']} batch of 64 images: register-query attention, 16 heads summed. Column 0 = frame 0 "
                     "(the reference frame); titles give the median over query images", x=0.01, ha="left",
                     fontsize=12)
        fig.savefig(args.out / f"frame0_matrices_batch{bi}_{d['kind']}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # lifts over all 6 batches
    blocks = np.arange(n_blocks)
    f0_q, oth_q, head_lift = [], [], np.zeros((16, n_blocks))
    for b in blocks:
        f0_vals, oth_vals = [], []
        for d in data:
            L = lift_matrix(d["mass"][b]["reg"])
            f0_vals.append(L[1:, 0])
            o = L[1:, 1:].copy()
            np.fill_diagonal(o, np.nan)
            oth_vals.append(np.nanmean(o, 1))                       # each query's mean lift to a non-frame-0 image
            for h in range(16):
                Mh = row_norm(d["mass"][b]["reg"][h].astype(np.float64)) * 63
                head_lift[h, b] += Mh[1:, 0].mean() / len(data)
        f0_vals, oth_vals = np.concatenate(f0_vals), np.concatenate(oth_vals)
        f0_q.append(np.percentile(f0_vals, [10, 50, 90]))
        oth_q.append(np.percentile(oth_vals, [10, 50, 90]))
    f0_q, oth_q = np.array(f0_q), np.array(oth_q)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
    for q, color, label in ((f0_q, FRAME0, "to frame 0"), (oth_q, OTHER, "to any other image (mean per query)")):
        a1.fill_between(blocks, q[:, 0], q[:, 2], color=color, alpha=0.18, lw=0)
        a1.plot(blocks, q[:, 1], "-o", color=color, lw=2, ms=4.5, label=f"{label}: median, 10-90% band")
    a1.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    a1.set_yscale("log")
    a1.set_xticks(blocks)
    a1.set_xlabel("aggregator block (shaded = register-only block)")
    a1.set_ylabel("attention ÷ uniform (16 heads summed)")
    a1.set_title("Every image's attention to frame 0 vs to the other images (6 batches, 63 query images each)",
                 loc="left", fontsize=10)
    for b in REGISTER_BLOCKS:
        a1.axvspan(b - 0.4, b + 0.4, color="#f0efec", zorder=0)
    a1.grid(axis="y", color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.legend(loc="upper left", frameon=False, fontsize=8.5)
    im = a2.imshow(head_lift, cmap=BLUES, norm=LogNorm(vmin=0.3, vmax=63), aspect="auto", interpolation="nearest")
    a2.set_xticks(blocks), a2.set_yticks(range(16))
    a2.set_xlabel("aggregator block"), a2.set_ylabel("head")
    a2.set_title("Attention to frame 0 ÷ uniform, per head (63 = all of the head's attention on frame 0)",
                 loc="left", fontsize=10)
    for (h, b), v in np.ndenumerate(head_lift):
        if v >= 10:
            a2.text(b, h, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=SURFACE)
    cb = fig.colorbar(im, ax=a2, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "frame0_lift.png", dpi=130)
    plt.close(fig)
    print("block  frame0 median (10-90%)        others median")
    for b in blocks:
        print(f"{b:5d}  {f0_q[b, 1]:6.1f} ({f0_q[b, 0]:5.1f}-{f0_q[b, 2]:5.1f})   {oth_q[b, 1]:6.2f}")
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
