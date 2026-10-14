"""Where does VGGT-Omega decide that two images are related? Frame-to-frame scores from register attention.

Hooks the inter-frame attention of all 24 aggregator blocks and keeps the queries / keys of the camera +
register tokens of every frame (in the 5 register blocks these are the whole attention; in the 19 global
blocks they are the camera / register rows and columns of the full attention). Per block and frame pair
(i, j) it scores:

  max      max over heads of max over the 16 x 16 register logits q_i . k_j / sqrt(d)
  headmean mean over heads of that 16 x 16 max
  mass     softmax attention mass from frame i's registers to frame j's camera + registers
           (softmax over all frames' camera + register keys; mean over heads and queries)

symmetrized over (i, j) / (j, i). Each score is compared with the reference graph (Bentley AT tie points:
an edge when two images share >= 30 tie points): AP, ROC-AUC and Spearman vs the tie-point overlap. Baseline:
VGGT-Omega's final depth-reprojection overlap (the measure our graph builder uses today).

    python scripts/egr_register_affinity.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.eval.overlap_metrics import binary_metrics, spearman  # noqa: E402
from fgsfm.fm.overlap import batch_geometry  # noqa: E402
from fgsfm.geometry.transforms import centers  # noqa: E402
from fgsfm.io.scene import load_reference_scene  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

PRE = 17                                   # camera token + 16 registers per frame


class QKProbe:
    """Wraps one inter-frame attention module; keeps q, k of the camera + register tokens."""

    def __init__(self, attn, n_tok_per_frame_fn):
        self.attn, self.store, self.tok = attn, None, n_tok_per_frame_fn
        orig = attn.compute_attention

        def compute_attention(qkv, attn_bias=None, rope=None):
            B, L, _ = qkv.shape
            C, H = attn.qkv.in_features, attn.num_heads
            q, k, _ = torch.unbind(qkv.reshape(B, L, 3, H, C // H), 2)
            q, k = q.transpose(1, 2), k.transpose(1, 2)              # (B, H, L, d)
            if attn.use_qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)
            T = self.tok(L)                                          # tokens per frame in this sequence
            idx = (torch.arange(L // T, device=q.device)[:, None] * T + torch.arange(PRE, device=q.device)).reshape(-1)
            self.store = (q[0][:, idx].float(), k[0][:, idx].float())  # (H, N*17, d)
            return orig(qkv, attn_bias=attn_bias, rope=rope)
        attn.compute_attention = compute_attention


def frame_scores(q, k, n):
    """q, k: (H, n*17, d) -> (scores: dict of symmetric (n, n), per-head features (n, n, 3H))."""
    H, _, d = q.shape
    logit = torch.einsum("hqd,hkd->hqk", q, k) / d ** 0.5                       # (H, 17n, 17n)
    L = logit.view(H, n, PRE, n, PRE)
    reg = L[:, :, 1:, :, 1:].permute(0, 1, 3, 2, 4).reshape(H, n, n, -1)          # (H, n, n, 256) registers only
    mx = reg.amax(-1)                                                            # (H, n, n)
    mean = reg.mean(-1)
    top8 = reg.topk(8, dim=-1).values.mean(-1)
    att = torch.softmax(logit, dim=-1).view(H, n, PRE, n, PRE)
    mass_h = att[:, :, 1:].sum(-1).mean(2)                                       # (H, n, n) register queries
    mass_cam = att.sum(-1).mean(dim=(0, 2))                                      # camera + register queries
    m = mean.mean(0)
    off = ~torch.eye(n, dtype=torch.bool, device=m.device)
    mu = (m * off).sum(1, keepdim=True) / (n - 1)
    sd = (((m - mu) ** 2) * off).sum(1, keepdim=True).div(n - 2).sqrt().clamp(min=1e-6)
    kf = torch.nn.functional.normalize(k.view(H, n, PRE, d)[:, :, 1:].mean(2).permute(1, 0, 2).reshape(n, -1), dim=-1)
    out = {"max": mx.amax(0), "headmean": mx.mean(0), "mass": mass_h.mean(0), "mean": m, "top8": top8.mean(0),
           "mass_cam": mass_cam, "rowz": (m - mu) / sd, "kcos": kf @ kf.T}
    feats = torch.cat([mean, mx, mass_h], 0).permute(1, 2, 0)                      # (n, n, 3H)
    feats = 0.5 * (feats + feats.transpose(0, 1))
    return {key: (0.5 * (v + v.T)).cpu().numpy() for key, v in out.items()}, feats.cpu().numpy()


def probe_ap(X_by_batch, y_by_batch, epochs=300):
    """Leave-one-batch-out logistic regression; returns pooled held-out scores."""
    scores = []
    for t in range(len(X_by_batch)):
        Xtr = np.concatenate([x for i, x in enumerate(X_by_batch) if i != t])
        ytr = np.concatenate([y for i, y in enumerate(y_by_batch) if i != t])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        X = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
        y = torch.tensor(ytr, dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b0 = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b0], lr=0.05)
        pos_w = (1 - y.mean()) / y.mean().clamp(min=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b0, y, pos_weight=pos_w) \
                + 1e-3 * (w ** 2).sum()
            loss.backward()
            opt.step()
        Xte = torch.tensor((X_by_batch[t] - mu) / sd, dtype=torch.float32)
        scores.append((Xte @ w + b0).detach().numpy())
    return np.concatenate(scores)


def batches(C, rng, n):
    """(kind, indices): spatial neighbourhoods, random sets, and half/half mixes."""
    out = []
    for seed in rng.choice(len(C), 2, replace=False):
        out.append(("spatial", np.sort(np.argsort(np.linalg.norm(C - C[seed], axis=1))[:n])))
    for _ in range(2):
        out.append(("random", np.sort(rng.choice(len(C), n, replace=False))))
    for seed in rng.choice(len(C), 2, replace=False):
        near = np.argsort(np.linalg.norm(C - C[seed], axis=1))[:n // 2]
        rest = rng.choice(np.setdiff1d(np.arange(len(C)), near), n - n // 2, replace=False)
        out.append(("mixed", np.sort(np.concatenate([near, rest]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-shared", type=int, default=30, help="reference edge: >= this many shared tie points")
    ap.add_argument("--out", type=Path, default=Path("runs/egr_affinity"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ref = load_reference_scene(args.scene)
    names = [c.name for c in ref.graph.cameras]
    C = centers(np.stack([c.pose_w2c for c in ref.graph.cameras]))
    pts_per_img = np.bincount(ref.obs_image, minlength=len(names))
    overlap_ref = ref.shared / np.maximum(np.minimum.outer(pts_per_img, pts_per_img), 1)

    ckpt = args.ckpt
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agg = model.aggregator
    kinds = agg.inter_frame_attention_types
    state = {}
    probes = [QKProbe(agg.inter_frame_blocks[b].attn,
                      (lambda L, b=b: PRE if kinds[b] == "register" else state["T"])) for b in range(agg.depth)]

    rng = np.random.default_rng(0)
    rows = []                    # per pair: batch kind, label, ref overlap, scores per (block, pooling), baseline
    for kind, idx in batches(C, rng, args.batch):
        images = load_and_preprocess_images([str(args.scene / "images" / names[g]) for g in idx],
                                            image_resolution=512).to("cuda")
        Hh, Ww = images.shape[-2:]
        n = len(idx)
        state["T"] = PRE + (Hh // 16) * (Ww // 16)
        t = time.time()
        with torch.inference_mode():
            pred = model(images[None])
        ext, intr = encoding_to_camera(pred["pose_enc"], (Hh, Ww))
        dep, conf = pred["depth"][0].float(), pred["depth_conf"][0].float()
        dep, conf = dep.reshape(n, Hh, Ww), conf.reshape(n, Hh, Ww)          # drop a trailing channel dim
        geo = batch_geometry(ext[0].float(), intr[0].float(), dep, conf)
        base = (torch.minimum(geo.covis, geo.covis.T)).cpu().numpy()
        with torch.inference_mode():
            fs = [frame_scores(*p.store, n) for p in probes]
        per_block = [f[0] for f in fs]
        iu, ju = np.triu_indices(n, 1)
        gi, gj = idx[iu], idx[ju]
        rows.append(dict(kind=kind, label=ref.shared[gi, gj] >= args.min_shared, overlap=overlap_ref[gi, gj],
                         base=base[iu, ju],
                         scores={(b, key): s[key][iu, ju] for b, s in enumerate(per_block) for key in s},
                         feats={b: f[1][iu, ju] for b, f in enumerate(fs)}))
        print(f"{kind:8s} batch: {n} images, {int(rows[-1]['label'].sum())}/{len(iu)} reference edges, "
              f"{time.time() - t:.1f} s", flush=True)
        del pred, geo

    def evaluate(sel):
        lab = np.concatenate([r["label"] for r in sel])
        ov = np.concatenate([r["overlap"] for r in sel])
        res = {"pairs": int(len(lab)), "edges": int(lab.sum())}
        def m(score):
            bm = binary_metrics(score, lab)
            return dict(ap=float(bm["average_precision"]), auc=float(bm["roc_auc"]), spearman=float(spearman(score, ov)))
        res["baseline_depth_overlap"] = m(np.concatenate([r["base"] for r in sel]))
        for b in range(agg.depth):
            for key in ("max", "headmean", "mass", "mean", "top8", "mass_cam", "rowz", "kcos"):
                res[f"b{b:02d}_{kinds[b][:3]}_{key}"] = m(np.concatenate([r["scores"][(b, key)] for r in sel]))
            F = np.concatenate([r["feats"][b] for r in sel])
            res[f"b{b:02d}_best_head"] = max(binary_metrics(F[:, h], lab)["average_precision"] for h in range(F.shape[1]))
            if len(sel) > 1:
                res[f"b{b:02d}_{kinds[b][:3]}_probe"] = m(probe_ap([r["feats"][b] for r in sel], [r["label"] for r in sel]))
        if len(sel) > 1:
            regb = [b for b in range(agg.depth) if kinds[b] == "register"]
            res["probe_all_register_blocks"] = m(probe_ap([np.concatenate([r["feats"][b] for b in regb], 1) for r in sel],
                                                          [r["label"] for r in sel]))
            early = [b for b in range(agg.depth) if b <= 9]
            res["probe_blocks_0_9"] = m(probe_ap([np.concatenate([r["feats"][b] for b in early], 1) for r in sel],
                                                 [r["label"] for r in sel]))
        return res

    report = {"all": evaluate(rows)}
    for kind in ("spatial", "random", "mixed"):
        report[kind] = evaluate([r for r in rows if r["kind"] == kind])
    (args.out / "metrics.json").write_text(json.dumps(report, indent=1))

    a = report["all"]
    print(f"\nall batches: {a['pairs']} pairs, {a['edges']} reference edges")
    print(f"baseline (final depth-reprojection overlap): AP {a['baseline_depth_overlap']['ap']:.3f} "
          f"AUC {a['baseline_depth_overlap']['auc']:.3f} rho {a['baseline_depth_overlap']['spearman']:.3f}")
    keys = ("max", "headmean", "mean", "top8", "mass", "mass_cam", "rowz", "kcos", "probe")
    print("AP per pooling (chance = %.3f); best_head = best single head (diagnostic)" % (a["edges"] / a["pairs"]))
    print("blk type     " + " ".join(f"{k:>8s}" for k in keys) + "  best_head")
    for b in range(agg.depth):
        print(f"{b:3d} {kinds[b]:8s} " + " ".join(f"{a[f'b{b:02d}_{kinds[b][:3]}_{k}']['ap']:8.3f}" for k in keys)
              + f"  {a[f'b{b:02d}_best_head']:8.3f}")
    for k in ("probe_all_register_blocks", "probe_blocks_0_9"):
        print(f"{k}: AP {a[k]['ap']:.3f} AUC {a[k]['auc']:.3f} rho {a[k]['spearman']:+.3f}")

if __name__ == "__main__":
    main()
