"""Within-batch, per-query-frame evaluation of VGGT-Omega's inter-frame attention as an image-relation signal.

The softmax of an attention row makes the keys compete for the query's budget, so a post-softmax score is only
comparable within the row it came from, and never across batches. Here, per inter-frame block and head:

  M_h[i, j] = attention mass that frame i's query tokens send to ALL tokens of frame j (summed over j's keys,
              averaged over i's queries), from the full softmax of that block
  query sets: cam (camera token), reg (16 registers), patch (64 evenly spaced patches; global blocks only)

Each score matrix gets its diagonal (self-frame mass) dropped and its rows renormalised. Then:

  row AP / R-precision : for each query frame i, rank the other frames j against the reference graph; mean over
                         frames (rows with no reference edge or no non-edge are skipped)
  batch AP / rho       : (M + M^T)/2 within one batch vs the reference graph / tie-point overlap; mean over batches
  heads: summed, and the best single head picked by leave-one-batch-out row AP (choice made without the test batch)

Reference graph: Bentley AT tie points (edge >= 30 shared). Baseline: VGGT-Omega's final depth-reprojection
overlap, scored the same way. Batches: the 6 of scripts/egr_register_affinity.py (seed 0, 64 images).

    python scripts/egr_attention_mass.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_register_affinity import batches  # noqa: E402
from fgsfm.eval.overlap_metrics import spearman  # noqa: E402
from fgsfm.fm.overlap import batch_geometry  # noqa: E402
from fgsfm.geometry.transforms import centers  # noqa: E402
from fgsfm.io.scene import load_reference_scene  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

PRE = 17                                   # camera token + 16 registers per frame
N_PATCH_Q = 64                             # patch queries per frame in global blocks
CHUNK = 2048


class MassProbe:
    """Wraps one inter-frame attention; computes per-head frame-to-frame attention mass inside the forward."""

    def __init__(self, attn, tokens_per_frame):
        self.attn, self.tpf, self.mass = attn, tokens_per_frame, None
        orig = attn.compute_attention

        def compute_attention(qkv, attn_bias=None, rope=None):
            B, L, _ = qkv.shape
            C, H = attn.qkv.in_features, attn.num_heads
            q, k, _ = torch.unbind(qkv.reshape(B, L, 3, H, C // H), 2)
            q, k = q[0].transpose(0, 1), k[0].transpose(0, 1)          # (H, L, d)
            if attn.use_qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)
            self.mass = frame_mass(q.float(), k.float(), self.tpf(L))
            return orig(qkv, attn_bias=attn_bias, rope=rope)
        attn.compute_attention = compute_attention


def frame_mass(q, k, T):
    """q, k (H, L, d), frames laid out contiguously with T tokens each -> {query set: (H, n, n) numpy}."""
    H, L, d = q.shape
    n = L // T
    base = torch.arange(n, device=q.device)[:, None] * T
    sets = {"cam": base + 0, "reg": base + torch.arange(1, PRE, device=q.device)}
    if T > PRE:
        sets["patch"] = base + PRE + torch.linspace(0, T - PRE - 1, N_PATCH_Q, device=q.device).long()
    out = {}
    for name, idx in sets.items():
        m = idx.shape[1]
        flat = idx.reshape(-1)
        res = torch.empty(H, n, n, device=q.device)
        for h in range(H):
            acc = []
            for s in range(0, len(flat), CHUNK):
                a = torch.softmax(q[h, flat[s:s + CHUNK]] @ k[h].T / d ** 0.5, dim=-1)   # (c, L)
                acc.append(a.view(-1, n, T).sum(-1))                                     # (c, n) mass per frame
            res[h] = torch.cat(acc).view(n, m, n).mean(1)
        out[name] = res.cpu().numpy()
    return out


def ap(score, label):
    order = np.argsort(-score, kind="stable")
    g = label[order]
    tp = np.cumsum(g)
    return float((tp[g] / (np.flatnonzero(g) + 1)).mean())


def row_norm(M):
    M = M.astype(np.float64).copy()
    np.fill_diagonal(M, 0)
    return M / np.maximum(M.sum(1, keepdims=True), 1e-12)


def row_metrics(M, G):
    """mean over query frames of AP and R-precision (other frames ranked by M[i])."""
    aps, rps = [], []
    for i in range(len(M)):
        j = np.arange(len(M)) != i
        s, g = M[i, j], G[i, j]
        if 0 < g.sum() < len(g):
            aps.append(ap(s, g))
            rps.append(float(g[np.argsort(-s)[:g.sum()]].mean()))
    return float(np.mean(aps)), float(np.mean(rps))


def batch_metrics(M, G, O):
    S = 0.5 * (M + M.T)
    iu = np.triu_indices(len(M), 1)
    return ap(S[iu], G[iu]), float(spearman(S[iu], O[iu]))


def score_all(M, G, O):
    ra, rp = row_metrics(M, G)
    ba, rho = batch_metrics(M, G, O)
    return dict(row_ap=ra, row_rprec=rp, batch_ap=ba, batch_rho=rho)


def mean_dict(ds):
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--scene", type=Path, required=True)
    ap_.add_argument("--ckpt", default=None)
    ap_.add_argument("--batch", type=int, default=64)
    ap_.add_argument("--min-shared", type=int, default=30)
    ap_.add_argument("--out", type=Path, default=Path("runs/egr_attention_mass"))
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ref = load_reference_scene(args.scene)
    names = [c.name for c in ref.graph.cameras]
    C = centers(np.stack([c.pose_w2c for c in ref.graph.cameras]))
    pts = np.bincount(ref.obs_image, minlength=len(names))
    overlap_ref = ref.shared / np.maximum(np.minimum.outer(pts, pts), 1)

    ckpt = args.ckpt
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agg = model.aggregator
    kinds = agg.inter_frame_attention_types
    state = {}
    probes = [MassProbe(agg.inter_frame_blocks[b].attn,
                        (lambda L, b=b: PRE if kinds[b] == "register" else state["T"])) for b in range(agg.depth)]

    rng = np.random.default_rng(0)
    data = []                   # per batch: kind, G, O, baseline directed overlap, mass[block][set] (H, n, n)
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
            dep = pred["depth"][0].float().reshape(n, Hh, Ww)
            conf = pred["depth_conf"][0].float().reshape(n, Hh, Ww)
            geo = batch_geometry(ext[0].float(), intr[0].float(), dep, conf)
        G = ref.shared[np.ix_(idx, idx)] >= args.min_shared
        np.fill_diagonal(G, False)
        data.append(dict(kind=kind, idx=idx, G=G, O=overlap_ref[np.ix_(idx, idx)], base=geo.covis.cpu().numpy(),
                         mass=[p.mass for p in probes]))
        rows_with_edges = int(((G.sum(1) > 0) & (G.sum(1) < n - 1)).sum())
        print(f"{kind:8s} batch: {n} images, {int(G.sum()) // 2} reference edges, {rows_with_edges} scorable rows, "
              f"{time.time() - t:.1f} s", flush=True)
        del pred, geo
        torch.cuda.empty_cache()

    torch.save(dict(names=names, data=[dict(d, mass=[{k: v.astype(np.float16) for k, v in m.items()} for m in d["mass"]])
                                       for d in data]), args.out / "mass.pt")
    nb = len(data)
    report = {"chance": {}, "baseline": {}, "blocks": {}}
    for d in data:
        d["chance"] = float(np.mean([g.sum() / (len(g) - 1) for g in d["G"] if 0 < g.sum() < len(g) - 1]))
    report["chance"] = {"row_ap_mean_pos_rate": float(np.mean([d["chance"] for d in data])),
                        "batch_edge_rate": float(np.mean([d["G"].sum() / (len(d["G"]) * (len(d["G"]) - 1))
                                                          for d in data]))}
    per_batch_base = [score_all(row_norm(d["base"]), d["G"], d["O"]) for d in data]
    report["baseline"] = dict(mean=mean_dict(per_batch_base), per_batch=per_batch_base)

    for b in range(agg.depth):
        blk = {"type": kinds[b]}
        for qs in data[0]["mass"][b]:
            # per batch, per head row AP (for leave-one-batch-out head choice)
            head_row = np.array([[row_metrics(row_norm(d["mass"][b][qs][h]), d["G"])[0]
                                  for h in range(d["mass"][b][qs].shape[0])] for d in data])   # (nb, H)
            summed = [score_all(row_norm(d["mass"][b][qs].sum(0)), d["G"], d["O"]) for d in data]
            lobo = []
            for t in range(nb):
                h = int(np.argmax(np.delete(head_row, t, 0).mean(0)))
                lobo.append(dict(head=h, **score_all(row_norm(data[t]["mass"][b][qs][h]), data[t]["G"], data[t]["O"])))
            blk[qs] = dict(sum_heads=mean_dict(summed),
                           lobo_head=mean_dict([{k: v for k, v in r.items() if k != "head"} for r in lobo]),
                           lobo_heads_chosen=[r["head"] for r in lobo],
                           sum_heads_per_batch=summed,
                           best_head_oracle=float(head_row.mean(0).max()))
        report["blocks"][b] = blk
    (args.out / "metrics.json").write_text(json.dumps(report, indent=1))

    kind_names = [d["kind"] for d in data]
    print(f"\nbatches: {kind_names}")
    print(f"chance: row AP ~{report['chance']['row_ap_mean_pos_rate']:.3f}, "
          f"batch AP ~{report['chance']['batch_edge_rate']:.3f}")
    bm = report["baseline"]["mean"]
    print(f"baseline depth overlap: row AP {bm['row_ap']:.3f} R-prec {bm['row_rprec']:.3f} | "
          f"batch AP {bm['batch_ap']:.3f} rho {bm['batch_rho']:.3f}")
    print("\nblk type     | query  heads-summed: rowAP  Rprec batchAP  rho | LOBO head: rowAP batchAP")
    for b in range(agg.depth):
        blk = report["blocks"][b]
        for qs in ("cam", "reg", "patch"):
            if qs not in blk:
                continue
            s, l = blk[qs]["sum_heads"], blk[qs]["lobo_head"]
            print(f"{b:3d} {blk['type']:8s} | {qs:5s}  {s['row_ap']:.3f}  {s['row_rprec']:.3f}  {s['batch_ap']:.3f}  "
                  f"{s['batch_rho']:+.2f} | {l['row_ap']:.3f}  {l['batch_ap']:.3f}", flush=True)
    print("\nper batch row AP (registers, heads summed) for register blocks + baseline:")
    print("      " + " ".join(f"{k:>8s}" for k in kind_names))
    print("base  " + " ".join(f"{r['row_ap']:8.3f}" for r in per_batch_base))
    for b in range(agg.depth):
        if kinds[b] == "register" or b in (15, 23):
            r = report["blocks"][b]["reg"]["sum_heads_per_batch"]
            print(f"b{b:02d}   " + " ".join(f"{x['row_ap']:8.3f}" for x in r))


if __name__ == "__main__":
    main()
