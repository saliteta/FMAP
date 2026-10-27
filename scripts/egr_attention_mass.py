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
POOLS = ("mean", "qmax", "max")     # pooling of the token-to-token weights of a frame pair


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
    """Frame-to-frame attention mass of one inter-frame block, per head.

    q, k: (H, L, d) queries / keys of EVERY token this block attends over, for ALL n frames of the batch (after
    QK-norm, exactly what the block's softmax sees). Frames are laid out contiguously, T tokens each:
        register block: T = 17   (camera + 16 registers per frame; patches are not in the sequence), L = 17 n
        global block  : T = 1031 (camera + 16 registers + 1014 patches),                             L = 1031 n
    Query sets per frame: "cam" (camera token), "reg" (16 registers), "patch" (64 evenly spaced patches, global
    blocks only). Returns pooled_mass(...) of those sets; in global blocks also split by the key token type.
    """
    n = q.shape[1] // T
    # token positions of frame f start at f * T: offset 0 = camera, 1..16 = registers, 17.. = patches
    base = torch.arange(n, device=q.device)[:, None] * T                          # (n, 1) first token of each frame
    sets = {"cam": base + 0,                                                      # (n, 1)  camera token
            "reg": base + torch.arange(1, PRE, device=q.device)}                  # (n, 16) registers
    key_groups = {"cam": slice(0, 1), "reg": slice(1, PRE)}                       # mass split by key token type
    if T > PRE:                                                                   # global block: also some patches
        sets["patch"] = base + PRE + torch.linspace(0, T - PRE - 1, N_PATCH_Q, device=q.device).long()  # (n, 64)
        key_groups["patch"] = slice(PRE, T)
    return pooled_mass(q, k, T, sets, q.shape[-1] ** -0.5, key_groups)


def pooled_mass(q, k, T, sets, scale, key_groups=None):
    """Pooled frame-to-frame attention weights, per head.

    q, k: (H, L, d) with n frames laid out contiguously, T tokens each; sets: {name: (n, m) query token positions};
    the softmax of each query runs over ALL L keys (every token of every frame, own frame included), with logits
    scale * q.k. For head h and frame pair (i, j), the weights from the m query tokens of frame i to the T key
    tokens of frame j (e.g. 16 x 17 in a register block) are pooled into:
      "<set>"        sum over j's key tokens, mean over i's query tokens: the share of i's attention on frame j
                     (each row sums to 1 over j, frame i itself included)
      "<set>_qmax"   sum over j's key tokens, max over i's query tokens: the query token that leans most on j
      "<set>_max"    max over all (query token, key token) weights: the single strongest token-to-token link
      "<set>><grp>"  as "<set>", but only over j's key tokens of one type (key_groups: {grp: slice of offsets})
    Returns {name: (H, n, n) numpy}.
    """
    H, L, _ = q.shape
    n = L // T
    key_groups = key_groups or {}
    out = {}
    for name, idx in sets.items():
        m = idx.shape[1]                                  # query tokens per frame in this set
        flat = idx.reshape(-1)                            # (n*m,) frame 0's queries, then frame 1's, ...
        res = {key: torch.empty(H, n, n, device=q.device)
               for key in [name, f"{name}_qmax", f"{name}_max"] + [f"{name}>{g}" for g in key_groups]}
        for h in range(H):                                # every head has its own softmax, kept separate
            acc = {"sum": [], "max": [], **{g: [] for g in key_groups}}
            for s in range(0, len(flat), CHUNK):          # chunk the queries to bound memory (c x L at a time)
                # the block's real attention rows for these queries: softmax over ALL L keys of the batch,
                # i.e. every token of every frame (own frame included), as in scaled_dot_product_attention
                a = torch.softmax(q[h, flat[s:s + CHUNK]] @ k[h].T * scale, dim=-1)       # (c, L), rows sum to 1
                a = a.view(-1, n, T)                      # keys grouped by frame: (c, key frame, key token)
                acc["sum"].append(a.sum(-1))              # (c, n) mass per key frame (sum over its T tokens)
                acc["max"].append(a.amax(-1))             # (c, n) strongest single key token of each key frame
                for g, sl in key_groups.items():
                    acc[g].append(a[..., sl].sum(-1))     # (c, n) mass on key frame's tokens of type g
            per_q = {key: torch.cat(v).view(n, m, n) for key, v in acc.items()}   # (query frame, query token, key frame)
            res[name][h] = per_q["sum"].mean(1)           # sum over key tokens, mean over query tokens (default)
            res[f"{name}_qmax"][h] = per_q["sum"].amax(1)  # sum over key tokens, max over query tokens
            res[f"{name}_max"][h] = per_q["max"].amax(1)   # max over all (query token x key token) weights
            for g in key_groups:
                res[f"{name}>{g}"][h] = per_q[g].mean(1)
        out.update({key: r.cpu().numpy() for key, r in res.items()})
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


def evaluate_scores(data, get, keys):
    """{key: metrics} for per-head scores get(d, key) -> (H, n, n): heads summed, and the best single head chosen
    leave-one-batch-out by row AP (the choice never sees the batch it is scored on)."""
    nb, res = len(data), {}
    for key in keys:
        head_row = np.array([[row_metrics(row_norm(get(d, key)[h]), d["G"])[0]
                              for h in range(get(d, key).shape[0])] for d in data])   # (nb, H)
        summed = [score_all(row_norm(get(d, key).sum(0)), d["G"], d["O"]) for d in data]
        lobo = []
        for t in range(nb):
            h = int(np.argmax(np.delete(head_row, t, 0).mean(0)))
            lobo.append(dict(head=h, **score_all(row_norm(get(data[t], key)[h]), data[t]["G"], data[t]["O"])))
        res[key] = dict(sum_heads=mean_dict(summed),
                        lobo_head=mean_dict([{k: v for k, v in r.items() if k != "head"} for r in lobo]),
                        lobo_heads_chosen=[r["head"] for r in lobo],
                        sum_heads_per_batch=summed,
                        best_head_oracle=float(head_row.mean(0).max()))
    return res


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
        blk.update(evaluate_scores(data, lambda d, key, b=b: d["mass"][b][key], data[0]["mass"][b].keys()))
        report["blocks"][b] = blk
    (args.out / "metrics.json").write_text(json.dumps(report, indent=1))

    kind_names = [d["kind"] for d in data]
    print(f"\nbatches: {kind_names}")
    print(f"chance: row AP ~{report['chance']['row_ap_mean_pos_rate']:.3f}, "
          f"batch AP ~{report['chance']['batch_edge_rate']:.3f}")
    bm = report["baseline"]["mean"]
    print(f"baseline depth overlap: row AP {bm['row_ap']:.3f} R-prec {bm['row_rprec']:.3f} | "
          f"batch AP {bm['batch_ap']:.3f} rho {bm['batch_rho']:.3f}")
    cols = ["", "_qmax", "_max", ">cam", ">reg", ">patch"]
    print("\nrow AP of the best head (leave-one-batch-out); pooling of a frame pair's token-to-token weights:")
    print("  mean = sum over key tokens, mean over query tokens | qmax = max over query tokens | max = max over all")
    print("  >cam/>reg/>patch = mean, only over the key frame's camera / register / patch tokens")
    print("blk type     query |   mean   qmax    max   >cam   >reg >patch | heads summed (mean)")
    for b in range(agg.depth):
        blk = report["blocks"][b]
        for qs in ("cam", "reg", "patch"):
            if qs not in blk:
                continue
            vals = [f"{blk[qs + c]['lobo_head']['row_ap']:6.3f}" if qs + c in blk else "     -" for c in cols]
            print(f"{b:3d} {blk['type']:8s} {qs:5s} | " + " ".join(vals) +
                  f" | {blk[qs]['sum_heads']['row_ap']:.3f}", flush=True)
    print("\nper batch row AP (registers, heads summed) for register blocks + baseline:")
    print("      " + " ".join(f"{k:>8s}" for k in kind_names))
    print("base  " + " ".join(f"{r['row_ap']:8.3f}" for r in per_batch_base))
    for b in range(agg.depth):
        if kinds[b] == "register" or b in (15, 23):
            r = report["blocks"][b]["reg"]["sum_heads_per_batch"]
            print(f"b{b:02d}   " + " ".join(f"{x['row_ap']:8.3f}" for x in r))


if __name__ == "__main__":
    main()
