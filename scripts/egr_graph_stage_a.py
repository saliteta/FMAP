"""Stage A: a fast initial view graph from half of VGGT-Omega (blocks 0-13, one attention head), top-k per image.

Each batch runs only aggregator blocks 0..13. In block 13's cross-frame attention, head 15 with 64 patch queries per
image gives the share of each image's attention on every other image of the batch (own frame dropped, row
renormalised; scripts/egr_attention_mass.py). Per batch, each image keeps its top-k images by that share; an image
in several batches merges its candidates by the largest share relative to uniform (lift = share x (n - 1)) and
keeps the top-k overall. The graph links i-j if j is in i's top-k or i in j's (union); "mutual" needs both.

Batch designs (all images covered; r = batches per image):
    seq     consecutive windows in name (flight) order, stride B/2                          (r = 2, no GPS)
    rand    r random partitions                                                              (no GPS)
    kd      r balanced spatial partitions (recursive median split, random rotation per pass) of the camera
            positions; stands in for GPS, so it is optimistic (reference positions)
Evaluation against the Bentley tie-point graph (edge = >= 30 shared tie points):
    precision@k  share of an image's k picks that are true edges
    strongest    share of an image's k strongest true neighbours (most shared tie points) among its picks
    edge P / R   precision and recall of the union graph's edges (recall is capped: true degree >> k)
    components   connected components of the union graph, and the largest one's share of the images
For HAV, the current VGGT graph's measured overlap (runs/e2e_HAV/graph/graph.npz) is scored the same way.

    python scripts/egr_graph_stage_a.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_attention_mass import PRE, pooled_mass  # noqa: E402
from egr_global_removal_depth import load_scene_reference  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402

N_PATCH_Q = 64


class HeadProbe:
    """Wraps block b's inter-frame attention; computes one head's frame-to-frame attention share (patch queries)."""

    def __init__(self, attn, head, tokens_per_frame):
        self.mass, self.tpf = None, tokens_per_frame
        orig = attn.compute_attention

        def compute_attention(qkv, attn_bias=None, rope=None):
            B, L, _ = qkv.shape
            C, H = attn.qkv.in_features, attn.num_heads
            q, k, _ = torch.unbind(qkv.reshape(B, L, 3, H, C // H), 2)
            q, k = q[0, :, head:head + 1].transpose(0, 1), k[0, :, head:head + 1].transpose(0, 1)   # (1, L, d)
            if attn.use_qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)
            T = self.tpf()
            n = L // T
            base = torch.arange(n, device=q.device)[:, None] * T
            idx = base + PRE + torch.linspace(0, T - PRE - 1, N_PATCH_Q, device=q.device).long()
            self.mass = pooled_mass(q.float(), k.float(), T, {"patch": idx}, q.shape[-1] ** -0.5)["patch"][0]
            return orig(qkv, attn_bias=attn_bias, rope=rope)
        attn.compute_attention = compute_attention


def load_images(root: Path, names, cache_dir: Path):
    scene = root.parent.name if root.name == "colmap_metrics" else root.name
    cache = cache_dir / f"vggt_omega512_{scene}.pt"
    if cache.exists():
        x = torch.load(cache)                         # written in sorted-name order
        assert len(x) == len(names)
        return x
    chunks = []
    for s in range(0, len(names), 64):
        chunks.append(load_and_preprocess_images([str(root / "images" / n) for n in names[s:s + 64]],
                                                 image_resolution=512).to(torch.float16))
    x = torch.cat(chunks)
    torch.save(x, cache)
    return x


def kd_partition(P, B, rng):
    """balanced spatial partition of points P (n, 2) into cells of <= B by recursive median splits."""
    th = rng.uniform(0, np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = P @ R.T
    cells, todo = [], [np.arange(len(P))]
    while todo:
        ids = todo.pop()
        if len(ids) <= B:
            cells.append(ids)
            continue
        ax = int(np.argmax(Q[ids].max(0) - Q[ids].min(0)))
        order = ids[np.argsort(Q[ids, ax])]
        h = len(order) // 2
        todo += [order[:h], order[h:]]
    return cells


def design_batches(design, n, B, C, rng, r):
    if design == "seq":
        starts = list(range(0, max(n - B, 0) + 1, B // 2))
        if starts[-1] + B < n:
            starts.append(n - B)
        return [np.arange(s, min(s + B, n)) for s in starts]
    if design == "rand":
        out = []
        for _ in range(r):
            perm = rng.permutation(n)
            k = int(np.ceil(n / B))
            out += [np.sort(p) for p in np.array_split(perm, k)]
        return out
    if design == "kd":
        out = []
        for _ in range(r):
            out += [np.sort(c) for c in kd_partition(C[:, :2] - C[:, :2].mean(0), B, rng)]
        return out
    raise ValueError(design)


def topk_graph(cands, n, k):
    """cands[i] = {j: best lift}; returns picks (n, k) (-1 padded) and the union adjacency."""
    picks = -np.ones((n, k), int)
    A = np.zeros((n, n), bool)
    for i in range(n):
        if cands[i]:
            js = sorted(cands[i], key=lambda j: -cands[i][j])[:k]
            picks[i, :len(js)] = js
            A[i, js] = True
    return picks, A


def evaluate(picks, A, shared, min_shared, k):
    n = len(shared)
    G = shared >= min_shared
    np.fill_diagonal(G, False)
    prec, strong = [], []
    for i in range(n):
        p = picks[i][picks[i] >= 0]
        if len(p) == 0:
            continue
        prec.append(G[i, p].mean())
        s = shared[i].copy()
        s[i] = -1
        true_top = np.argsort(-s)[:k]
        true_top = true_top[G[i, true_top]]
        if len(true_top):
            strong.append(np.isin(true_top, p).mean())
    U = A | A.T
    M = A & A.T
    iu = np.triu_indices(n, 1)
    out = {}
    for name, E in (("union", U), ("mutual", M)):
        e, g = E[iu], G[iu]
        ncomp, lab = connected_components(E, directed=False)
        out[name] = dict(edges=int(e.sum()), edge_precision=float(g[e].mean()) if e.any() else 0.0,
                         edge_recall=float(e[g].mean()), components=int(ncomp),
                         largest_component=float(np.bincount(lab).max() / n),
                         isolated=int((E.sum(1) == 0).sum()))
    out["precision_at_k"] = float(np.mean(prec))
    out["strongest_found"] = float(np.mean(strong))
    out["true_degree_median"] = float(np.median(G.sum(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--designs", default="seq,rand,kd")
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--r", type=int, default=2, help="batches per image for rand / kd")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--block", type=int, default=13)
    ap.add_argument("--head", type=int, default=15)
    ap.add_argument("--min-shared", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("runs/egr_graph_stage_a"))
    ap.add_argument("--cache", type=Path, default=Path("runs/cache"))
    args = ap.parse_args()
    scene = args.scene.parent.name if args.scene.name == "colmap_metrics" else args.scene.name
    out_dir = args.out / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    names, C, shared = load_scene_reference(args.scene)
    assert names == sorted(names)
    n = len(names)
    t0 = time.time()
    images = load_images(args.scene, names, args.cache)
    t_load = time.time() - t0
    print(f"{scene}: {n} images, loaded in {t_load:.0f} s", flush=True)

    from huggingface_hub import hf_hub_download
    model = VGGTOmega(enable_camera=False, enable_depth=False).eval()
    model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
                                     map_location="cpu"), strict=False)
    agg = model.aggregator.to("cuda")
    agg.depth = args.block + 1                     # blocks after the probe block are never run
    state = {}
    probe = HeadProbe(agg.inter_frame_blocks[args.block].attn, args.head, lambda: state["T"])
    Hh, Ww = images.shape[-2:]
    state["T"] = PRE + (Hh // 16) * (Ww // 16)

    results = {}
    rng = np.random.default_rng(0)
    for B in args.batch:
        for design in args.designs.split(","):
            plan = design_batches(design, n, B, C, rng, args.r)
            cands = [dict() for _ in range(n)]
            torch.cuda.synchronize()
            t = time.time()
            for idx in plan:
                x = images[idx].float().cuda()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    agg(x[None])
                M = probe.mass.astype(np.float64)
                np.fill_diagonal(M, 0)
                M = M / M.sum(1, keepdims=True) * (len(idx) - 1)               # lift; 1 = uniform
                for a in range(len(idx)):
                    for b_ in np.argsort(-M[a])[:args.k]:
                        gi, gj = int(idx[a]), int(idx[b_])
                        cands[gi][gj] = max(cands[gi].get(gj, 0.0), float(M[a, b_]))
            torch.cuda.synchronize()
            t_inf = time.time() - t
            picks, A = topk_graph(cands, n, args.k)
            ev = evaluate(picks, A, shared, args.min_shared, args.k)
            ev.update(batches=len(plan), images_per_batch=B, inference_s=t_inf)
            key = f"{design}_B{B}"
            results[key] = ev
            u = ev["union"]
            print(f"  {key:9s}: {len(plan):3d} batches, {t_inf:5.1f} s | precision@{args.k} {ev['precision_at_k']:.3f} "
                  f"strongest {ev['strongest_found']:.3f} | union: {u['edges']} edges, P {u['edge_precision']:.3f} "
                  f"R {u['edge_recall']:.3f}, {u['components']} comp (largest {u['largest_component']:.0%}) | "
                  f"mutual: {ev['mutual']['components']} comp (largest {ev['mutual']['largest_component']:.0%})",
                  flush=True)
            np.savez(out_dir / f"graph_{key}.npz", picks=picks, adjacency=A, names=np.array(names))

    current = Path("runs/e2e_HAV/graph/graph.npz")
    if scene == "HAV" and current.exists():          # the current VGGT graph's overlap, same top-k rule
        g = np.load(current, allow_pickle=True)
        assert list(g["names"]) == names
        meas = g["measured"].copy()
        np.fill_diagonal(meas, -np.inf)
        cands = [{int(j): float(meas[i, j]) for j in np.argsort(-meas[i])[:args.k] if meas[i, j] > 0}
                 for i in range(n)]
        picks, A = topk_graph(cands, n, args.k)
        ev = evaluate(picks, A, shared, args.min_shared, args.k)
        results["current_vggt_graph"] = ev
        u = ev["union"]
        print(f"  current VGGT graph (459 batches, 125 s inference): precision@{args.k} {ev['precision_at_k']:.3f} "
              f"strongest {ev['strongest_found']:.3f} | union: {u['edges']} edges, P {u['edge_precision']:.3f} "
              f"R {u['edge_recall']:.3f}, {u['components']} comp (largest {u['largest_component']:.0%})", flush=True)
    results["load_s"] = t_load
    G = shared >= args.min_shared
    np.fill_diagonal(G, False)
    results["true_degree_median"] = float(np.median(G.sum(1)))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
