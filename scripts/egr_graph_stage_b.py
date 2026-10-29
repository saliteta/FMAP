"""Stage B: initial poses from full VGGT-Omega run once per graph cluster, merged by Sim(3).

Input: a stage-A graph (scripts/egr_graph_stage_a.py, graph_<design>_B<B>.npz; union of the per-image top-k).
    clusters  every image's ball = the first B images of a breadth-first search from it on the graph (within a hop
              layer, images with more links into the ball come first). Greedy set cover over the balls: take the
              ball covering most uncovered images until all are covered. Balls next to covered ground pull in
              covered images, which gives the overlap the merge needs.
    frame 0   the ball's centre: the image with the smallest mean hop distance to the rest of the cluster
              (ties: most links). VGGT-Omega's poses are about 2x better for pairs with frame 0, and errors grow
              with hops from it.
    inference full VGGT-Omega once per cluster, frame 0 first: poses and depth.
    merge     maximum spanning tree of the cluster-overlap graph from the largest cluster; each child is taken into
              its parent's frame by a Sim(3) from the shared cameras (rotation: chordal mean; scale: median log
              depth ratio on the shared images; translation: median; fgsfm.geometry.transforms
              .fit_sim3_from_cameras). An image in several clusters keeps the pose from the cluster where it is
              fewest hops from frame 0.
Per cluster, the overlap of every pair is measured from the depth as in the graph builder (fgsfm.fm.overlap,
symmetric; max over clusters), so graph.npz can drive scripts/run_global_ba.py --graph-npz like a builder checkpoint.
Evaluation: fgsfm.eval.pose_metrics of the merged poses, and of each cluster alone, against the reference AT poses.
For HAV the current graph builder's numbers are printed from runs/e2e_HAV/graph/metrics.json.

    python scripts/egr_graph_stage_b.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_graph_stage_a import load_images  # noqa: E402
from fgsfm.eval.pose_metrics import pose_metrics  # noqa: E402
from fgsfm.fm.overlap import batch_geometry, symmetric_overlap  # noqa: E402
from fgsfm.geometry.transforms import Sim3, centers, fit_sim3_from_cameras, robust_umeyama, to4x4  # noqa: E402
from fgsfm.io.colmap_text import ColmapModel  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

DEPTH_STRIDE = 8                 # depth kept at 1/8 resolution for the merge scale
OVERLAP_STRIDE = 8               # pixel stride of the overlap measurement (64 x 64 pairs per cluster)


def ball(adj, src, B):
    """first B images of a BFS from src; within a hop layer, images with more links into the ball first."""
    out, seen, layer = [src], {src}, [src]
    while layer and len(out) < B:
        nxt = {j for i in layer for j in adj[i] if j not in seen}
        S = set(out)
        nxt = sorted(nxt, key=lambda j: (-len(adj[j] & S), j))
        nxt = nxt[:B - len(out)]
        out += nxt
        seen.update(nxt)
        layer = nxt
    return out


def hops_from(adj, src, members):
    """hop distances from src inside the subgraph of members."""
    d, q = {src: 0}, deque([src])
    while q:
        i = q.popleft()
        for j in adj[i] & members:
            if j not in d:
                d[j] = d[i] + 1
                q.append(j)
    return d


def cover(adj, n, B):
    balls = [ball(adj, v, B) for v in range(n)]
    covered = np.zeros(n, bool)
    clusters = []
    while not covered.all():
        gain = [(~covered[b]).sum() for b in balls]
        v = int(np.argmax(gain))
        members = set(balls[v])
        best = None
        for c in members:                                # centre: smallest mean hop, then most links
            d = hops_from(adj, c, members)
            key = (len(members) - len(d), np.mean(list(d.values())), -len(adj[c] & members))
            if best is None or key < best[0]:
                best = (key, c, d)
        _, f0, d = best
        order = [f0] + sorted(members - {f0}, key=lambda j: (d.get(j, 99), j))
        clusters.append(dict(images=order, hops=[d.get(j, 99) for j in order]))
        covered[balls[v]] = True
    return clusters


def reference_w2c(root):
    ims = ColmapModel.load(root / "sparse" / "0").sorted_images()
    return [im.name for im in ims], np.stack([np.r_[np.c_[im.R, im.tvec], [[0, 0, 0, 1]]] for im in ims])


def merge(clusters, n):
    """Sim(3) of every cluster into the root cluster's frame along the max-overlap spanning tree."""
    m = len(clusters)
    sets = [set(c["images"]) for c in clusters]
    ov = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            ov[a, b] = len(sets[a] & sets[b])
    tree = minimum_spanning_tree(-(ov + ov.T) * (ov + ov.T > 0)).toarray()
    tree = (tree + tree.T) != 0
    root = int(np.argmax([len(s) for s in sets]))
    to_root = {root: Sim3.identity()}
    links, q = [], deque([root])
    while q:
        p = q.popleft()
        for c in np.flatnonzero(tree[p]):
            c = int(c)
            if c in to_root:
                continue
            shared = sorted(sets[p] & sets[c])
            fit = _link_fit(clusters[p], clusters[c], shared)
            to_root[c] = to_root[p].compose(fit.sim3)
            links.append(dict(parent=p, child=c, shared=len(shared), inliers=fit.num_inliers,
                              rot_res_median=float(np.median(fit.rot_res_deg)),
                              scale_spread=fit.scale_spread))
            q.append(c)
    return (*place(clusters, n, to_root), links, len(to_root))


def _link_fit(P, C, shared):
    """Sim(3) taking cluster C's frame into cluster P's frame from their shared images."""
    ip = [P["images"].index(g) for g in shared]
    ic = [C["images"].index(g) for g in shared]
    ratios = []
    for a, b in zip(ip, ic):
        dp, dc = P["depth"][a], C["depth"][b]
        ok = P["conf_ok"][a] & C["conf_ok"][b] & (dp > 0) & (dc > 0)
        ratios.append(np.log(dp[ok] / dc[ok]))
    return fit_sim3_from_cameras(P["w2c"][ip], C["w2c"][ic], np.concatenate(ratios))


def place(clusters, n, to_world):
    """Every image's pose from the placed cluster where it is fewest hops from frame 0.
    Returns w2c, K and src: (cluster, index in cluster) per image, (-1, -1) if unplaced."""
    w2c, K = np.full((n, 4, 4), np.nan), np.full((n, 3, 3), np.nan)
    best_hop, src = np.full(n, 1 << 30), np.full((n, 2), -1)
    for ci, C in enumerate(clusters):
        if ci not in to_world:
            continue
        cam = to_world[ci].apply_w2c(C["w2c"])
        for k, (g, h) in enumerate(zip(C["images"], C["hops"])):
            if h < best_hop[g]:
                best_hop[g], w2c[g], K[g], src[g] = h, cam[k], C["K"][k], (ci, k)
    return w2c, K, src


def merge_gps(clusters, n, gps, min_spread=0.1):
    """Sim(3) of every cluster straight into the GPS (ENU metre) frame: trimmed Umeyama of its camera centres onto
    their GPS positions. No chain, so one bad link cannot carry a block away, and clusters that share no images
    with the rest are placed too. A cluster whose GPS positions are nearly collinear (second singular value below
    min_spread of the first: the roll about the line is unconstrained) is instead fitted to its most-overlapping
    GPS-placed neighbour by fit_sim3_from_cameras, as in merge()."""
    sets = [set(c["images"]) for c in clusters]
    to_world, links, pending = {}, [], []
    for ci, C in enumerate(clusters):
        G = gps[C["images"]]
        sv = np.linalg.svd(G - G.mean(0), compute_uv=False)
        if sv[1] < min_spread * sv[0]:
            pending.append(ci)
            continue
        S = robust_umeyama(centers(C["w2c"]), G)
        res = np.linalg.norm(S.apply_points(centers(C["w2c"])) - G, axis=1)
        to_world[ci] = S
        links.append(dict(child=ci, gps=True, res_median=float(np.median(res)), res_max=float(res.max())))
    while pending:
        best = max(((len(sets[c] & sets[p]), c, p) for c in pending for p in to_world), default=(0, None, None))
        if best[0] < 3:                       # nothing placed overlaps: GPS fit anyway
            for ci in pending:
                to_world[ci] = robust_umeyama(centers(clusters[ci]["w2c"]), gps[clusters[ci]["images"]])
                links.append(dict(child=ci, gps=True, degenerate=True))
            break
        _, c, p = best
        fit = _link_fit(clusters[p], clusters[c], sorted(sets[p] & sets[c]))
        to_world[c] = to_world[p].compose(fit.sim3)
        links.append(dict(parent=p, child=c, gps=False, shared=best[0], inliers=fit.num_inliers))
        pending.remove(c)
    return (*place(clusters, n, to_world), links, len(to_world))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--graph", default="kd_B64", help="stage-A graph name")
    ap.add_argument("--batch", type=int, default=64, help="images per cluster")
    ap.add_argument("--stage-a", type=Path, default=Path("runs/egr_graph_stage_a"))
    ap.add_argument("--out", type=Path, default=Path("runs/egr_graph_stage_b"))
    ap.add_argument("--cache", type=Path, default=Path("runs/cache"))
    args = ap.parse_args()
    scene = args.scene.parent.name if args.scene.name == "colmap_metrics" else args.scene.name
    out_dir = args.out / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    g = np.load(args.stage_a / scene / f"graph_{args.graph}.npz")
    names = [str(s) for s in g["names"]]
    ref_names, ref_w2c = reference_w2c(args.scene)
    assert ref_names == names
    n = len(names)
    U = g["adjacency"] | g["adjacency"].T
    adj = [set(np.flatnonzero(U[i]).tolist()) for i in range(n)]

    t = time.time()
    clusters = cover(adj, n, args.batch)
    t_cluster = time.time() - t
    sizes = [len(c["images"]) for c in clusters]
    print(f"{scene}: {n} images, {len(clusters)} clusters (sizes {min(sizes)}-{max(sizes)}, "
          f"{sum(sizes) / n:.2f} runs per image, max hop {max(max(c['hops']) for c in clusters)}), "
          f"clustering {t_cluster:.1f} s", flush=True)

    images = load_images(args.scene, names, args.cache)
    from huggingface_hub import hf_hub_download
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
                                     map_location="cpu"))
    measured, cobatched = np.zeros((n, n)), np.zeros((n, n), bool)
    t_overlap = 0.0
    torch.cuda.synchronize()
    t = time.time()
    for C in clusters:
        x = images[C["images"]].float().cuda()
        with torch.inference_mode():
            pred = model(x[None])
        ext, intr = encoding_to_camera(pred["pose_enc"], x.shape[-2:])
        C["w2c"] = to4x4(ext[0].double().cpu().numpy())
        C["K"] = intr[0].double().cpu().numpy()
        t1 = time.time()
        geo = batch_geometry(ext[0].float(), intr[0].float(), pred["depth"][0, ..., 0].float(),
                             pred["depth_conf"][0].float(), stride=OVERLAP_STRIDE)
        ix = np.ix_(C["images"], C["images"])
        measured[ix] = np.maximum(measured[ix], symmetric_overlap(geo.covis).cpu().numpy())
        cobatched[ix] = True
        t_overlap += time.time() - t1
        dep = pred["depth"][0, :, ::DEPTH_STRIDE, ::DEPTH_STRIDE, 0].float().cpu().numpy()
        conf = pred["depth_conf"][0, :, ::DEPTH_STRIDE, ::DEPTH_STRIDE].float().cpu().numpy()
        C["depth"] = dep
        C["conf_ok"] = conf >= np.median(conf.reshape(len(conf), -1), 1)[:, None, None]
    torch.cuda.synchronize()
    t_inf = time.time() - t
    print(f"  inference: {len(clusters)} runs, {t_inf:.1f} s (overlap measurement {t_overlap:.1f} s of it)", flush=True)

    per_cluster = []
    for C in clusters:
        pm = pose_metrics(C["w2c"], ref_w2c[C["images"]])
        per_cluster.append(dict(size=len(C["images"]), frame0=names[C["images"][0]], max_hop=max(C["hops"]),
                                rel_rot_median_deg=pm["rel_rot_median_deg"], AUC5=pm["AUC@5"]))
    rr = np.array([p["rel_rot_median_deg"] for p in per_cluster])
    au = np.array([p["AUC5"] for p in per_cluster])
    print(f"  clusters alone: rel rot median {np.median(rr):.2f} deg (range {rr.min():.2f}-{rr.max():.2f}), "
          f"AUC@5 {np.median(au):.3f} (range {au.min():.3f}-{au.max():.3f})", flush=True)

    t = time.time()
    w2c, K, _, links, n_merged = merge(clusters, n)
    t_merge = time.time() - t
    ok = ~np.isnan(w2c[:, 0, 0])
    pm = pose_metrics(w2c[ok], ref_w2c[ok])
    sh = [l["shared"] for l in links]
    print(f"  merge: {n_merged}/{len(clusters)} clusters, {ok.sum()}/{n} images, shared per link "
          f"{min(sh) if sh else 0}-{max(sh) if sh else 0} (median {np.median(sh) if sh else 0:.0f}), "
          f"{t_merge:.1f} s", flush=True)
    print(f"  merged: ATE median {pm['ATE_median']:.2f} m (rmse {pm['ATE_rmse']:.2f}), rel rot median "
          f"{pm['rel_rot_median_deg']:.2f} deg, rel trans dir median {pm['rel_trans_dir_median_deg']:.2f} deg, "
          f"AUC@5 {pm['AUC@5']:.3f}, AUC@15 {pm['AUC@15']:.3f}, ATE < 5 m {pm['cameras_ate_lt_5m']:.1%}", flush=True)

    results = dict(images=n, clusters=len(clusters), sizes=sizes, cluster_s=t_cluster, inference_s=t_inf,
                   merge_s=t_merge, merged=pm, links=links, per_cluster=per_cluster)
    cur = Path(f"runs/e2e_{scene}/graph/metrics.json")
    if cur.exists():
        m = json.loads(cur.read_text())
        before = [s for s in m["stages"] if s["tag"].startswith("bev")][-1]["seed"]
        final = [s for s in m["stages"] if s["tag"] == "final"][0]["seed"]
        for tag, s in (("before GPS coarse BA", before), ("after GPS coarse BA", final)):
            print(f"  current graph builder, {tag}: {s['num_cameras']} images, ATE median {s['ATE_median']:.2f} m, "
                  f"rel rot median {s['rel_rot_median_deg']:.2f} deg, rel trans dir median "
                  f"{s['rel_trans_dir_median_deg']:.2f} deg, AUC@5 {s['AUC@5']:.3f}, AUC@15 {s['AUC@15']:.3f} "
                  f"({m['time']['batches']} batches, {m['time']['inference_s']:.0f} s inference)", flush=True)
        results["current"] = dict(before_gps_ba=before, after_gps_ba=final, time=m["time"])
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    np.fill_diagonal(cobatched, False)
    np.savez(out_dir / "graph.npz", w2c=w2c, K=K, image_hw=np.array(images.shape[-2:]), measured=measured,
             cobatched=cobatched, names=np.array(names),
             clusters=np.array([np.array(c["images"]) for c in clusters], dtype=object))


if __name__ == "__main__":
    main()
