"""Stage A + B in one process: view graph from half of VGGT-Omega, then poses + depth from full VGGT-Omega per cluster.

The model and the images are loaded once. Stage A runs the same model's aggregator up to block 13 (block 13 head 15
patch-query attention share, top-k per image, spatial cells; scripts/egr_graph_stage_a.py); stage B covers the graph
with clusters, runs the full model once per cluster with a hub frame 0 and merges the clusters by Sim(3)
(scripts/egr_graph_stage_b.py; default: each cluster straight onto EXIF GPS, merge_gps). RootSIFT extraction for the BA runs meanwhile in a CPU thread, into the SIFT cache
that scripts/run_global_ba.py reads. Every step is timed; the comparison with the reference runs after the clock stops.

    python scripts/run_stage_ab.py --scene .../SZTU/colmap_metrics --out runs/stage_ab/SZTU --sift-cache runs/stage_ab/SZTU/sift
    python scripts/run_global_ba.py --scene ... --graph-npz runs/stage_ab/SZTU/graph.npz --out ... --use-gps \
        --tracks-per-image 800 --sift-cache runs/stage_ab/SZTU/sift
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

T0 = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_attention_mass import PRE  # noqa: E402
from egr_graph_stage_a import HeadProbe, kd_partition, load_images, topk_graph  # noqa: E402
from egr_graph_stage_b import DEPTH_STRIDE, OVERLAP_STRIDE, cover, merge, merge_gps, reference_w2c  # noqa: E402
from fgsfm.geometry.densify import STRIDE as DENSE_STRIDE  # noqa: E402
from fgsfm.fm.overlap import batch_geometry, symmetric_overlap  # noqa: E402
from fgsfm.geometry.transforms import to4x4  # noqa: E402
from fgsfm.io.colmap_text import ColmapModel  # noqa: E402
from fgsfm.graph.islands import repair  # noqa: E402
from fgsfm.io.gps import flights_for, gimbal_w2c_for, gps_enu_for  # noqa: E402
from fgsfm.match.sift import extract_features  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", choices=("gps", "reference"), default="gps",
                    help="camera positions for the stage-A spatial cells: EXIF GPS, or the reference AT centres")
    ap.add_argument("--no-island-repair", action="store_true", help="skip the island search (fgsfm/graph/islands.py)")
    ap.add_argument("--island-overlap", type=float, default=0.05,
                    help="measured overlap that counts as a pair-graph edge (run_global_ba.py --topk-min-overlap)")
    ap.add_argument("--no-flight-offsets", action="store_true",
                    help="trust every flight's GPS as is (default: per-flight offsets when gps.npz has 'flight')")
    ap.add_argument("--no-gimbal", action="store_true",
                    help="ignore DJI gimbal angles in gps.npz (default: rotation prior for the GPS merge when present)")
    ap.add_argument("--merge", choices=("gps", "chain"), default="gps",
                    help="clusters into one frame: each straight onto EXIF GPS, or chained by shared cameras")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--r", type=int, default=2, help="stage-A cells per image")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--block", type=int, default=13)
    ap.add_argument("--head", type=int, default=15)
    ap.add_argument("--sift-cache", type=Path, default=None, help="extract RootSIFT here in a CPU thread meanwhile")
    ap.add_argument("--match-width", type=int, default=2736)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--cache", type=Path, default=Path("runs/cache"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    T = {"imports_s": time.time() - T0}

    def tick(name, t):
        T[name] = time.time() - t
        print(f"  {name:<22}{T[name]:7.1f} s", flush=True)
        return time.time()

    t = time.time()
    names = [im.name for im in ColmapModel.load(args.scene / "sparse" / "0").sorted_images()]   # image list only
    n = len(names)
    gps, gps_ok = gps_enu_for(names, args.scene / "AT-export.xml", cache=args.cache / f"{args.scene.parent.name}_gps.npz")
    if args.merge == "gps":
        assert gps_ok.all(), "--merge gps needs GPS for every image"
    if args.positions == "gps":
        assert gps_ok.all(), "every image needs GPS for the spatial cells"
        P = gps
    else:
        from fgsfm.geometry.transforms import centers
        P = centers(reference_w2c(args.scene)[1])
    t = tick("image list + positions", t)

    sift = {}
    if args.sift_cache is not None:
        def run_sift():
            s = time.time()
            extract_features([args.scene / "images" / nm for nm in names], args.sift_cache, args.match_width,
                             args.max_features)
            sift["s"] = time.time() - s
        th = threading.Thread(target=run_sift, daemon=True)
        th.start()

    images = load_images(args.scene, names, args.cache)
    t = tick("images", t)
    from huggingface_hub import hf_hub_download
    model = VGGTOmega()
    model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
                                     map_location="cpu"))
    model = model.to("cuda").eval()
    agg = model.aggregator
    state = {"T": PRE + (images.shape[-2] // 16) * (images.shape[-1] // 16)}
    probe = HeadProbe(agg.inter_frame_blocks[args.block].attn, args.head, lambda: state["T"])
    torch.cuda.synchronize()
    t = tick("model load", t)

    # ---- stage A: blocks 0..block only
    rng = np.random.default_rng(0)
    plan = []
    for _ in range(args.r):
        plan += [np.sort(c) for c in kd_partition(P[:, :2] - P[:, :2].mean(0), args.batch, rng)]
    cands = [dict() for _ in range(n)]
    agg.depth, full_depth = args.block + 1, agg.depth
    for idx in plan:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            agg(images[idx].float().cuda()[None])
        M = probe.mass.astype(np.float64)
        np.fill_diagonal(M, 0)
        M = M / M.sum(1, keepdims=True) * (len(idx) - 1)
        for a in range(len(idx)):
            for b in np.argsort(-M[a])[:args.k]:
                gi, gj = int(idx[a]), int(idx[b])
                cands[gi][gj] = max(cands[gi].get(gj, 0.0), float(M[a, b]))
    agg.depth = full_depth
    _, A = topk_graph(cands, n, args.k)
    torch.cuda.synchronize()
    t = tick("stage A", t)

    # ---- stage B: clusters, full model per cluster, merge
    U = A | A.T
    adj = [set(np.flatnonzero(U[i]).tolist()) for i in range(n)]
    clusters = cover(adj, n, args.batch)
    t = tick("clustering", t)
    measured, cobatched = np.zeros((n, n)), np.zeros((n, n), bool)

    def run_cluster(C):
        x = images[C["images"]].float().cuda()
        with torch.inference_mode():
            pred = model(x[None])
        ext, intr = encoding_to_camera(pred["pose_enc"], x.shape[-2:])
        C["w2c"] = to4x4(ext[0].double().cpu().numpy())
        C["K"] = intr[0].double().cpu().numpy()
        C["depth"] = pred["depth"][0, :, ::DEPTH_STRIDE, ::DEPTH_STRIDE, 0].float().cpu().numpy()
        conf = pred["depth_conf"][0, :, ::DEPTH_STRIDE, ::DEPTH_STRIDE].float().cpu().numpy()
        C["conf_ok"] = conf >= np.median(conf.reshape(len(conf), -1), 1)[:, None, None]
        # stride-4 depth for the 3DGS densification (fgsfm/geometry/densify.py); valid = top 70% confidence
        d4 = pred["depth"][0, :, ::DENSE_STRIDE, ::DENSE_STRIDE, 0].float().cpu().numpy()
        c4 = pred["depth_conf"][0, :, ::DENSE_STRIDE, ::DENSE_STRIDE].float().cpu().numpy()
        C["depth4"] = d4.astype(np.float16)
        C["valid4"] = (c4 >= np.quantile(c4.reshape(len(c4), -1), 0.3, axis=1)[:, None, None]) & (d4 > 0)
        geo = batch_geometry(ext[0].float(), intr[0].float(), pred["depth"][0, ..., 0].float(),
                             pred["depth_conf"][0].float(), stride=OVERLAP_STRIDE)
        ix = np.ix_(C["images"], C["images"])
        measured[ix] = np.maximum(measured[ix], symmetric_overlap(geo.covis).cpu().numpy())
        cobatched[ix] = True

    for C in clusters:
        run_cluster(C)
    torch.cuda.synchronize()
    t = tick("stage B inference", t)

    # ---- no isolated islands (fgsfm/graph/islands.py): the pair graph of run_global_ba.py is measured overlap
    # >= --island-overlap between co-batched images; search each island with its GPS neighbours, widening to all
    bridge = np.zeros((n, n), bool)
    discarded = np.zeros(n, bool)
    if not args.no_island_repair:
        half = args.batch // 2

        def search(part, cand):
            order = [int(part[0])] + [int(v) for v in part[1:]] + [int(v) for v in cand[:args.batch - len(part)]]
            C = dict(images=order, hops=[-1] * len(part) + [99] * (len(order) - len(part)), repair=True)
            run_cluster(C)                                    # island images prefer this batch for placement
            clusters.append(C)
            cand_used = np.array(order[len(part):])
            new = measured[np.ix_(part, cand_used)] >= args.island_overlap
            bridge[np.ix_(part, cand_used)] |= new
            bridge[np.ix_(cand_used, part)] |= new.T
            return cobatched & (measured >= args.island_overlap)

        _, discarded, island_report = repair(cobatched & (measured >= args.island_overlap), P, search,
                                             step=args.batch - half, max_island_part=half)
        T["islands"] = island_report
        torch.cuda.synchronize()
        t = tick("island repair", t)
    torch.cuda.synchronize()
    t = tick("stage B inference", t)
    if args.merge == "gps":
        gimbal = None if args.no_gimbal else gimbal_w2c_for(names, args.scene / "AT-export.xml")
        flights = None if args.no_flight_offsets else flights_for(names, args.scene / "AT-export.xml")
        w2c, K, src, links, n_merged = merge_gps(clusters, n, gps, gimbal=gimbal, flights=flights)
        if gimbal is not None:
            ng = sum(bool(l.get("gimbal")) for l in links)
            print(f"merge: gimbal rotation prior on, {ng} / {len(links)} clusters placed with the gimbal rotation", flush=True)
    else:
        w2c, K, src, links, n_merged = merge(clusters, n)
    if discarded.any():                                        # islands that never linked: not in the model
        w2c[discarded] = np.nan
        src[discarded] = -1
        print(f"discarded {int(discarded.sum())} images in unlinked islands", flush=True)
    depth4 = np.zeros((n,) + clusters[0]["depth4"].shape[1:], np.float16)
    valid4 = np.zeros(depth4.shape, bool)
    for g, (ci, k) in enumerate(src):
        if ci >= 0:
            depth4[g], valid4[g] = clusters[ci]["depth4"][k], clusters[ci]["valid4"][k]
    np.fill_diagonal(cobatched, False)
    np.savez(args.out / "graph.npz", w2c=w2c, K=K, image_hw=np.array(images.shape[-2:]), measured=measured,
             cobatched=cobatched, bridge=bridge, discarded=discarded, names=np.array(names), depth4=depth4, valid4=valid4,
             clusters=np.array([np.array(c["images"]) for c in clusters], dtype=object),
             cluster_w2c=np.array([c["w2c"] for c in clusters], dtype=object),       # raw VGGT-Omega poses, so the
             cluster_hops=np.array([np.array(c["hops"]) for c in clusters], dtype=object))  # merge can be redone
    t = tick("merge + save", t)
    if args.sift_cache is not None:
        th.join()
        t = tick("wait for SIFT", t)
        T["sift_thread_s"] = sift["s"]
    T["total_s"] = time.time() - T0
    print(f"{args.scene.parent.name}: {n} images, {len(plan)} stage-A cells, {len(clusters)} clusters, "
          f"{n_merged} merged, {int(np.isfinite(w2c[:, 0, 0]).sum())} placed; SIFT thread "
          f"{T.get('sift_thread_s', 0):.0f} s; total {T['total_s']:.0f} s (imports {T['imports_s']:.1f} s)", flush=True)

    # ---- evaluation (not timed)
    from fgsfm.eval.pose_metrics import pose_metrics
    ref_names, ref = reference_w2c(args.scene)
    assert ref_names == names
    okc = np.isfinite(w2c[:, 0, 0])
    pm = pose_metrics(w2c[okc], ref[okc])
    print(f"  merged: ATE median {pm['ATE_median']:.2f} m, rel rot median {pm['rel_rot_median_deg']:.2f} deg, "
          f"AUC@5 {pm['AUC@5']:.3f}", flush=True)
    (args.out / "stage_ab.json").write_text(json.dumps(dict(time=T, images=n, merge=args.merge, cells=len(plan), clusters=len(clusters),
                                                            merged=pm, links=links), indent=1, default=float))


if __name__ == "__main__":
    main()
