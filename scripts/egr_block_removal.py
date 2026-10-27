"""Block removal in VGGT-Omega, as in "Does the VGGT Family Need All Its Layers?" (arXiv 2609.36842), on HAV.

Removed blocks are replaced by the identity inside the aggregator's _run_block, with no recovery map; every other
block still runs, so later blocks can compensate (unlike reading out an earlier block, which drops everything
after it). Removing a whole block (frame-attention + inter-frame sub-block) keeps the cached head inputs
concat(frame output, inter-frame output) a matched pair (both halves = the incoming tokens).
Configurations:
    intact; every single whole block; whole-block intervals of 2-4 blocks ending at block 14..23;
    the inter-frame (cross-view) sub-block alone of blocks 10..23 ("g<b>")
Batches: the 6 HAV batches of scripts/egr_attention_mass.py, each run with frame 0 = its first image and with
frame 0 = its best-connected image. Errors are relative poses against the Bentley reference, per pair:
rotation error and translation-direction error (deg); pose error = the larger of the two.
    all pairs       mean, median, standard deviation, 90th percentile of each; AUC@5 / AUC@10 of the pose error
    by hop from f0  the same statistics for the (frame 0, i) poses, by graph distance from frame 0
Raw per-pair errors of every configuration are saved to <out>/errors/<name>.npz.

    python scripts/egr_block_removal.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_register_affinity import batches  # noqa: E402
from fgsfm.eval.overlap_metrics import pose_auc, relative_pose_errors  # noqa: E402
from fgsfm.geometry.transforms import centers  # noqa: E402
from fgsfm.io.scene import load_reference_scene  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

HOPS = ["1", "2", "3+", "none"]


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), std=float(x.std()),
                var=float(x.var()), p90=float(np.percentile(x, 90)))


def configs(depth):
    """(name, whole blocks removed, inter-frame-only blocks removed)."""
    out = [("intact", (), ())]
    out += [(f"{b}", (b,), ()) for b in range(depth)]
    for length in (2, 3, 4):
        for end in range(14, depth):
            out.append((f"{end - length + 1}-{end}", tuple(range(end - length + 1, end + 1)), ()))
    out += [(f"g{b}", (), (b,)) for b in range(10, depth)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--min-shared", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("runs/egr_block_removal"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ref = load_reference_scene(args.scene)
    names = [c.name for c in ref.graph.cameras]
    w2c_ref_all = np.stack([c.pose_w2c for c in ref.graph.cameras])
    C = centers(w2c_ref_all)
    from huggingface_hub import hf_hub_download
    model = VGGTOmega(enable_depth=False).to("cuda").eval()
    model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
                                     map_location="cpu"), strict=False)
    agg = model.aggregator
    removed = set()
    orig_run_block = agg._run_block

    def run_block(block, tokens, rope_sincos):
        return tokens if id(block) in removed else orig_run_block(block, tokens, rope_sincos)
    agg._run_block = run_block

    runs = []                                         # (images, reference w2c, hop label per frame)
    for kind, idx in batches(C, np.random.default_rng(0), 64):
        G = ref.shared[np.ix_(idx, idx)] >= args.min_shared
        np.fill_diagonal(G, False)
        D = shortest_path(G.astype(float), unweighted=True, directed=False)
        images = load_and_preprocess_images([str(args.scene / "images" / names[g]) for g in idx],
                                            image_resolution=512)
        for f0 in (0, int(np.argmax(G.sum(1)))):
            order = np.r_[f0, [k for k in range(len(idx)) if k != f0]]
            hop = np.array(["f0"] + ["none" if not np.isfinite(D[f0, k]) else "1" if D[f0, k] == 1 else
                                     "2" if D[f0, k] == 2 else "3+" for k in order[1:]])
            runs.append((images[order], w2c_ref_all[idx[order]], hop))
    hw = runs[0][0].shape[-2:]

    results, t0 = {}, time.time()
    (args.out / "errors").mkdir(exist_ok=True)
    for name, whole, inter_only in configs(agg.depth):
        removed.clear()
        for b in whole:
            removed.update({id(agg.frame_blocks[b]), id(agg.inter_frame_blocks[b])})
        for b in inter_only:
            removed.add(id(agg.inter_frame_blocks[b]))
        rot_all, tr_all, hop_all = [], [], []
        for images, w2c_ref, hop in runs:
            with torch.inference_mode():
                pred = model(images.to("cuda")[None])
            ext, _ = encoding_to_camera(pred["pose_enc"], hw, build_intrinsics=False)
            w2c = np.tile(np.eye(4), (len(hop), 1, 1))
            w2c[:, :3, :] = ext[0].float().cpu().numpy()
            rot, tr, i, j = relative_pose_errors(w2c, w2c_ref)
            rot_all.append(rot)
            tr_all.append(tr)
            hop_all.append(np.where(i == 0, hop[j], "pair"))          # hop of image j for (frame 0, j) pairs
            del pred
        rot_all, tr_all, hop_all = np.concatenate(rot_all), np.concatenate(tr_all), np.concatenate(hop_all)
        np.savez_compressed(args.out / "errors" / f"{name}.npz", rot=rot_all.astype(np.float32),
                            tdir=tr_all.astype(np.float32), hop=hop_all)
        pose = np.maximum(rot_all, tr_all)
        results[name] = dict(whole=list(whole), inter_only=list(inter_only),
                             rot=stats(rot_all), tdir=stats(tr_all), pose=stats(pose),
                             auc5=pose_auc(rot_all, tr_all, 5.0), auc10=pose_auc(rot_all, tr_all, 10.0),
                             hop={h: dict(rot=stats(rot_all[hop_all == h]), tdir=stats(tr_all[hop_all == h]))
                                  for h in HOPS if (hop_all == h).any()})
        r = results[name]
        print(f"remove {name:6s} ({time.time() - t0:5.0f} s): rot mean {r['rot']['mean']:6.2f} med {r['rot']['median']:5.2f} "
              f"std {r['rot']['std']:6.2f} | tdir mean {r['tdir']['mean']:6.2f} med {r['tdir']['median']:5.2f} "
              f"std {r['tdir']['std']:6.2f} | AUC@5 {r['auc5']:.3f} AUC@10 {r['auc10']:.3f} | to f0 rot mean by hop " +
              " ".join(f"{h}:{r['hop'][h]['rot']['mean']:.2f}" for h in HOPS if h in r["hop"]), flush=True)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
