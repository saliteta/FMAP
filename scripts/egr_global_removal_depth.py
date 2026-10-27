"""What happens to VGGT-Omega's depth when the cross-frame (global) attention of late blocks is removed?

The inter-frame sub-block of the chosen blocks is replaced by the identity (the per-frame sub-block still runs);
no recovery map. Same 6 HAV batches as scripts/egr_attention_mass.py, frame 0 = the batch's first image. Compared
with the intact model on the same batch, per image (confident pixels = the intact model's top 70% confidence):
    depth scale   s_i = median(depth_removed / depth_intact)
    scale spread  standard deviation of log s_i over the 64 images of a batch (0 = every image rescaled alike)
    depth shape   median |depth_removed / s_i - depth_intact| / depth_intact (after per-image rescaling)
    camera scale  median over images of |C_i - C_0| removed / intact (predicted camera centres)
    depth/camera  s_i / camera scale: 1 = depth and poses rescaled together (still mutually consistent)
    multi-view    mean depth-reprojection covisibility over reference edges (>= 30 shared tie points),
                  i.e. whether the depth maps of overlapping images still agree under the predicted poses
Within-image relative depth ("does it differ from the intact depth only by a scale?"), per image, in log depth:
    residual      |log d_removed - log d_intact - log s_i| after the per-image scale s_i: median and 90th
                  percentile over the image's pixels (reported as % depth error)
    R2            1 - var(residual) / var(log d_intact): share of the image's own relative-depth variation that
                  survives up to one scale (1 = identical up to scale)
    affine gain   median residual of the scale-only fit / that of a scale + offset fit in depth (1 = an offset
                  adds nothing, i.e. the change is a pure scale)
Configurations "g..." remove the inter-frame sub-block only; "w..." remove whole blocks (per-frame sub-block too).

    python scripts/egr_global_removal_depth.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
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
from fgsfm.fm.overlap import batch_geometry  # noqa: E402
from fgsfm.geometry.transforms import centers  # noqa: E402
from fgsfm.io.colmap_text import ColmapModel  # noqa: E402
from fgsfm.io.scene import load_reference_scene, reference_covisibility  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

CONFIGS = [("intact", ()), ("g15", (15,))] + [(f"g{b}", (b,)) for b in range(17, 24)] + \
          [("g19-21", (19, 20, 21)), ("g17-22", tuple(range(17, 23))), ("g17-23", tuple(range(17, 24))),
           ("g16", (16,)), ("g18-22", tuple(range(18, 23))),      # controls: no head reads blocks 16, 18-22
           ("w20", (20,)), ("w18-22", tuple(range(18, 23))), ("w17-23", tuple(range(17, 24)))]   # whole blocks


def load_scene_reference(root):
    """(image names, reference camera centres, shared tie-point counts). Scenes whose points3D.ply is a dense cloud
    (not the AT tie points) take the covisibility straight from AT-export.xml."""
    try:
        ref = load_reference_scene(root)
        return ([c.name for c in ref.graph.cameras], centers(np.stack([c.pose_w2c for c in ref.graph.cameras])),
                ref.shared)
    except ValueError:
        ims = ColmapModel.load(root / "sparse" / "0").sorted_images()
        names, shared = reference_covisibility(root)
        assert names == [im.name for im in ims]
        return names, centers(np.stack([np.r_[np.c_[im.R, im.tvec], [[0, 0, 0, 1]]] for im in ims])), shared


def frame0_drift(ext, cam):
    """rotation of frame 0's predicted world-to-camera from the identity (deg), and its camera centre's distance
    from the origin relative to the median distance of the other cameras from it."""
    R0 = ext[0, :, :3]
    ang = float(torch.rad2deg(torch.arccos(((torch.trace(R0) - 1) / 2).clamp(-1, 1))))
    off = float(cam[0].norm() / (cam[1:] - cam[0]).norm(dim=1).median().clamp(min=1e-9))
    return ang, off


def run(model, images):
    with torch.inference_mode():
        pred = model(images.to("cuda")[None])
    hw = images.shape[-2:]
    ext, intr = encoding_to_camera(pred["pose_enc"], hw)
    n = images.shape[0]
    dep = pred["depth"][0].float().reshape(n, *hw)
    conf = pred["depth_conf"][0].float().reshape(n, *hw)
    return ext[0].float(), intr[0].float(), dep, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--min-shared", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("runs/egr_global_removal_depth"))
    ap.add_argument("--only", default=None, help="comma-separated configuration names (intact always runs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names, C, shared = load_scene_reference(args.scene)
    print(f"scene {args.scene}: {len(names)} images", flush=True)
    from huggingface_hub import hf_hub_download
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
                                     map_location="cpu"))
    agg = model.aggregator
    removed = set()
    orig_run_block = agg._run_block

    def run_block(block, tokens, rope_sincos):
        return tokens if id(block) in removed else orig_run_block(block, tokens, rope_sincos)
    agg._run_block = run_block

    batch_list = []
    for kind, idx in batches(C, np.random.default_rng(0), 64):
        G = shared[np.ix_(idx, idx)] >= args.min_shared
        np.fill_diagonal(G, False)
        images = load_and_preprocess_images([str(args.scene / "images" / names[g]) for g in idx],
                                            image_resolution=512)
        batch_list.append((kind, images, torch.tensor(G, device="cuda")))

    intact, results, t0 = [], {}, time.time()
    configs = [c for c in CONFIGS if args.only is None or c[0] == "intact" or c[0] in args.only.split(",")]
    for name, blocks in configs:
        removed.clear()
        removed.update(id(agg.inter_frame_blocks[b]) for b in blocks)
        if name.startswith("w"):
            removed.update(id(agg.frame_blocks[b]) for b in blocks)
        per_batch = []
        for bi, (kind, images, G) in enumerate(batch_list):
            ext, intr, dep, conf = run(model, images)
            geo = batch_geometry(ext, intr, dep, conf)
            covis = float(torch.minimum(geo.covis, geo.covis.T)[G].mean())
            R, t = ext[:, :, :3], ext[:, :, 3]
            cam = -(R.transpose(1, 2) @ t[..., None])[..., 0]                      # camera centres
            f0_rot, f0_off = frame0_drift(ext, cam)
            if name == "intact":
                thr = torch.quantile(conf.flatten(1), 0.3, dim=1)[:, None, None]
                intact.append(dict(dep=dep, mask=(conf >= thr) & (dep > 0), cam=cam, covis=covis))
                per_batch.append(dict(covis=covis, f0_rot=f0_rot, f0_off=f0_off))
                continue
            ref_b = intact[bi]
            m = ref_b["mask"] & (dep > 0)
            s, shape, res_med, res_p90, r2, aff = [], [], [], [], [], []
            for i in range(dep.shape[0]):
                dn, di = dep[i][m[i]].double(), ref_b["dep"][i][m[i]].double()
                si = (dn / di).median()
                s.append(float(si))
                shape.append(float(((dn / si - di).abs() / di).median()))
                res = (dn.log() - di.log() - si.log()).abs()
                res_med.append(float(res.median().exp() - 1))
                res_p90.append(float(torch.quantile(res.float(), 0.9).exp() - 1))
                r2.append(float(1 - (dn.log() - di.log() - si.log()).var() / di.log().var()))
                A = torch.stack([di, torch.ones_like(di)], 1)                    # dn ~ a * di + b
                coef = torch.linalg.lstsq(A, dn[:, None]).solution[:, 0]
                aff_res = ((dn - A @ coef).abs() / (coef[0] * di)).median()
                aff.append(float(((dn / si - di).abs() / di).median() / max(float(aff_res), 1e-12)))
            s = np.array(s)
            d_new = (cam[1:] - cam[0]).norm(dim=1)
            d_old = (ref_b["cam"][1:] - ref_b["cam"][0]).norm(dim=1)
            cam_scale = float((d_new / d_old.clamp(min=1e-9)).median())
            per_batch.append(dict(kind=kind, f0_rot=f0_rot, f0_off=f0_off, depth_scale_median=float(np.median(s)),
                                  depth_scale_logstd=float(np.log(s).std()),
                                  depth_scale_range=[float(s.min()), float(s.max())],
                                  shape_err=float(np.median(shape)), cam_scale=cam_scale,
                                  res_med=float(np.median(res_med)), res_p90=float(np.median(res_p90)),
                                  res_med_worst10=float(np.percentile(res_med, 90)),
                                  r2=float(np.median(r2)), r2_worst10=float(np.percentile(r2, 10)),
                                  affine_gain=float(np.median(aff)),
                                  depth_over_cam=float(np.median(s) / cam_scale), covis=covis,
                                  covis_intact=ref_b["covis"]))
        results[name] = per_batch
        if name == "intact":
            print(f"intact ({time.time() - t0:.0f} s): multi-view covisibility on reference edges "
                  f"{np.mean([p['covis'] for p in per_batch]):.3f} | frame 0 vs identity: rotation "
                  f"{np.mean([p['f0_rot'] for p in per_batch]):.2f} deg, offset {np.mean([p['f0_off'] for p in per_batch]):.1%}",
                  flush=True)
            continue
        f = lambda k: np.mean([p[k] for p in per_batch])                          # noqa: E731
        print(f"remove {name:7s} ({time.time() - t0:4.0f} s): depth scale x{np.exp(np.mean([np.log(p['depth_scale_median']) for p in per_batch])):.3f} "
              f"(per-batch {min(p['depth_scale_median'] for p in per_batch):.3f}-{max(p['depth_scale_median'] for p in per_batch):.3f}), "
              f"spread over images (log std) {f('depth_scale_logstd'):.3f}, shape err {f('shape_err'):.3f} | "
              f"camera scale x{np.exp(np.mean([np.log(p['cam_scale']) for p in per_batch])):.3f} | "
              f"depth/camera {np.exp(np.mean([np.log(p['depth_over_cam']) for p in per_batch])):.3f} | "
              f"covis {f('covis'):.3f} (intact {f('covis_intact'):.3f})", flush=True)
        print(f"        within-image, up to one scale: residual median {f('res_med'):.2%} p90 {f('res_p90'):.2%} "
              f"(worst 10% of images: median {f('res_med_worst10'):.2%}) | R2 {f('r2'):.4f} (worst 10%: "
              f"{f('r2_worst10'):.4f}) | scale+offset fit improves residual x{f('affine_gain'):.2f} | frame 0 vs "
              f"identity: rotation {f('f0_rot'):.2f} deg (max {max(p['f0_rot'] for p in per_batch):.2f}), offset "
              f"{f('f0_off'):.1%}", flush=True)
        (args.out / ("results.json" if args.only is None else f"results_{args.only.replace(',', '_')}.json")).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
