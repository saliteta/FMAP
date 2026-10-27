"""Depth maps and point clouds of VGGT-Omega with and without the late cross-frame attention (blocks 18-22).

Runs the 6 HAV batches of scripts/egr_attention_mass.py twice: intact, and with the inter-frame sub-blocks of
blocks 18-22 replaced by the identity. Picks the 6 mutually overlapping images (every pair shares >= 30 Bentley
tie points) whose cross-view agreement (depth-reprojection covisibility, min of both directions) drops the most,
then writes
  depth_compare.png   per image: RGB | intact depth | removed depth (same colour range) | removed / intact,
                      shown relative to the image's own scale s (a pure scale change is a flat map)
  intact_rgb.ply, removed_rgb.ply           the 6 depth maps lifted with each run's own poses, RGB colour
  intact_by_image.ply, removed_by_image.ply the same points, one colour per image (shows misregistration)
  selection.json      the images, per-image scales and pairwise covisibility of both runs
  cameras_intact.json, cameras_removed.json   each run's predicted cameras of the 6 images (+ the batch's frame 0):
                      world-to-camera matrix, pinhole intrinsics, image size, display colour, image path

    python scripts/egr_depth_scale_viz.py --scene /mnt/d/Dataset/GauUscene/GauUsceneDepth/HAV/colmap_metrics
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egr_attention_viz import INK, INK2, SURFACE  # noqa: E402
from egr_register_affinity import batches  # noqa: E402
from fgsfm.fm.overlap import batch_geometry  # noqa: E402
from fgsfm.geometry.transforms import centers  # noqa: E402
from fgsfm.io.scene import load_reference_scene  # noqa: E402
from vggt_omega.models import VGGTOmega  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

REMOVE = tuple(range(18, 23))
N_SEL = 6
DEPTH = LinearSegmentedColormap.from_list(
    "depth", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])   # near light -> far dark
DEPTH.set_bad("#d9d8d3")
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a3", "#d03b3b"])
DIVERGING.set_bad("#d9d8d3")
IMAGE_COLORS = np.array([[42, 120, 214], [235, 104, 52], [27, 175, 122], [237, 161, 0], [232, 123, 164],
                         [98, 80, 214]], dtype=np.uint8)            # categorical slots 1-5 + violet


def predict(model, images):
    with torch.inference_mode():
        pred = model(images.to("cuda")[None])
    hw = images.shape[-2:]
    ext, intr = encoding_to_camera(pred["pose_enc"], hw)
    n = images.shape[0]
    return dict(ext=ext[0].float(), intr=intr[0].float(),
                dep=pred["depth"][0].float().reshape(n, *hw), conf=pred["depth_conf"][0].float().reshape(n, *hw))


def pair_covis(p):
    geo = batch_geometry(p["ext"], p["intr"], p["dep"], p["conf"])
    return torch.minimum(geo.covis, geo.covis.T).cpu().numpy()


def best_clique(G, drop, k=N_SEL):
    """k mutually connected images maximising the summed covisibility drop (greedy from every seed edge)."""
    best, best_val = None, -np.inf
    n = len(G)
    for a, b in zip(*np.nonzero(np.triu(G, 1))):
        sel = [a, b]
        while len(sel) < k:
            cand = [c for c in range(n) if c not in sel and G[c, sel].all()]
            if not cand:
                break
            sel.append(max(cand, key=lambda c: drop[c, sel].sum()))
        if len(sel) == k:
            val = sum(drop[i, j] for i, j in itertools.combinations(sel, 2))
            if val > best_val:
                best, best_val = sel, val
    return best, best_val


def best_diverse(G, drop, D, A, k=N_SEL, min_dist=25.0, cross_heading=45.0, min_links=2):
    """k images at least min_dist apart (reference camera centres), seeded by an overlapping pair whose headings
    differ by >= cross_heading, each added image overlapping >= min_links already chosen images; maximises the
    summed covisibility drop over the overlapping pairs of the set."""
    best, best_val = None, -np.inf
    n = len(G)
    for a, b in zip(*np.nonzero(np.triu(G & (A >= cross_heading) & (D >= min_dist), 1))):
        sel = [a, b]
        while len(sel) < k:
            need = min(min_links, len(sel))
            cand = [c for c in range(n) if c not in sel and (D[c, sel] >= min_dist).all() and G[c, sel].sum() >= need]
            if not cand:
                break
            sel.append(max(cand, key=lambda c: (drop[c, sel] * G[c, sel]).sum()))
        if len(sel) == k:
            val = sum(drop[i, j] for i, j in itertools.combinations(sel, 2) if G[i, j])
            if val > best_val:
                best, best_val = sel, val
    return best, best_val


def lift(p, i, images, stride=2, conf_q=0.3):
    dep, conf = p["dep"][i], p["conf"][i]
    H, W = dep.shape
    ys, xs = torch.meshgrid(torch.arange(0, H, stride, device=dep.device), torch.arange(0, W, stride, device=dep.device),
                            indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    d, c = dep[ys, xs], conf[ys, xs]
    keep = (c >= torch.quantile(conf.flatten(), conf_q)) & (d > 0)
    ys, xs, d = ys[keep], xs[keep], d[keep]
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(d)], -1)
    Xc = (pix @ torch.linalg.inv(p["intr"][i]).T) * d[:, None]
    R, t = p["ext"][i][:, :3], p["ext"][i][:, 3]
    Xw = (Xc - t) @ R                                                    # R^T (Xc - t)
    rgb = (images[i][:, ys.cpu(), xs.cpu()].permute(1, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Xw.cpu().numpy(), rgb


def write_ply(path, xyz, rgb):
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--min-shared", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("runs/egr_depth_scale_viz"))
    ap.add_argument("--diverse", action="store_true",
                    help="cameras >= --min-dist apart, one cross-heading pair, connected overlap (not a clique)")
    ap.add_argument("--min-dist", type=float, default=25.0)
    ap.add_argument("--cross-heading", type=float, default=45.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ref = load_reference_scene(args.scene)
    names = [c.name for c in ref.graph.cameras]
    W_ref = np.stack([c.pose_w2c for c in ref.graph.cameras])
    C = centers(W_ref)
    from huggingface_hub import hf_hub_download
    model = VGGTOmega().to("cuda").eval()
    model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
                                     map_location="cpu"))
    agg = model.aggregator
    removed = set()
    orig_run_block = agg._run_block
    agg._run_block = lambda blk, tok, rope: tok if id(blk) in removed else orig_run_block(blk, tok, rope)

    best = None
    for bi, (kind, idx) in enumerate(batches(C, np.random.default_rng(0), 64)):
        G = ref.shared[np.ix_(idx, idx)] >= args.min_shared
        np.fill_diagonal(G, False)
        images = load_and_preprocess_images([str(args.scene / "images" / names[g]) for g in idx],
                                            image_resolution=512)
        removed.clear()
        p_int = predict(model, images)
        removed.update(id(agg.inter_frame_blocks[b]) for b in REMOVE)
        p_rem = predict(model, images)
        cov_int, cov_rem = pair_covis(p_int), pair_covis(p_rem)
        if args.diverse:
            Dm = np.linalg.norm(C[idx][:, None] - C[idx][None], axis=-1)
            Rr = W_ref[idx][:, :3, :3]
            Am = np.degrees(np.arccos(np.clip((np.einsum("aij,bij->ab", Rr, Rr) - 1) / 2, -1, 1)))
            sel, val = best_diverse(G, cov_int - cov_rem, Dm, Am, min_dist=args.min_dist,
                                    cross_heading=args.cross_heading)
        else:
            sel, val = best_clique(G, cov_int - cov_rem)
        print(f"batch {bi} {kind:8s}: best {N_SEL}-image clique drop {val:.3f}", flush=True)
        if sel is not None and (best is None or val > best["val"]):
            best = dict(val=val, batch=bi, kind=kind, idx=idx, sel=sel, images=images, p_int=p_int, p_rem=p_rem,
                        cov_int=cov_int, cov_rem=cov_rem)
        else:
            del p_int, p_rem
        torch.cuda.empty_cache()

    b, sel, images = best, best["sel"], best["images"]
    print(f"selected batch {b['batch']} ({b['kind']}), images {[names[b['idx'][i]] for i in sel]}")

    # per-image scale and depth comparison figure
    plt.rcParams.update({"font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2, "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(N_SEL, 4, figsize=(14, 2.55 * N_SEL + 0.6))
    info = []
    for r, i in enumerate(sel):
        di, dr = b["p_int"]["dep"][i].cpu().numpy(), b["p_rem"]["dep"][i].cpu().numpy()
        conf = b["p_int"]["conf"][i].cpu().numpy()
        m = (conf >= np.quantile(conf, 0.3)) & (di > 0) & (dr > 0)
        s = float(np.median(dr[m] / di[m]))
        rel = dr / di / s
        res = np.abs(np.log(rel[m]))
        r2 = 1 - np.var(np.log(dr[m] / di[m])) / np.var(np.log(di[m]))
        info.append(dict(image=names[b["idx"][i]], scale=s, residual_median=float(np.exp(np.median(res)) - 1),
                         residual_p90=float(np.exp(np.percentile(res, 90)) - 1), r2=float(r2)))
        lo, hi = np.percentile(di[m], [2, 98])
        rgb = images[i].permute(1, 2, 0).numpy()
        panels = [(rgb, None, None, None, f"{names[b['idx'][i]][-30:]}"),
                  (np.where(m, di, np.nan), DEPTH, lo, hi, "intact depth"),
                  (np.where(m, dr, np.nan), DEPTH, lo, hi, f"blocks 18-22 cross-frame removed\n(same colour range)"),
                  (np.where(m, rel, np.nan), DIVERGING, 0.95, 1.05,
                   f"removed / intact ÷ s, s = {s:.3f}\nresidual {info[-1]['residual_median']:.2%} (p90 "
                   f"{info[-1]['residual_p90']:.2%}), R² {r2:.4f}")]
        for c, (img, cmap, vmin, vmax, title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(title, fontsize=8, loc="left")
            if c == 3 and r == 0:
                cb = fig.colorbar(im, ax=axes[:, 3], fraction=0.05, pad=0.02)
                cb.set_label("ratio relative to the image's own scale (1 = pure scale change)")
                cb.outline.set_visible(False)
    fig.suptitle(f"Removing the cross-frame attention of blocks 18-22 rescales each depth map but keeps its shape "
                 f"({b['kind']} batch, the 6 images with the largest loss of cross-view agreement)\n"
                 "Depth: light = near, dark = far; each row uses its intact map's 2-98% range for both depth panels. "
                 "Grey = low-confidence pixels (excluded).", x=0.01, ha="left", fontsize=10.5)
    fig.savefig(args.out / "depth_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # point clouds
    for tag, p in (("intact", b["p_int"]), ("removed", b["p_rem"])):
        xyz, rgb, col = [], [], []
        for k, i in enumerate(sel):
            x, c = lift(p, i, images)
            xyz.append(x)
            rgb.append(c)
            col.append(np.repeat(IMAGE_COLORS[k][None], len(x), 0))
        xyz, rgb, col = np.concatenate(xyz), np.concatenate(rgb), np.concatenate(col)
        write_ply(args.out / f"{tag}_rgb.ply", xyz, rgb)
        write_ply(args.out / f"{tag}_by_image.ply", xyz, col)
        if tag == "removed":
            E0 = np.eye(4)
            E0[:3] = p["ext"][0].cpu().numpy()
            xyz0 = (xyz @ E0[:3, :3].T + E0[:3, 3]).astype(np.float32)
            write_ply(args.out / "removed_f0frame_rgb.ply", xyz0, rgb)
            write_ply(args.out / "removed_f0frame_by_image.ply", xyz0, col)
        print(f"{tag}: {len(xyz)} points")
    H, W = images.shape[-2:]
    for tag, p in (("intact", b["p_int"]), ("removed", b["p_rem"])):
        cams = []
        for k, i in enumerate([0] + list(sel)):
            w2c = np.eye(4)
            w2c[:3] = p["ext"][i].cpu().numpy()
            K = p["intr"][i].cpu().numpy()
            cams.append(dict(name=names[b["idx"][i]], frame0=bool(i == 0), w2c=w2c.tolist(),
                             fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                             width=int(W), height=int(H), image=str(args.scene / "images" / names[b["idx"][i]]),
                             color=[0, 0, 0] if i == 0 else IMAGE_COLORS[k - 1].tolist()))
        (args.out / f"cameras_{tag}.json").write_text(json.dumps(cams, indent=1))
        if tag == "removed":                      # re-expressed in the run's own frame-0 camera coordinates
            T0 = np.array(cams[0]["w2c"])
            cams_f0 = [dict(c, w2c=(np.array(c["w2c"]) @ np.linalg.inv(T0)).tolist()) for c in cams]
            (args.out / "cameras_removed_f0frame.json").write_text(json.dumps(cams_f0, indent=1))
    sub = np.ix_(sel, sel)
    iu = np.triu_indices(N_SEL, 1)
    (args.out / "selection.json").write_text(json.dumps(dict(
        batch=b["batch"], kind=b["kind"], images=[names[b["idx"][i]] for i in sel],
        image_colors=IMAGE_COLORS.tolist(), per_image=info,
        covis_intact=b["cov_int"][sub].tolist(), covis_removed=b["cov_rem"][sub].tolist(),
        mean_covis_intact=float(b["cov_int"][sub][iu].mean()), mean_covis_removed=float(b["cov_rem"][sub][iu].mean())),
        indent=1))
    print(f"mean pairwise covisibility of the 6 images: intact {b['cov_int'][sub][iu].mean():.3f}, "
          f"removed {b['cov_rem'][sub][iu].mean():.3f}")
    for d in info:
        print(f"  {d['image']}: scale {d['scale']:.3f}, residual {d['residual_median']:.2%} (p90 "
              f"{d['residual_p90']:.2%}), R2 {d['r2']:.4f}")


if __name__ == "__main__":
    main()
