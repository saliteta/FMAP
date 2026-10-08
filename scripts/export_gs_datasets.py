"""Export COLMAP-reference and our reconstruction as two gsplat-ready datasets with the SAME images.

Image set = intersection of cameras registered well by both methods:
  - COLMAP reference: every image in its model;
  - ours: cameras in the BA result with >= --min-obs BA track observations
    (a criterion that does not look at the reference).
Images are downscaled once to --width (default 1600, the 3DGS "1.6k" default)
and shared by both datasets; models keep full-resolution intrinsics (gsplat
rescales K to the actual image size). Test split is gsplat's: sorted names,
index % 8 == 0.

    <out>/images/                      shared 1600 px images (JPEG q98, 4:4:4)
    <out>/colmap/{images -> ../images, sparse/0/{cameras.bin, images.bin, points3D.ply}}
    <out>/ours/  {images -> ../images, sparse/0/{...}}
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fgsfm.io.colmap_text import ColmapModel


def write_cameras_bin(path: Path, cams: list[tuple[int, int, int, np.ndarray]]) -> None:
    """cams: (camera_id, width, height, [fx, fy, cx, cy]) PINHOLE (model id 1)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, w, h, p in cams:
            f.write(struct.pack("<iiQQ", cid, 1, w, h))
            f.write(struct.pack("<4d", *p))


def write_images_bin(path: Path, images: list[tuple[int, np.ndarray, int, str]]) -> None:
    """images: (image_id, w2c 4x4, camera_id, name); no 2D points."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, w2c, cid, name in images:
            x, y, z, w = Rotation.from_matrix(w2c[:3, :3]).as_quat()
            f.write(struct.pack("<i4d3di", iid, w, x, y, z, *w2c[:3, 3], cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T.astype(np.float32)
    v["red"], v["green"], v["blue"] = rgb.T
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def resize_one(src: Path, dst: Path, width: int) -> None:
    if dst.exists():
        return
    im = Image.open(src)
    im.draft("RGB", (width, width * im.height // im.width))
    im = im.convert("RGB")
    h = round(im.height * width / im.width)
    im.resize((width, h), Image.LANCZOS).save(dst, quality=98, subsampling=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--ba", type=Path, required=True, help="ba_result.npz from run_global_ba.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--min-obs", type=int, default=100)
    ap.add_argument("--restrict-to", type=Path, default=None,
                    help="split.json of an earlier export: use exactly its image list (fails if a name is missing)")
    ap.add_argument("--only", choices=("both", "colmap", "ours"), default="both")
    ap.add_argument("--min-point-track-len", type=int, default=2,
                    help="ours: initialize 3DGS only from BA tracks seen in >= this many images")
    ap.add_argument("--max-point-reproj", type=float, default=float("inf"),
                    help="ours: drop points whose mean reprojection error (full-res px, final BA poses) exceeds this")
    ap.add_argument("--min-point-angle", type=float, default=0.0,
                    help="ours: drop points whose max triangulation angle (deg) is below this")
    args = ap.parse_args()

    ref = ColmapModel.load(args.scene / "sparse" / "0")
    ref_by_name = {im.name: im for im in ref.images.values()}
    d = dict(np.load(args.ba))          # materialize once (npz access decompresses on every read)
    ours_names = [str(n) for n in d["names"]]
    well = d["obs_per_image"] >= args.min_obs
    dropped = [n for n, ok in zip(ours_names, well) if not ok]
    names = sorted(set(n for n, ok in zip(ours_names, well) if ok) & set(ref_by_name))
    if args.restrict_to is not None:
        want = json.loads(args.restrict_to.read_text())["names"]
        missing = sorted(set(want) - set(names))
        assert not missing, f"{len(missing)} images of {args.restrict_to} are not well registered here: {missing}"
        names = sorted(want)
    n_test = sum(1 for i in range(len(names)) if i % 8 == 0)
    print(f"images: COLMAP {len(ref_by_name)}, ours {len(ours_names)} (dropped {len(dropped)} with < {args.min_obs} obs: "
          f"{dropped}) -> intersection {len(names)}  (train {len(names) - n_test} / test {n_test})")

    # shared downscaled images
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda n: resize_one(args.scene / "images" / n, img_dir / n, args.width), names))

    def make(dataset: str, cams, images, xyz, rgb):
        root = args.out / dataset
        sp = root / "sparse" / "0"
        sp.mkdir(parents=True, exist_ok=True)
        link = root / "images"
        if not link.exists():
            link.symlink_to(Path("..") / "images")
        write_cameras_bin(sp / "cameras.bin", cams)
        write_images_bin(sp / "images.bin", images)
        write_points_ply(sp / "points3D.ply", xyz, rgb)
        print(f"{dataset}: {len(images)} images, {len(xyz)} points -> {root}")

    # --- COLMAP reference: its own intrinsics, poses and tie points
    cams, images = {}, []
    for k, n in enumerate(names):
        im = ref_by_name[n]
        c = ref.cameras[im.camera_id]
        cams[im.camera_id] = (im.camera_id, c.width, c.height, c.params[:4])
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = im.R, im.tvec
        images.append((k + 1, w2c, im.camera_id, n))
    if args.only in ("both", "colmap"):
        make("colmap", list(cams.values()), images, ref.points_xyz, ref.points_rgb)

    # --- ours: BA poses + shared BA intrinsics + BA tracks colored from their observations
    pos = {n: i for i, n in enumerate(ours_names)}
    W, H = (int(x) for x in d["image_wh"])
    K = d["K"]
    images = [(k + 1, d["w2c"][pos[n]], 1, n) for k, n in enumerate(names)]
    xyz = d["points"].astype(np.float64)
    rgb = np.zeros((len(xyz), 3), np.uint8)
    track_len = np.bincount(d["obs_track"], minlength=len(xyz))
    # per-point quality from the final BA: mean reprojection error and max triangulation angle
    t_, im_, uv_ = d["obs_track"], d["obs_image"], d["obs_xy"].astype(np.float64)
    Xc = np.einsum("nij,nj->ni", d["w2c"][im_, :3, :3], xyz[t_]) + d["w2c"][im_, :3, 3]
    pr = Xc @ K.T
    err = np.linalg.norm(pr[:, :2] / pr[:, 2:] - uv_, axis=1)
    mean_reproj = np.bincount(t_, err, minlength=len(xyz)) / np.maximum(track_len, 1)
    Cc = -np.einsum("nji,nj->ni", d["w2c"][:, :3, :3], d["w2c"][:, :3, 3])
    ray = xyz[t_] - Cc[im_]
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    r0 = ray[np.unique(t_, return_index=True)[1]][t_]
    max_angle = np.zeros(len(xyz))
    np.maximum.at(max_angle, t_, np.degrees(np.arccos(np.clip((ray * r0).sum(1), -1, 1))))
    first = np.unique(d["obs_track"], return_index=True)[1]          # one observation per track
    s = args.width / W
    first_img = d["obs_image"][first]
    for li in np.unique(first_img):
        obs = first[first_img == li]
        src = img_dir / ours_names[li]
        im = Image.open(src) if src.exists() else \
            Image.open(args.scene / "images" / ours_names[li]).resize((args.width, round(H * s)))
        arr = np.asarray(im.convert("RGB"))
        xy = np.clip(np.round(d["obs_xy"][obs] * s).astype(int), 0, [arr.shape[1] - 1, arr.shape[0] - 1])
        rgb[d["obs_track"][obs]] = arr[xy[:, 1], xy[:, 0]]
    if args.only in ("both", "ours"):
        keep = (track_len >= args.min_point_track_len) & (mean_reproj <= args.max_point_reproj) & \
            (max_angle >= args.min_point_angle)
        print(f"ours: {keep.sum()} / {len(xyz)} points (track >= {args.min_point_track_len} views, "
              f"mean reproj <= {args.max_point_reproj} px, tri angle >= {args.min_point_angle} deg)")
        make("ours", [(1, W, H, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))], images, xyz[keep], rgb[keep])

    (args.out / "split.json").write_text(json.dumps(dict(
        names=names, test=[n for i, n in enumerate(names) if i % 8 == 0], dropped_ours=dropped,
        min_obs=args.min_obs, width=args.width), indent=1))


if __name__ == "__main__":
    main()
