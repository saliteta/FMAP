"""Interactive viewer for 3DGS initial point clouds (gsplat datasets written by export_gs_datasets.py).

    python scripts/view_point_clouds.py tracks=runs/gs_ab5/ours dense=runs/gs_dense/ours \
        colmap=runs/gs_HAV_fast/colmap_gps_pp --port 8080

Each dataset is <dir>/sparse/0/{points3D.ply, images.bin}; all must share one world frame. Panel: dataset,
color (RGB / source / height), point size, cameras. "source" colors the points a dataset shares with another
listed dataset (its prefix, e.g. the BA tracks inside the dense cloud) orange and the rest cyan.
"""
from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf


def read_ply(path: Path):
    raw = path.read_bytes()
    k = raw.index(b"end_header\n") + len(b"end_header\n")
    v = np.frombuffer(raw[k:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    return np.c_[v["x"], v["y"], v["z"]].astype(np.float32), np.c_[v["r"], v["g"], v["b"]]


def read_images_bin(path: Path):
    """(name, w2c) of images.bin as written by export_gs_datasets.write_images_bin (zero 2D points)."""
    buf, out = path.read_bytes(), []
    n, = struct.unpack_from("<Q", buf, 0)
    o = 8
    for _ in range(n):
        _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack_from("<i4d3di", buf, o)
        o += struct.calcsize("<i4d3di")
        e = buf.index(b"\x00", o)
        name, o = buf[o:e].decode(), e + 1 + 8          # + uint64 number of 2D points (0)
        w2c = np.eye(4)
        w2c[:3, :3] = vtf.SO3(np.array([qw, qx, qy, qz])).as_matrix()
        w2c[:3, 3] = [tx, ty, tz]
        out.append((name, w2c))
    return out


def height_colors(z: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    lo, hi = np.percentile(z, [2, 98])
    return (cm.turbo(np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="name=path/to/gsplat_dataset")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    data = {}
    for spec in args.datasets:
        name, path = spec.split("=", 1)
        xyz, rgb = read_ply(Path(path) / "sparse/0/points3D.ply")
        _, _, _, w, h, fx, fy, cx, cy = struct.unpack_from("<QiiQQ4d", (Path(path) / "sparse/0/cameras.bin").read_bytes())
        data[name] = dict(xyz=xyz, rgb=rgb, cams=read_images_bin(Path(path) / "sparse/0/images.bin"),
                          fov=float(2 * np.arctan2(h / 2, fy)), aspect=w / h)
        print(f"{name}: {len(xyz)} points, {len(data[name]['cams'])} cameras")
    # source labels: points equal to another dataset's whole cloud at the start (e.g. tracks inside dense)
    for a in data.values():
        a["prefix"] = 0
        for b in data.values():
            n = len(b["xyz"])
            if b is not a and n < len(a["xyz"]) and np.array_equal(a["xyz"][:n], b["xyz"]):
                a["prefix"] = max(a["prefix"], n)

    first = next(iter(data.values()))
    C = np.stack([-w2c[:3, :3].T @ w2c[:3, 3] for _, w2c in first["cams"]])
    offset = C.mean(0)
    extent = float(np.ptp(C, axis=0).max())

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    g = server.gui
    gui_set = g.add_dropdown("Dataset", tuple(data))
    gui_color = g.add_dropdown("Color", ("RGB", "source (orange = tracks)", "height"))
    gui_size = g.add_slider("Point size (m)", min=0.02, max=2.0, step=0.02, initial_value=0.3)
    gui_cams = g.add_checkbox("Show cameras", True)
    gui_info = g.add_markdown("")
    handles = {}

    def colors(d):
        mode = gui_color.value
        if mode == "RGB":
            return d["rgb"]
        if mode == "height":
            return height_colors(d["xyz"][:, 2])
        c = np.tile(np.array([[0, 200, 220]], np.uint8), (len(d["xyz"]), 1))
        c[:d["prefix"]] = [255, 140, 0]
        return c

    def redraw(_=None):
        d = data[gui_set.value]
        handles["pc"] = server.scene.add_point_cloud("points", d["xyz"] - offset, colors(d),
                                                     point_size=gui_size.value, point_shape="circle")
        gui_info.content = (f"**{gui_set.value}**: {len(d['xyz']):,} points"
                            + (f" ({d['prefix']:,} tracks + {len(d['xyz']) - d['prefix']:,} VGGT)" if d["prefix"] else ""))
        for i, (name, w2c) in enumerate(d["cams"]):
            c2w = np.linalg.inv(w2c)
            server.scene.add_camera_frustum(
                f"cams/{i:04d}", fov=d["fov"], aspect=d["aspect"], scale=extent * 0.006,
                wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz, position=c2w[:3, 3] - offset,
                color=(40, 90, 255), visible=gui_cams.value)

    def resize(_):
        handles["pc"].point_size = gui_size.value

    gui_set.on_update(redraw)
    gui_color.on_update(redraw)
    gui_size.on_update(resize)
    gui_cams.on_update(lambda _: redraw())
    redraw()
    print(f"open http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
