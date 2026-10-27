"""Interactive viser viewer for plain binary PLY point clouds (x, y, z float32 + red, green, blue uchar) and camera
sets, e.g. the files written by scripts/egr_depth_scale_viz.py.

    D=runs/egr_depth_scale_viz
    python scripts/view_ply.py intact=$D/intact_by_image.ply removed=$D/removed_by_image.ply \
        --cameras intact=$D/cameras_intact.json removed=$D/cameras_removed.json --port 8080

A camera file is a JSON list of {name, w2c (4x4), fx, fy, cx, cy, width, height, color, frame0}. Frustums use each
camera's colour (frame 0 black); the first camera set is drawn solid, the others as thinner frustums. With two
camera sets, the panel lists per camera the rotation difference and the camera-centre distance between them.
Panel: a checkbox per cloud and per camera set, point size, frustum size, up axis (VGGT-Omega's frame-0 camera
frame has +y pointing down).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as vtf

from view_point_clouds import read_ply


def load_cameras(path):
    cams = json.loads(Path(path).read_text())
    for c in cams:
        w2c = np.array(c["w2c"])
        c["R_c2w"] = w2c[:3, :3].T
        c["center"] = -w2c[:3, :3].T @ w2c[:3, 3]
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clouds", nargs="+", help="name=path/to/file.ply")
    ap.add_argument("--cameras", nargs="*", default=[], help="name=path/to/cameras.json")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.001)
    ap.add_argument("--frustum-scale", type=float, default=0.02)
    args = ap.parse_args()
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")

    clouds = {}
    for spec in args.clouds:
        name, path = spec.split("=", 1)
        xyz, rgb = read_ply(Path(path))
        clouds[name] = server.scene.add_point_cloud(f"/clouds/{name}", points=xyz, colors=rgb,
                                                    point_size=args.point_size)
        print(f"{name}: {len(xyz)} points from {path}", flush=True)

    cam_sets = {}
    for k, spec in enumerate(args.cameras):
        name, path = spec.split("=", 1)
        cams = load_cameras(path)
        handles = []
        for c in cams:
            fov = 2 * np.arctan(c["height"] / 2 / c["fy"])
            h = server.scene.add_camera_frustum(
                f"/cameras/{name}/{c['name']}", fov=float(fov), aspect=c["width"] / c["height"],
                scale=args.frustum_scale, color=tuple(c["color"]), line_width=3.0 if k == 0 else 1.5,
                wxyz=vtf.SO3.from_matrix(c["R_c2w"]).wxyz, position=c["center"])
            handles.append(h)
        cam_sets[name] = (cams, handles)
        print(f"{name}: {len(cams)} cameras from {path}", flush=True)

    with server.gui.add_folder("Point clouds"):
        for name, h in clouds.items():
            cb = server.gui.add_checkbox(name, initial_value=True)

            @cb.on_update
            def _(event, h=h):
                h.visible = event.target.value
    if cam_sets:
        with server.gui.add_folder("Cameras"):
            for name, (_, handles) in cam_sets.items():
                cb = server.gui.add_checkbox(f"{name} cameras", initial_value=True)

                @cb.on_update
                def _(event, handles=handles):
                    for h in handles:
                        h.visible = event.target.value
            fs = server.gui.add_slider("frustum size", min=0.002, max=0.1, step=0.001,
                                       initial_value=args.frustum_scale)

            @fs.on_update
            def _(_):
                for _, handles in cam_sets.values():
                    for h in handles:
                        h.scale = fs.value
    size = server.gui.add_slider("point size", min=0.0001, max=0.005, step=0.0001, initial_value=args.point_size)

    @size.on_update
    def _(_):
        for h in clouds.values():
            h.point_size = size.value
    up = server.gui.add_dropdown("up axis", options=("-y", "+y", "+z", "-z"), initial_value="-y")

    @up.on_update
    def _(_):
        server.scene.set_up_direction(up.value)

    if len(cam_sets) >= 2:                     # pose differences between the first two camera sets
        (na, (ca, _)), (nb, (cb_, _)) = list(cam_sets.items())[:2]
        spread = np.median([np.linalg.norm(c["center"] - ca[0]["center"]) for c in ca[1:]])
        rows = [f"| image | rotation diff (deg) | centre distance | as % of median distance to frame 0 |",
                "|---|---|---|---|"]
        for c1, c2 in zip(ca, cb_):
            dR = c1["R_c2w"].T @ c2["R_c2w"]
            ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            dist = np.linalg.norm(c1["center"] - c2["center"])
            label = "frame 0" if c1.get("frame0") else c1["name"][-30:]
            rows.append(f"| {label} | {ang:.2f} | {dist:.4f} | {100 * dist / spread:.1f}% |")
            print(f"  {label}: rotation diff {ang:.2f} deg, centre distance {dist:.4f} ({100 * dist / spread:.1f}% of "
                  f"the median distance to frame 0)", flush=True)
        with server.gui.add_folder(f"Pose difference: {na} vs {nb}"):
            server.gui.add_markdown("\n".join(rows))
    print(f"viewer at http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
