"""Interactive view-graph viewer (viser), following VGGT's demo_viser conventions.

Select a camera (click its frustum, or use the slider); every other frustum is
recolored by its relation score to the selected one: red = highly correlated,
blue = not correlated. The score source is any (N, N) matrix, so the same
viewer shows reference covisibility now and FM-predicted scores later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import viser
import viser.transforms as vtf
from PIL import Image

from fgsfm.graph.view_graph import ViewGraph

SELECTED_RGB = np.array([0, 220, 0])
BLUE = np.array([30, 60, 255], np.float64)
RED = np.array([255, 20, 20], np.float64)
TIE_POINT_RGB = np.array([255, 220, 0], np.uint8)
UNEVALUATED_RGB = np.array([150, 150, 150], np.uint8)     # pair never scored (NaN), e.g. not co-batched


def score_to_rgb(t: np.ndarray) -> np.ndarray:
    """t in [0, 1] -> blue..red, interpolated in RGB."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    return ((1 - t) * BLUE + t * RED).astype(np.uint8)


def load_thumbnail(path: Path, max_side: int) -> np.ndarray:
    im = Image.open(path)
    im.draft("RGB", (max_side, max_side))       # fast JPEG DCT downscale
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    return np.asarray(im)


class CovisibilityViewer:
    def __init__(self, graph: ViewGraph, score_fns: dict[str, Callable[[], np.ndarray]],
                 points_xyz: np.ndarray | None = None, points_rgb: np.ndarray | None = None,
                 points_of_image: Callable[[int], np.ndarray] | None = None,
                 port: int = 8080, frustum_scale: float | None = None,
                 thumbnails: bool = False, max_points: int = 300_000):
        self.graph = graph
        self.score_fns = score_fns
        self.score_cache: dict[str, np.ndarray] = {}
        self.points_of_image = points_of_image
        self.selected = 0

        centers = np.stack([c.center for c in graph.cameras])
        self.offset = centers.mean(axis=0)                   # recenter for a usable orbit
        if frustum_scale is None:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(centers).query(centers, 2)
            frustum_scale = 0.6 * float(np.median(d[:, 1]))

        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self.server.scene.set_up_direction(self._up_direction())

        self.points_xyz = self.points_rgb = None
        self.point_cloud = None
        if points_xyz is not None:
            self._n_points_total = len(points_xyz)
            if len(points_xyz) > max_points:
                keep = np.random.default_rng(0).choice(len(points_xyz), max_points, replace=False)
                self._point_index = keep
                points_xyz, points_rgb = points_xyz[keep], points_rgb[keep]
            else:
                self._point_index = np.arange(len(points_xyz))
            self.points_xyz = (points_xyz - self.offset).astype(np.float32)
            self.points_rgb = points_rgb
            self.point_cloud = self.server.scene.add_point_cloud(
                "points", self.points_xyz, self.points_rgb, point_size=0.6 * frustum_scale / 4)

        self._build_gui(frustum_scale)
        self._build_frustums(frustum_scale, thumbnails)
        self.select(0)

    # ---------------------------------------------------------------- scene

    def _up_direction(self) -> tuple[float, float, float]:
        """World up = mean camera 'up' (-y in OpenCV camera axes), snapped to an axis if close."""
        ups = np.stack([-c.pose_c2w[:3, 1] for c in self.graph.cameras])
        fwd = np.stack([c.pose_c2w[:3, 2] for c in self.graph.cameras])
        # nadir rigs: camera y is horizontal, so use the reverse viewing direction instead
        up = -fwd.mean(0) if np.linalg.norm(fwd.mean(0)) > np.linalg.norm(ups.mean(0)) else ups.mean(0)
        up /= np.linalg.norm(up)
        return tuple(float(x) for x in up)

    def _build_frustums(self, scale: float, thumbnails: bool) -> None:
        self.frustums: list[viser.CameraFrustumHandle] = []
        for cam in self.graph.cameras:
            c2w = cam.pose_c2w
            fy = cam.K[1, 1]
            fov = 2 * np.arctan2(cam.height / 2, fy)
            img = None
            if thumbnails and cam.image_path is not None and cam.image_path.exists():
                img = load_thumbnail(cam.image_path, 96)
            fr = self.server.scene.add_camera_frustum(
                f"cams/{cam.image_id:05d}", fov=float(fov), aspect=cam.width / cam.height,
                scale=scale, thickness=0.04 * scale, image=img,
                wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz, position=c2w[:3, 3] - self.offset,
                color=tuple(BLUE.astype(int)))
            fr.on_click(lambda _ev, i=cam.image_id: self.select(i))
            self.frustums.append(fr)
        self.lines = None

    # ---------------------------------------------------------------- gui

    def _build_gui(self, scale: float) -> None:
        g = self.server.gui
        n = len(self.graph)
        with g.add_folder("Selection"):
            self.gui_cam = g.add_slider("Camera", min=0, max=n - 1, step=1, initial_value=0)
            self.gui_name = g.add_text("Name", initial_value="", disabled=True)
            self.gui_fly = g.add_button("Fly to camera")
        with g.add_folder("Scores"):
            self.gui_score = g.add_dropdown("Score", tuple(self.score_fns.keys()))
            self.gui_norm = g.add_dropdown("Color scale", ("relative to best", "absolute"))
            self.gui_gamma = g.add_slider("Gamma", min=0.2, max=2.0, step=0.05, initial_value=0.6)
            self.gui_hide = g.add_checkbox("Hide uncorrelated", initial_value=False)
            self.gui_topk = g.add_slider("Top-k links", min=0, max=50, step=1, initial_value=10)
        with g.add_folder("Display"):
            self.gui_scale = g.add_slider("Frustum scale", min=0.1 * scale, max=5 * scale,
                                          step=0.05 * scale, initial_value=scale)
            self.gui_points = g.add_checkbox("Show points", initial_value=self.point_cloud is not None)
            self.gui_tie = g.add_checkbox("Highlight tie points", initial_value=True)
        self.gui_info = g.add_markdown("")
        self.gui_image = g.add_image(np.zeros((8, 12, 3), np.uint8), label="Selected image")

        self.gui_cam.on_update(lambda _: self.select(int(self.gui_cam.value)))
        for h in (self.gui_score, self.gui_norm, self.gui_gamma, self.gui_hide, self.gui_topk, self.gui_tie):
            h.on_update(lambda _: self.refresh())
        self.gui_scale.on_update(lambda _: self._set_scale(self.gui_scale.value))
        self.gui_points.on_update(lambda _: self._set_points_visible(self.gui_points.value))
        self.gui_fly.on_click(lambda ev: self._fly_to(ev.client))

    def _set_scale(self, s: float) -> None:
        for fr in self.frustums:
            fr.scale, fr.thickness = s, 0.04 * s

    def _set_points_visible(self, v: bool) -> None:
        if self.point_cloud is not None:
            self.point_cloud.visible = v

    def _fly_to(self, client: viser.ClientHandle | None) -> None:
        if client is None:
            return
        fr = self.frustums[self.selected]
        client.camera.wxyz = fr.wxyz
        client.camera.position = fr.position

    # ---------------------------------------------------------------- logic

    def scores(self, key: str | None = None) -> np.ndarray:
        key = key or self.gui_score.value
        if key not in self.score_cache:
            self.score_cache[key] = self.score_fns[key]()
        return self.score_cache[key]

    def select(self, i: int) -> None:
        self.selected = i
        if int(self.gui_cam.value) != i:
            self.gui_cam.value = i          # triggers on_update -> select again, guarded above
            return
        cam = self.graph.cameras[i]
        self.gui_name.value = cam.name
        if cam.image_path is not None and cam.image_path.exists():
            self.gui_image.image = load_thumbnail(cam.image_path, 640)
        self.refresh()

    def refresh(self) -> None:
        i = self.selected
        raw = self.scores()[i].astype(np.float64).copy()
        raw[i] = 0.0
        unevaluated = np.isnan(raw)
        s = np.nan_to_num(raw, nan=0.0)
        if self.gui_norm.value == "relative to best":
            t = s / s.max() if s.max() > 0 else s
        else:
            t = s / max(1.0, s.max()) if self.gui_score.value == "shared" else s
        t = t ** self.gui_gamma.value
        rgb = score_to_rgb(t)
        rgb[unevaluated] = UNEVALUATED_RGB
        rgb[i] = SELECTED_RGB
        hide = self.gui_hide.value
        for j, fr in enumerate(self.frustums):
            fr.color = tuple(int(x) for x in rgb[j])
            fr.visible = (not hide) or j == i or s[j] > 0

        self._draw_links(i, s, rgb)
        self._highlight_points(i)

        order = [j for j in np.argsort(-s)[:10] if s[j] > 0]
        fmt = ".0f" if self.gui_score.value == "shared" else ".3f"
        others = {k: self.scores(k) for k in self.score_fns if k not in (self.gui_score.value, "shared")}
        hdr = " | ".join([self.gui_score.value] + list(others))
        rows = "\n".join(
            f"| {j} | " + " | ".join([f"{s[j]:{fmt}}"] + [f"{o[i, j]:.3f}" for o in others.values()]) + " |"
            for j in order)
        connected = int((s > 0).sum())
        self.gui_info.content = (
            f"**Camera {i}** — {connected}/{len(s) - 1} cameras with score > 0"
            + (f", {int(unevaluated.sum())} not evaluated (gray)" if unevaluated.any() else "") + "\n\n"
            f"| id | {hdr} |\n|" + "---|" * (2 + len(others)) + f"\n{rows}")

    def _draw_links(self, i: int, s: np.ndarray, rgb: np.ndarray) -> None:
        if self.lines is not None:
            self.lines.remove()
            self.lines = None
        k = int(self.gui_topk.value)
        nbrs = [j for j in np.argsort(-s)[:k] if s[j] > 0]
        if not nbrs:
            return
        p0 = self.frustums[i].position
        pts = np.stack([np.stack([p0, self.frustums[j].position]) for j in nbrs]).astype(np.float32)
        cols = np.stack([np.stack([rgb[j], rgb[j]]) for j in nbrs])
        self.lines = self.server.scene.add_line_segments(
            "links", pts, cols, thickness=0.05 * self.gui_scale.value)

    def _highlight_points(self, i: int) -> None:
        if self.point_cloud is None:
            return
        colors = self.points_rgb.copy()
        if self.gui_tie.value and self.points_of_image is not None:
            seen = np.zeros(self._n_points_total, bool)
            seen[self.points_of_image(i)] = True
            colors[seen[self._point_index]] = TIE_POINT_RGB
        self.point_cloud.colors = colors

    def run_forever(self) -> None:
        import time
        print(f"viser running at http://localhost:{self.server.get_port()}  (Ctrl+C to quit)")
        while True:
            time.sleep(1.0)
