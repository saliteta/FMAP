"""Bundle adjustment with the FULL pinhole intrinsics optimized (focal AND principal point).

InstantSfM's ReprojectionModel keeps the principal point as a fixed input (camera_pps) and optimizes
only focal / distortion. On HAV the calibrated principal point is 20 / 27 px off the image center, so a
fixed center pp biases poses. This model reuses InstantSfM's data preparation and bae's LM + PCG
solver unchanged, but its intrinsic parameter vector contains the principal point:

    mode "focal+pp":     [fx, fy, cx, cy]
    mode "shared_f+pp":  [f, cx, cy]          (fx = fy, as in COLMAP SIMPLE_* models)
"""
from __future__ import annotations

import numpy as np
import pypose as pp
import torch
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.utils.pysolvers import PCG
from pypose.optim.kernel import Huber
from torch import nn

import fgsfm.ba.instantsfm_ba  # noqa: F401  (puts third_party/InstantSfM on sys.path)
from instantsfm.processors.bundle_adjustment import _prepare_single_camera_data


@map_transform
def _project_fxfycxcy(points, extrinsics, intrinsics):
    pc = pp.SE3(extrinsics).Act(points)
    return pc[..., :2] / pc[..., 2:3] * intrinsics[..., 0:2] + intrinsics[..., 2:4]


@map_transform
def _project_fcxcy(points, extrinsics, intrinsics):
    pc = pp.SE3(extrinsics).Act(points)
    return pc[..., :2] / pc[..., 2:3] * intrinsics[..., 0:1] + intrinsics[..., 1:3]


class FullIntrinsicsReprojection(nn.Module):
    def __init__(self, image_extrs, intrinsics, points_3d, project):
        super().__init__()
        self.extrinsics = nn.Parameter(TrackingTensor(image_extrs))
        self.intrinsics = nn.Parameter(TrackingTensor(intrinsics))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.extrinsics.trim_SE3_grad = True
        self.project = project

    def forward(self, points_2d, image_indices, camera_indices, point_indices):
        return self.project(self.points_3d[point_indices], self.extrinsics[image_indices],
                            self.intrinsics[camera_indices]) - points_2d


def solve_full_intrinsics(cameras, images, tracks, options: dict, mode: str = "focal+pp", device: str = "cuda:0"):
    """Same contract as InstantSfM TorchBA.Solve (PINHOLE cameras): updates containers in place."""
    (image_extrs, camera_intrs, points_3d, camera_pps, remaining_indices, pp_indices, points_2d,
     image_indices, camera_indices, point_indices, _, image_idx2id, camera_idx2id) = _prepare_single_camera_data(
        cameras, images, tracks, _PINHOLE_INFO, device, options["min_num_view_per_track"], False)
    if mode == "shared_f+pp":
        intr = torch.cat([camera_intrs.mean(dim=1, keepdim=True), camera_pps], dim=1)
        project = _project_fcxcy
    else:
        intr = torch.cat([camera_intrs, camera_pps], dim=1)
        project = _project_fxfycxcy
    model = FullIntrinsicsReprojection(image_extrs, intr, points_3d, project)
    strategy = pp.optim.strategy.TrustRegion(radius=1e4, max=1e10, up=2.0, down=0.5 ** 4)
    optimizer = LM(model, strategy=strategy, solver=PCG(tol=1e-5), kernel=Huber(options["thres_loss_function"]),
                   reject=30)
    inputs = dict(points_2d=points_2d, image_indices=image_indices, camera_indices=camera_indices,
                  point_indices=point_indices)
    hist = []
    for _ in range(options["max_num_iterations"]):
        hist.append(float(optimizer.step(inputs)))
        if len(hist) >= 8:
            prev, recent = np.mean(hist[-8:-4]), np.mean(hist[-4:])
            if abs(prev - recent) / max(prev, 1e-12) < options["function_tolerance"] or hist[-1] == hist[-2]:
                break
    with torch.no_grad():
        tracks.xyzs[:] = model.points_3d.detach().cpu().numpy()
        E = pp.SE3(model.extrinsics.detach()).matrix().cpu().numpy()
        for k in range(len(E)):
            images.world2cams[image_idx2id[k]] = E[k]
        I = model.intrinsics.detach().cpu().numpy()
        for k in range(len(I)):
            fx, fy, cx, cy = (I[k, 0], I[k, 0], I[k, 1], I[k, 2]) if mode == "shared_f+pp" else I[k, :4]
            cameras.set_params(camera_idx2id[k], np.array([fx, fy, cx, cy]))
    return len(hist)


_PINHOLE_INFO = {"name": "PINHOLE", "num_params": 4, "focal": [0, 1], "pp": [2, 3], "k": [], "p": [], "omega": [],
                 "sx": [], "optimize": [0, 1]}
