"""VGGT backbone adapter (third_party/vggt, facebook/VGGT-1B)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_VGGT_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "vggt"
if str(_VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VGGT_ROOT))

from vggt.models.vggt import VGGT                                   # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images           # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri        # noqa: E402


@dataclass
class BatchResult:
    """One batch of FM predictions, in the batch's own frame (frame of view 0)."""
    extrinsic: torch.Tensor     # (S, 3, 4) world-to-camera, OpenCV
    intrinsic: torch.Tensor     # (S, 3, 3) at model resolution
    depth: torch.Tensor         # (S, H, W)
    depth_conf: torch.Tensor    # (S, H, W)
    image_hw: tuple[int, int]
    inference_time: float
    peak_vram_gb: float


class VGGTAdapter:
    name = "vggt_1b"
    license = "VGGT license (facebook/VGGT-1B; non-commercial — check before product use)"
    is_metric = False
    supports_pose_conditioning = False
    has_matching_head = True

    def __init__(self, model_id: str = "facebook/VGGT-1B", device: str = "cuda"):
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = VGGT.from_pretrained(model_id).to(device).eval()

    @torch.inference_mode()
    def infer(self, image_paths: list[str | Path]) -> BatchResult:
        images = load_and_preprocess_images([str(p) for p in image_paths], mode="crop").to(self.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        # same as VGGT.forward: aggregator in bf16, heads in fp32; skip point & track heads
        imgs = images[None]
        with torch.autocast("cuda", dtype=self.dtype):
            tokens, ps_idx = self.model.aggregator(imgs)
        with torch.autocast("cuda", enabled=False):
            pose_enc = self.model.camera_head(tokens)[-1]
            depth, depth_conf = self.model.depth_head(tokens, images=imgs, patch_start_idx=ps_idx)
        torch.cuda.synchronize()
        dt = time.time() - t0
        extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), images.shape[-2:])
        return BatchResult(extr[0], intr[0], depth[0, ..., 0].float(), depth_conf[0].float(),
                           tuple(images.shape[-2:]), dt, torch.cuda.max_memory_allocated() / 1e9)
