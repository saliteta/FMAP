"""VGGT backbone adapter (third_party/vggt, facebook/VGGT-1B)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    images: torch.Tensor | None = None       # (S, 3, H, W) in [0, 1]
    descriptors: torch.Tensor | None = None  # (S, D) L2-normalized per-frame global descriptor


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
        # per-frame DINOv2 patch tokens (before cross-frame attention) -> retrieval descriptor
        self._patch_tokens = None
        self.model.aggregator.patch_embed.register_forward_hook(self._grab_patch_tokens)

    def _grab_patch_tokens(self, _module, _inp, out):
        self._patch_tokens = out["x_norm_patchtokens"] if isinstance(out, dict) else out

    @staticmethod
    def _gem(tokens: torch.Tensor, p: float = 3.0) -> torch.Tensor:
        """Generalized-mean pooling over patches of L2-normalized tokens -> (S, D)."""
        x = F.normalize(tokens.float(), dim=-1).clamp(min=1e-6)
        return F.normalize(x.pow(p).mean(1).pow(1 / p), dim=-1)

    @torch.inference_mode()
    def infer(self, image_paths: list[str | Path], keep_images: bool = False) -> BatchResult:
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
        desc = self._gem(self._patch_tokens) if self._patch_tokens is not None else None
        return BatchResult(extr[0], intr[0], depth[0, ..., 0].float(), depth_conf[0].float(),
                           tuple(images.shape[-2:]), dt, torch.cuda.max_memory_allocated() / 1e9,
                           images if keep_images else None, desc)
