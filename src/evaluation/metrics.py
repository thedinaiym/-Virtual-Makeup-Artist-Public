"""
src/evaluation/metrics.py
──────────────────────────
Evaluation metrics: FID, L1, SSIM, LPIPS, inference latency.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) in [-1,1] → (B,H,W,3) uint8."""
    arr = ((t.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return arr.transpose(0, 2, 3, 1)   # BCHW → BHWC


# ─── L1 ─────────────────────────────────────────────────────

def compute_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


# ─── SSIM ────────────────────────────────────────────────────

def compute_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SSIM over batch using skimage."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        logger.warning("scikit-image not installed; skipping SSIM")
        return float("nan")

    pred_np   = tensor_to_uint8(pred)
    target_np = tensor_to_uint8(target)
    scores = [
        ssim_fn(p, t, channel_axis=-1, data_range=255)
        for p, t in zip(pred_np, target_np)
    ]
    return float(np.mean(scores))


# ─── LPIPS ───────────────────────────────────────────────────

class LPIPSMetric:
    """Lazy-loaded LPIPS metric."""

    def __init__(self, device: torch.device, net: str = "alex"):
        self.device = device
        self._model = None
        self._net   = net

    def _get_model(self):
        if self._model is None:
            try:
                import lpips
                self._model = lpips.LPIPS(net=self._net).to(self.device)
            except ImportError:
                logger.warning("lpips not installed; LPIPS will be NaN")
        return self._model

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        model = self._get_model()
        if model is None:
            return float("nan")
        scores = model(pred.to(self.device), target.to(self.device))
        return float(scores.mean().item())


# ─── FID ─────────────────────────────────────────────────────

def compute_fid(
    real_dir:  str | Path,
    fake_dir:  str | Path,
    batch_size: int = 50,
    device:     str = "cuda",
) -> float:
    """
    Compute FID between two directories of PNG images.
    Requires clean-fid package.
    """
    try:
        from cleanfid import fid
        score = fid.compute_fid(
            str(real_dir), str(fake_dir),
            device=device, batch_size=batch_size,
            verbose=False,
        )
        return float(score)
    except ImportError:
        logger.warning("clean-fid not installed; FID will be NaN")
        return float("nan")


# ─── Inference latency ───────────────────────────────────────

def measure_inference_ms(
    model:    torch.nn.Module,
    device:   torch.device,
    image_size: int = 256,
    n_warmup: int = 10,
    n_runs:   int = 100,
) -> float:
    """Returns mean inference time in milliseconds (single image)."""
    model.eval()
    dummy_x = torch.randn(1, 5, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) / n_runs * 1000


# ─── Full Evaluation Pass ────────────────────────────────────

@torch.no_grad()
def evaluate(
    generator,
    val_loader,
    device: torch.device,
    lpips_metric: Optional[LPIPSMetric] = None,
    image_size: int = 256,
) -> dict:
    """
    Runs the full evaluation suite.

    Returns dict with keys: l1, ssim, lpips, inference_ms
    (FID requires saving files separately — see scripts/evaluate.py)
    """
    generator.eval()
    l1_list, ssim_list, lpips_list = [], [], []

    for batch in val_loader:
        x    = batch["x"].to(device)
        y    = batch["y"].to(device)
        inten= batch["intensity"].to(device)
        fake = generator(x, inten)

        l1_list.append(compute_l1(fake, y))
        ssim_list.append(compute_ssim_batch(fake, y))
        if lpips_metric:
            lpips_list.append(lpips_metric(fake, y))

    ms = measure_inference_ms(generator, device, image_size)

    return {
        "l1":           float(np.mean(l1_list)),
        "ssim":         float(np.mean(ssim_list)),
        "lpips":        float(np.mean(lpips_list)) if lpips_list else float("nan"),
        "inference_ms": ms,
    }
