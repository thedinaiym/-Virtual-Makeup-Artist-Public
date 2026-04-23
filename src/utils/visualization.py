"""
src/utils/visualization.py
────────────────────────────
Helpers for visualising masks, pairs, and training progress.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.pipeline.face_parser import FaceParseResult


def draw_landmarks(bgr: np.ndarray, parse: FaceParseResult,
                   radius: int = 1) -> np.ndarray:
    """Draw all 468 landmarks as coloured dots."""
    out = bgr.copy()
    for x, y in parse.landmarks_px.astype(int):
        cv2.circle(out, (x, y), radius, (0, 255, 0), -1)
    return out


def draw_masks(bgr: np.ndarray, parse: FaceParseResult,
               zones: Optional[list[str]] = None) -> np.ndarray:
    """Overlay coloured masks for selected zones."""
    ZONE_COLORS = {
        "lips_outer":  (0, 0, 220),
        "left_eye":    (220, 150, 0),
        "right_eye":   (220, 150, 0),
        "left_cheek":  (0, 180, 100),
        "right_cheek": (0, 180, 100),
        "face_oval":   (180, 180, 0),
        "forehead":    (100, 0, 220),
    }
    out = bgr.copy().astype(np.float32)
    zones = zones or list(parse.masks.keys())
    for zone in zones:
        mask = parse.masks.get(zone)
        if mask is None:
            continue
        color = ZONE_COLORS.get(zone, (200, 200, 200))
        overlay = np.zeros_like(out)
        overlay[mask > 0] = color
        out = cv2.addWeighted(out, 1.0, overlay, 0.4, 0)
    return out.astype(np.uint8)


def save_pair_grid(
    x_bgr:  np.ndarray,
    y_bgr:  np.ndarray,
    save_path: str | Path,
    title: str = "",
):
    """Save side-by-side (original, makeup) image."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cv2.cvtColor(x_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (X)")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(y_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Synthetic Makeup (Y)")
    axes[1].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    history: dict[str, list[float]],
    save_path: str | Path,
):
    """Plot training curves for G and D losses."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.get("g_total", []), label="G total")
    axes[0].plot(history.get("g_l1",    []), label="G L1", linestyle="--")
    axes[0].plot(history.get("g_perc",  []), label="G Perceptual", linestyle=":")
    axes[0].set_title("Generator Loss"); axes[0].legend()
    axes[1].plot(history.get("d_loss", []), label="D loss", color="red")
    axes[1].set_title("Discriminator Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
