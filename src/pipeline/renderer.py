"""
src/pipeline/renderer.py
─────────────────────────
Applies a MakeupPlan to a BGR image given FaceParseResult masks.
Uses Gaussian softening + cv2.seamlessClone (Poisson Image Editing)
for photorealistic blending.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from .face_parser import FaceParseResult
from .expert_system import MakeupPlan


class MakeupRenderer:
    """
    Renders makeup layers onto a BGR face image.

    Parameters
    ----------
    blur_kernel : int   — Gaussian kernel size for alpha masks (must be odd)
    use_seamless: bool  — use Poisson blending (slower but realistic)
    """

    def __init__(self, blur_kernel: int = 15, use_seamless: bool = True):
        self.blur_kernel  = blur_kernel | 1   # ensure odd
        self.use_seamless = use_seamless

    # ─────────────────────────────────────────────────────────
    def render(
        self,
        bgr: np.ndarray,
        parse: FaceParseResult,
        plan: MakeupPlan,
    ) -> np.ndarray:
        """
        Apply all makeup layers in order (bottom → top).

        Returns a new BGR uint8 image (same size as input).
        """
        out = bgr.copy()

        # 1. Foundation-level contour & highlight
        out = self._apply_contour(out, parse, plan)

        # 2. Blush
        out = self._apply_blush(out, parse, plan)

        # 3. Eyeshadow (both eyes)
        out = self._apply_eyeshadow(out, parse, plan)

        # 4. Lips (outermost layer, most visible)
        out = self._apply_lips(out, parse, plan)

        return out

    # ─────────────────────────────────────────────────────────
    def _apply_color_layer(
        self,
        bgr: np.ndarray,
        mask: np.ndarray,
        rgb_color: tuple[int, int, int],
        alpha: float,
        feather: Optional[int] = None,
    ) -> np.ndarray:
        """
        Core blending primitive.

        1. Creates solid colour layer.
        2. Multiplies by softened mask (alpha channel).
        3. Alpha-composites onto base image.
        4. Optionally runs seamlessClone inside the mask ROI.
        """
        h, w = bgr.shape[:2]
        ker = feather if feather is not None else self.blur_kernel

        # Feather the mask
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (ker, ker), 0)
        soft_mask = (soft_mask / 255.0 * alpha)[..., np.newaxis]  # (H,W,1)

        # Solid colour layer in BGR
        r, g, b = rgb_color
        color_bgr = np.full((h, w, 3), (b, g, r), dtype=np.uint8)

        # Alpha blend
        blended = (bgr.astype(np.float32) * (1 - soft_mask)
                   + color_bgr.astype(np.float32) * soft_mask).astype(np.uint8)

        if not self.use_seamless:
            return blended

        # Poisson seamless clone inside tight bounding box
        pts = np.argwhere(mask > 10)
        if len(pts) < 50:
            return blended

        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        # Dilate mask slightly for seamless clone
        kernel = np.ones((5, 5), np.uint8)
        sc_mask = cv2.dilate(mask, kernel, iterations=1)

        try:
            result = cv2.seamlessClone(
                blended, bgr, sc_mask,
                (cx, cy),
                cv2.NORMAL_CLONE,
            )
        except cv2.error:
            result = blended

        return result

    # ─────────────────────────────────────────────────────────
    def _apply_lips(self, bgr, parse, plan) -> np.ndarray:
        mask = parse.masks.get("lips_outer", None)
        if mask is None:
            return bgr
        return self._apply_color_layer(
            bgr, mask, plan.lip_color, plan.lip_alpha, feather=9
        )

    def _apply_blush(self, bgr, parse, plan) -> np.ndarray:
        for zone in ("left_cheek", "right_cheek"):
            mask = parse.masks.get(zone)
            if mask is not None:
                bgr = self._apply_color_layer(
                    bgr, mask, plan.blush_color, plan.blush_alpha, feather=31
                )
        return bgr

    def _apply_contour(self, bgr, parse, plan) -> np.ndarray:
        for zone in plan.contour_zones:
            mask = parse.masks.get(zone)
            if mask is not None:
                bgr = self._apply_color_layer(
                    bgr, mask, plan.contour_color, plan.contour_alpha,
                    feather=21,
                )
        return bgr

    def _apply_eyeshadow(self, bgr, parse, plan) -> np.ndarray:
        colors = plan.eyeshadow_colors
        eye_zones = [("left_eye", "left_brow"), ("right_eye", "right_brow")]

        for (eye_zone, brow_zone), color in zip(eye_zones, colors[:2]):
            eye_mask = parse.masks.get(eye_zone, np.zeros((parse.h, parse.w), np.uint8))
            # extend shadow slightly toward brow
            brow_mask = parse.masks.get(brow_zone, np.zeros_like(eye_mask))
            combined = cv2.bitwise_or(eye_mask, brow_mask)
            combined = cv2.dilate(combined, np.ones((7, 7), np.uint8), iterations=2)

            bgr = self._apply_color_layer(
                bgr, combined, color, plan.eyeshadow_alpha, feather=13
            )
        return bgr
