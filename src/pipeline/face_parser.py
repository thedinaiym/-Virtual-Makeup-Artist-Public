"""
src/pipeline/face_parser.py
────────────────────────────
Wraps MediaPipe Face Mesh to extract:
  • 468 3-D landmarks  →  polygon masks per zone
  • Face bounding box
  • Face shape classification
  • Dominant skin-tone RGB sampled from cheek region
"""

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── MediaPipe landmark index groups ──────────────────────────
#  (trimmed to the most reliable subsets)
LANDMARK_GROUPS = {
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                   308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
    "lips_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                   415, 310, 311, 312, 13, 82, 81, 80, 191],
    "left_eye":   [362, 382, 381, 380, 374, 373, 390, 249, 263,
                   466, 388, 387, 386, 385, 384, 398],
    "right_eye":  [33, 7, 163, 144, 145, 153, 154, 155, 133,
                   173, 157, 158, 159, 160, 161, 246],
    "left_brow":  [276, 283, 282, 295, 285, 336, 296, 334, 293, 300],
    "right_brow": [46, 53, 52, 65, 55, 107, 66, 105, 63, 70],
    "left_cheek": [116, 123, 147, 213, 192, 214, 210, 211, 212,
                   202, 204, 194, 32, 31, 228, 229, 230, 231],
    "right_cheek": [345, 352, 376, 433, 416, 434, 430, 431, 432,
                    422, 424, 418, 262, 261, 448, 449, 450, 451],
    "nose":       [1, 2, 98, 327, 4, 5, 197, 195, 196],
    "forehead":   [10, 338, 297, 332, 284, 251, 389, 356, 454,
                   323, 361, 288, 397, 365, 379, 378, 400, 377,
                   152, 148, 176, 149, 150, 136, 172, 58, 132,
                   93, 234, 127, 162, 21, 54, 103, 67, 109],
    "face_oval":  [10, 338, 297, 332, 284, 251, 389, 356, 454,
                   323, 361, 288, 397, 365, 379, 378, 400, 377,
                   152, 148, 176, 149, 150, 136, 172, 58, 132,
                   93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
}


@dataclass
class FaceParseResult:
    landmarks_px: np.ndarray           # (468, 2) pixel coords
    landmarks_norm: np.ndarray         # (468, 3) normalised 3D
    masks: dict[str, np.ndarray]       # zone → binary mask (H, W)
    skin_rgb: np.ndarray               # (3,) mean cheek RGB
    face_shape: str                    # "oval" | "round" | ...
    bbox: tuple[int, int, int, int]    # x1, y1, x2, y2
    h: int
    w: int


class FaceParser:
    """
    Stateless wrapper around MediaPipe FaceMesh.

    Usage:
        parser = FaceParser(min_confidence=0.8)
        result = parser.parse(bgr_image)   # returns FaceParseResult | None
    """

    def __init__(self, min_confidence: float = 0.8, refine_landmarks: bool = True):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_confidence,
        )

    # ─────────────────────────────────────────────────────────
    def parse(self, bgr: np.ndarray) -> Optional[FaceParseResult]:
        """
        Parameters
        ----------
        bgr : H×W×3 uint8

        Returns
        -------
        FaceParseResult or None if no face detected
        """
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        # ── Pixel coordinates ─────────────────────────────
        lm_px = np.array(
            [(lm.x * w, lm.y * h) for lm in face.landmark],
            dtype=np.float32,
        )  # (468, 2)
        lm_norm = np.array(
            [(lm.x, lm.y, lm.z) for lm in face.landmark],
            dtype=np.float32,
        )  # (468, 3)

        # ── Build binary masks ────────────────────────────
        masks: dict[str, np.ndarray] = {}
        for zone, indices in LANDMARK_GROUPS.items():
            pts = lm_px[indices].astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            masks[zone] = mask

        # ── Skin colour from cheeks ───────────────────────
        cheek_mask = cv2.bitwise_or(masks["left_cheek"], masks["right_cheek"])
        skin_rgb = self._sample_skin(bgr, cheek_mask)

        # ── Face shape ────────────────────────────────────
        face_shape = self._classify_face_shape(lm_px, h, w)

        # ── Bounding box ──────────────────────────────────
        oval_pts = lm_px[LANDMARK_GROUPS["face_oval"]].astype(np.int32)
        x1, y1 = oval_pts.min(axis=0)
        x2, y2 = oval_pts.max(axis=0)

        return FaceParseResult(
            landmarks_px=lm_px,
            landmarks_norm=lm_norm,
            masks=masks,
            skin_rgb=skin_rgb,
            face_shape=face_shape,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            h=h,
            w=w,
        )

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _sample_skin(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mean BGR of non-zero mask pixels, returned as RGB."""
        pixels = bgr[mask > 0]
        if len(pixels) == 0:
            return np.array([200, 170, 150], dtype=np.float32)
        mean_bgr = pixels.mean(axis=0)
        return mean_bgr[::-1].astype(np.float32)  # BGR → RGB

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _classify_face_shape(lm_px: np.ndarray, h: int, w: int) -> str:
        """
        Heuristic face-shape classification using landmark geometry.

        Rules (simplified professional guide):
          forehead_w / jaw_w  →  heart vs square vs round
          face_h / face_w     →  oblong vs others
          jaw angle           →  oval
        """
        # Key point indices
        jaw_left  = lm_px[234]
        jaw_right = lm_px[454]
        chin      = lm_px[152]
        forehead_l= lm_px[54]
        forehead_r= lm_px[284]
        cheek_l   = lm_px[234]
        cheek_r   = lm_px[454]
        top       = lm_px[10]

        jaw_w  = float(np.linalg.norm(jaw_left  - jaw_right))
        fore_w = float(np.linalg.norm(forehead_l - forehead_r))
        cheek_w= float(np.linalg.norm(cheek_l   - cheek_r))
        face_h = float(np.linalg.norm(chin - top))

        ratio_hw = face_h / (jaw_w + 1e-6)
        ratio_fw = fore_w / (jaw_w + 1e-6)

        if ratio_hw > 1.75:
            return "oblong"
        if ratio_fw > 1.2:
            return "heart"
        if abs(jaw_w - cheek_w) / (cheek_w + 1e-6) < 0.05 and ratio_hw < 1.2:
            return "round"
        if abs(jaw_w - fore_w) / (fore_w + 1e-6) < 0.1 and ratio_hw < 1.4:
            return "square"
        return "oval"

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
