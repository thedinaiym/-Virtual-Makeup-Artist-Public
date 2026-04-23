"""
src/pipeline/expert_system.py
──────────────────────────────
Rule-Based Expert System that maps:
  (face_shape, skin_undertone)  →  MakeupPlan

A MakeupPlan holds:
  • lip_color   : (R, G, B)
  • blush_color : (R, G, B)
  • contour_color: (R, G, B)
  • highlight_color: (R, G, B)
  • eyeshadow_colors: list of (R,G,B)
  • contour_zones: list[str]  — which face zones to contour
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ─── Undertone detection ─────────────────────────────────────

def detect_undertone(skin_rgb: np.ndarray) -> str:
    """
    Classify skin undertone as 'warm' | 'cool' | 'neutral'
    using the relationship between R, G, B channels and
    a simplified ITA (Individual Typology Angle) proxy.

    Parameters
    ----------
    skin_rgb : (3,) float array — mean RGB of cheek region
    """
    r, g, b = skin_rgb.astype(float)

    # Warm: red/yellow dominant  (r > b by a clear margin)
    # Cool: pink/blue dominant   (b ≈ r, or b slightly higher)
    diff_rb = r - b        # positive → warm, negative → cool
    diff_rg = r - g        # high → reddish/warm

    if diff_rb > 15:
        return "warm"
    if diff_rb < -5:
        return "cool"
    return "neutral"


def skin_lightness(skin_rgb: np.ndarray) -> str:
    """'light' | 'medium' | 'dark' based on luminance."""
    r, g, b = skin_rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum > 180:
        return "light"
    if lum > 110:
        return "medium"
    return "dark"


# ─── Makeup Plan ─────────────────────────────────────────────

@dataclass
class MakeupPlan:
    lip_color:        tuple[int, int, int]
    blush_color:      tuple[int, int, int]
    contour_color:    tuple[int, int, int]
    highlight_color:  tuple[int, int, int]
    eyeshadow_colors: list[tuple[int, int, int]]
    contour_zones:    list[str]             # subset of mask keys
    lip_alpha:        float = 0.55
    blush_alpha:      float = 0.40
    contour_alpha:    float = 0.35
    highlight_alpha:  float = 0.30
    eyeshadow_alpha:  float = 0.50


# ─── Palette Library ─────────────────────────────────────────
#  Colours are professional makeup artist recommendations mapped
#  to approximate RGB values.

_PALETTES: dict[str, dict] = {
    # ── WARM undertone ──────────────────────────────────────
    "warm": {
        "light": {
            "lip":       (210, 100,  90),   # peachy coral
            "blush":     (235, 150, 120),   # warm peach
            "eyeshadow": [(180, 130,  90), (210, 170, 120), (90, 55, 30)],
        },
        "medium": {
            "lip":       (190,  70,  60),   # terracotta
            "blush":     (210, 120,  80),   # burnt sienna
            "eyeshadow": [(160, 100,  60), (190, 140,  80), (80, 45, 20)],
        },
        "dark": {
            "lip":       (160,  50,  40),   # deep amber
            "blush":     (180,  90,  60),   # deep copper
            "eyeshadow": [(130,  70,  30), (160, 110,  50), (60, 30, 10)],
        },
    },
    # ── COOL undertone ──────────────────────────────────────
    "cool": {
        "light": {
            "lip":       (220,  80, 110),   # rose pink
            "blush":     (240, 160, 180),   # baby pink
            "eyeshadow": [(160, 130, 180), (200, 170, 210), (80, 60, 100)],
        },
        "medium": {
            "lip":       (190,  50,  90),   # berry
            "blush":     (210, 110, 140),   # mauve
            "eyeshadow": [(140, 100, 160), (170, 140, 190), (60, 40,  80)],
        },
        "dark": {
            "lip":       (150,  30,  70),   # deep plum
            "blush":     (170,  70, 110),   # deep rose
            "eyeshadow": [(110,  60, 130), (140, 100, 160), (40, 20,  60)],
        },
    },
    # ── NEUTRAL undertone ───────────────────────────────────
    "neutral": {
        "light": {
            "lip":       (215,  90, 100),   # dusty rose
            "blush":     (235, 155, 150),   # soft rose
            "eyeshadow": [(170, 130, 135), (200, 165, 160), (85, 55, 50)],
        },
        "medium": {
            "lip":       (190,  65,  75),   # mauve rose
            "blush":     (210, 120, 115),   # warm rose
            "eyeshadow": [(150,  95, 100), (180, 130, 130), (65, 35, 30)],
        },
        "dark": {
            "lip":       (155,  40,  50),   # deep rose
            "blush":     (175,  80,  80),   # deep mauve
            "eyeshadow": [(120,  65,  70), (150, 100, 100), (45, 20, 20)],
        },
    },
}

# Contour/highlight are skin-tone relative (darker / lighter shade)
def _contour_color(skin_rgb: np.ndarray) -> tuple[int, int, int]:
    r, g, b = (np.clip(skin_rgb - 40, 0, 255)).astype(int)
    return (int(r), int(g), int(b))

def _highlight_color(skin_rgb: np.ndarray) -> tuple[int, int, int]:
    r, g, b = (np.clip(skin_rgb + 35, 0, 255)).astype(int)
    return (int(r), int(g), int(b))


# ─── Contour Maps per Face Shape ─────────────────────────────
#  Specifies which mask zones receive contouring (shadow)
#  and which receive highlight.

_CONTOUR_RULES: dict[str, dict] = {
    "oval": {
        # Oval is the "ideal" — minimal correction needed
        "contour": ["face_oval"],
        "highlight": ["nose", "forehead"],
    },
    "round": {
        # Elongate — contour sides, highlight centre vertical
        "contour": ["left_cheek", "right_cheek", "face_oval"],
        "highlight": ["forehead", "nose"],
    },
    "square": {
        # Soften jaw — contour corners and jaw
        "contour": ["left_cheek", "right_cheek"],
        "highlight": ["forehead", "nose"],
    },
    "heart": {
        # Balance narrow chin — contour forehead sides, highlight chin
        "contour": ["forehead"],
        "highlight": ["nose"],
    },
    "oblong": {
        # Widen — contour top/bottom, blush wide
        "contour": ["forehead"],
        "highlight": ["left_cheek", "right_cheek"],
    },
}


# ─── Main Expert Function ─────────────────────────────────────

class ExpertSystem:
    """
    Stateless expert system.

    Usage:
        plan = ExpertSystem().get_plan(skin_rgb, face_shape)
    """

    def get_plan(self, skin_rgb: np.ndarray, face_shape: str) -> MakeupPlan:
        undertone = detect_undertone(skin_rgb)
        lightness = skin_lightness(skin_rgb)

        palette = _PALETTES[undertone][lightness]
        rules   = _CONTOUR_RULES.get(face_shape, _CONTOUR_RULES["oval"])

        return MakeupPlan(
            lip_color        = palette["lip"],
            blush_color      = palette["blush"],
            contour_color    = _contour_color(skin_rgb),
            highlight_color  = _highlight_color(skin_rgb),
            eyeshadow_colors = palette["eyeshadow"],
            contour_zones    = rules["contour"],
        )

    # Convenience method: also returns debug info
    def analyze(self, skin_rgb: np.ndarray, face_shape: str) -> dict:
        undertone = detect_undertone(skin_rgb)
        lightness = skin_lightness(skin_rgb)
        plan      = self.get_plan(skin_rgb, face_shape)
        return {
            "undertone":  undertone,
            "lightness":  lightness,
            "face_shape": face_shape,
            "plan":       plan,
        }
