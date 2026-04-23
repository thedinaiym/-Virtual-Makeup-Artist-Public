#!/usr/bin/env python3
"""
scripts/generate_dataset.py
────────────────────────────
Stage 1: Generate synthetic paired makeup dataset from CelebA / FFHQ.

Improvements over v1:
  • Female-only filter via DeepFace (skip male faces)
  • Fixed alpha values — contour_alpha 0.35 → 0.15 (removes white-face artefact)
  • Stronger Gaussian blur for smoother blending
  • Saves visualisation grid every N pairs so you can monitor quality live
  • --no-filter flag to skip gender check (faster, for quick tests)

Usage:
    python3 scripts/generate_dataset.py \
        --source ffhq-dataset/flat_images \
        --output ffhq-dataset/synthetic \
        --pairs  5000 \
        --size   256 \
        --filter-gender          # skip male faces (recommended)
        --preview-every 200      # save preview grid every 200 pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── make project root importable ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.face_parser  import FaceParser
from src.pipeline.expert_system import ExpertSystem
from src.pipeline.renderer      import MakeupRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Gender filter ───────────────────────────────────────────

class GenderFilter:
    """
    Uses DeepFace to detect gender.
    Returns True if the face is female (or detection fails → include by default).
    Falls back gracefully if deepface is not installed.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._deepface = None

        if enabled:
            try:
                import deepface as _df
                self._deepface = _df.DeepFace
                logger.info("DeepFace gender filter: ENABLED")
            except ImportError:
                logger.warning(
                    "deepface not installed — gender filter disabled.\n"
                    "Install with:  pip install deepface"
                )
                self.enabled = False

    def is_female(self, bgr: np.ndarray) -> bool:
        if not self.enabled or self._deepface is None:
            return True   # include everything if filter is off
        try:
            result = self._deepface.analyze(
                bgr,
                actions=["gender"],
                enforce_detection=False,
                silent=True,
            )
            gender = result[0]["dominant_gender"]
            return gender == "Woman"
        except Exception:
            return True   # if analysis fails, include the image


# ─── Renderer settings (fixed alphas) ────────────────────────

RENDERER_KWARGS = dict(
    blur_kernel  = 25,    # was 15 — softer edges
    use_seamless = True,
)

# Override default plan alphas to fix white-face artefact
PLAN_OVERRIDES = dict(
    lip_alpha        = 0.45,   # was 0.55
    blush_alpha      = 0.25,   # was 0.40
    contour_alpha    = 0.15,   # was 0.35 ← main fix for white face
    highlight_alpha  = 0.10,   # was 0.30
    eyeshadow_alpha  = 0.35,   # was 0.50
)


# ─── Preview grid ────────────────────────────────────────────

def save_preview(pairs: list[tuple], out_path: Path, n_cols: int = 4):
    """Save a grid of (original, makeup) pairs for visual quality check."""
    cell_h, cell_w = 128, 128
    n = min(len(pairs), n_cols * 4)   # at most 4 rows
    rows = (n + n_cols - 1) // n_cols
    grid = np.zeros((rows * cell_h, n_cols * cell_w * 2, 3), dtype=np.uint8)

    for i, (x_bgr, y_bgr) in enumerate(pairs[:n]):
        row, col = divmod(i, n_cols)
        x_sm = cv2.resize(x_bgr, (cell_w, cell_h))
        y_sm = cv2.resize(y_bgr, (cell_w, cell_h))
        c = col * cell_w * 2
        r = row * cell_h
        grid[r:r+cell_h, c:c+cell_w]         = x_sm
        grid[r:r+cell_h, c+cell_w:c+cell_w*2] = y_sm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


# ─── Main generator ──────────────────────────────────────────

class DatasetGenerator:
    def __init__(
        self,
        source_dir:      str | Path,
        output_dir:      str | Path,
        image_size:      int  = 256,
        target_pairs:    int  = 5000,
        min_confidence:  float = 0.8,
        filter_gender:   bool = True,
        preview_every:   int  = 200,
    ):
        self.source_dir    = Path(source_dir)
        self.output_dir    = Path(output_dir)
        self.image_size    = image_size
        self.target_pairs  = target_pairs
        self.preview_every = preview_every

        self.parser        = FaceParser(min_confidence=min_confidence)
        self.expert        = ExpertSystem()
        self.renderer      = MakeupRenderer(**RENDERER_KWARGS)
        self.gender_filter = GenderFilter(enabled=filter_gender)

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    def generate(self) -> dict:
        image_paths = sorted(
            list(self.source_dir.rglob("*.png")) +
            list(self.source_dir.rglob("*.jpg")) +
            list(self.source_dir.rglob("*.jpeg"))
        )

        metadata       = {}
        success        = 0
        errors         = 0
        skipped_gender = 0
        recent_pairs   = []   # for preview grid

        pbar = tqdm(image_paths, desc="Generating pairs", unit="img")

        for img_path in pbar:
            if success >= self.target_pairs:
                break

            try:
                result = self._process_one(img_path, success + 1)

                if result is None:
                    errors += 1
                elif result == "gender_skip":
                    skipped_gender += 1
                else:
                    meta, x_bgr, y_bgr = result
                    stem = str(success + 1).zfill(5)
                    metadata[stem] = meta
                    recent_pairs.append((x_bgr, y_bgr))
                    success += 1

                    pbar.set_postfix(
                        ok=success,
                        err=errors,
                        skip_m=skipped_gender,
                    )

                    # Save preview grid
                    if success % self.preview_every == 0:
                        preview_path = (self.output_dir /
                                        f"preview_{success:05d}.jpg")
                        save_preview(recent_pairs[-self.preview_every:],
                                     preview_path)
                        logger.info(f"Preview saved → {preview_path}")

            except Exception as exc:
                errors += 1
                logger.debug(f"Failed {img_path.name}: {exc}")

        # Save metadata
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        stats = {
            "success":        success,
            "errors":         errors,
            "skipped_gender": skipped_gender,
            "total_seen":     success + errors + skipped_gender,
        }
        logger.info(f"Generation complete: {stats}")
        return stats

    # ─────────────────────────────────────────────────────────
    def _process_one(self, img_path: Path, idx: int):
        """
        Returns:
          (meta_dict, x_bgr, y_bgr)  on success
          "gender_skip"               if face is male
          None                        if face not detected / error
        """
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return None

        sz  = self.image_size
        bgr = cv2.resize(bgr, (sz, sz), interpolation=cv2.INTER_AREA)

        # ── Gender filter ─────────────────────────────────
        if not self.gender_filter.is_female(bgr):
            return "gender_skip"

        # ── Face parse ────────────────────────────────────
        parse = self.parser.parse(bgr)
        if parse is None:
            return None

        # ── Expert system ─────────────────────────────────
        analysis = self.expert.analyze(parse.skin_rgb, parse.face_shape)
        plan     = analysis["plan"]

        # Apply alpha overrides (fix white-face artefact)
        for attr, val in PLAN_OVERRIDES.items():
            setattr(plan, attr, val)

        # ── Render makeup ─────────────────────────────────
        bgr_y = self.renderer.render(bgr, parse, plan)

        # ── Save files ────────────────────────────────────
        stem = str(idx).zfill(5)
        img_dir = self.output_dir / "images"

        cv2.imwrite(str(img_dir / f"{stem}_X.png"), bgr)
        cv2.imwrite(str(img_dir / f"{stem}_Y.png"), bgr_y)

        lip_mask  = parse.masks.get(
            "lips_outer", np.zeros((sz, sz), np.uint8))
        skin_mask = parse.masks.get(
            "face_oval",  np.zeros((sz, sz), np.uint8))
        cv2.imwrite(str(img_dir / f"{stem}_lip_mask.png"),  lip_mask)
        cv2.imwrite(str(img_dir / f"{stem}_skin_mask.png"), skin_mask)

        meta = {
            "source_file": img_path.name,
            "face_shape":  analysis["face_shape"],
            "undertone":   analysis["undertone"],
            "lightness":   analysis["lightness"],
            "skin_rgb":    parse.skin_rgb.tolist(),
        }
        return meta, bgr, bgr_y

    def close(self):
        self.parser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─── CLI ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic makeup dataset (v2)"
    )
    p.add_argument("--source",         default="ffhq-dataset/flat_images")
    p.add_argument("--output",         default="ffhq-dataset/synthetic")
    p.add_argument("--pairs",          type=int,   default=5000)
    p.add_argument("--size",           type=int,   default=256)
    p.add_argument("--confidence",     type=float, default=0.8)
    p.add_argument("--filter-gender",  action="store_true", default=True,
                   help="Skip male faces (requires deepface)")
    p.add_argument("--no-filter",      action="store_true",
                   help="Disable gender filter (faster)")
    p.add_argument("--no-seamless",    action="store_true",
                   help="Skip Poisson blending (faster, lower quality)")
    p.add_argument("--preview-every",  type=int, default=200,
                   help="Save preview grid every N pairs")
    return p.parse_args()


def main():
    args = parse_args()

    filter_gender = args.filter_gender and not args.no_filter
    if args.no_seamless:
        RENDERER_KWARGS["use_seamless"] = False

    print(f"\n{'='*58}")
    print(f"  Aesthetics-Aware Makeup Transfer — Data Generation v2")
    print(f"{'='*58}")
    print(f"  Source        : {args.source}")
    print(f"  Output        : {args.output}")
    print(f"  Pairs         : {args.pairs}")
    print(f"  Size          : {args.size}×{args.size}")
    print(f"  Gender filter : {filter_gender}")
    print(f"  Seamless clone: {RENDERER_KWARGS['use_seamless']}")
    print(f"  Alphas        : lip={PLAN_OVERRIDES['lip_alpha']}  "
          f"contour={PLAN_OVERRIDES['contour_alpha']}  "
          f"blush={PLAN_OVERRIDES['blush_alpha']}")
    print(f"{'='*58}\n")

    with DatasetGenerator(
        source_dir     = args.source,
        output_dir     = args.output,
        image_size     = args.size,
        target_pairs   = args.pairs,
        min_confidence = args.confidence,
        filter_gender  = filter_gender,
        preview_every  = args.preview_every,
    ) as gen:
        stats = gen.generate()

    print(f"\n✓  Done!")
    print(f"   Generated : {stats['success']} pairs")
    print(f"   Errors    : {stats['errors']}")
    print(f"   Skipped ♂ : {stats['skipped_gender']}")
    print(f"   Total seen: {stats['total_seen']}\n")


if __name__ == "__main__":
    main()
