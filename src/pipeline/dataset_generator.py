"""
src/pipeline/dataset_generator.py
───────────────────────────────────
Orchestrates the full synthetic-data pipeline:
  1. Iterate over FFHQ (or CelebA-HQ) source images
  2. Run FaceParser  → landmarks + masks
  3. Run ExpertSystem → MakeupPlan
  4. Run MakeupRenderer → styled image Y
  5. Save (X, Y) pairs + metadata JSON

Output structure:
  data/synthetic/
    images/
      0001_X.png
      0001_Y.png
      ...
    metadata.json    ← face_shape, undertone, lightness per pair
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .face_parser import FaceParser
from .expert_system import ExpertSystem
from .renderer import MakeupRenderer

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        image_size: int = 256,
        target_pairs: int = 5000,
        min_confidence: float = 0.8,
        use_seamless: bool = True,
    ):
        self.source_dir   = Path(source_dir)
        self.output_dir   = Path(output_dir)
        self.image_size   = image_size
        self.target_pairs = target_pairs

        self.parser   = FaceParser(min_confidence=min_confidence)
        self.expert   = ExpertSystem()
        self.renderer = MakeupRenderer(use_seamless=use_seamless)

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    def generate(self) -> dict:
        """
        Main entry point. Returns summary stats.
        """
        image_paths = sorted(self.source_dir.rglob("*.png")) + \
                      sorted(self.source_dir.rglob("*.jpg"))

        metadata = {}
        success = 0
        errors  = 0

        pbar = tqdm(image_paths, desc="Generating pairs", unit="img")
        for img_path in pbar:
            if success >= self.target_pairs:
                break

            try:
                meta = self._process_one(img_path, success + 1)
                if meta is not None:
                    metadata[str(success + 1).zfill(5)] = meta
                    success += 1
                    pbar.set_postfix(success=success, errors=errors)
            except Exception as exc:
                errors += 1
                logger.warning(f"Failed {img_path.name}: {exc}")

        # Save metadata
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        stats = {"success": success, "errors": errors, "total_attempted": success + errors}
        logger.info(f"Dataset generation complete: {stats}")
        return stats

    # ─────────────────────────────────────────────────────────
    def _process_one(
        self, img_path: Path, idx: int
    ) -> Optional[dict]:
        """Process a single source image. Returns metadata dict or None."""
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return None

        # Resize to target
        bgr = cv2.resize(bgr, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_AREA)

        # Parse face
        parse = self.parser.parse(bgr)
        if parse is None:
            return None

        # Expert analysis
        analysis = self.expert.analyze(parse.skin_rgb, parse.face_shape)
        plan = analysis["plan"]

        # Render makeup
        bgr_y = self.renderer.render(bgr, parse, plan)

        # Save pair
        stem = str(idx).zfill(5)
        cv2.imwrite(str(self.output_dir / "images" / f"{stem}_X.png"), bgr)
        cv2.imwrite(str(self.output_dir / "images" / f"{stem}_Y.png"), bgr_y)

        # Also save mask channels for training
        lip_mask  = parse.masks.get("lips_outer",
                    np.zeros((self.image_size, self.image_size), np.uint8))
        skin_mask = parse.masks.get("face_oval",
                    np.zeros_like(lip_mask))
        cv2.imwrite(str(self.output_dir / "images" / f"{stem}_lip_mask.png"),  lip_mask)
        cv2.imwrite(str(self.output_dir / "images" / f"{stem}_skin_mask.png"), skin_mask)

        return {
            "source_file": img_path.name,
            "face_shape":  analysis["face_shape"],
            "undertone":   analysis["undertone"],
            "lightness":   analysis["lightness"],
            "skin_rgb":    parse.skin_rgb.tolist(),
        }

    def close(self):
        self.parser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
