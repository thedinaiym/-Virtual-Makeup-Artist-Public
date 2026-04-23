#!/usr/bin/env python3
"""
scripts/inference.py
─────────────────────
Apply trained makeup transfer to a single image.

Usage:
    python scripts/inference.py \
        --checkpoint checkpoints/ckpt_epoch_0150.pt \
        --input      my_photo.jpg \
        --output     result.png \
        --intensity  0.8          # 0.0 = no makeup, 1.0 = full
"""

import argparse
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.face_parser  import FaceParser
from src.models.generator      import MaskConditionedUNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def load_model(ckpt_path: str, device: torch.device) -> MaskConditionedUNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    G = MaskConditionedUNet()
    G.load_state_dict(ckpt["G_state"])
    G.eval().to(device)
    return G


def preprocess(bgr: np.ndarray, parse_result, image_size: int = 256):
    """BGR image + masks → 5-channel tensor (1, 5, H, W)."""
    bgr_r = cv2.resize(bgr, (image_size, image_size))
    rgb   = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tx    = torch.from_numpy(rgb.transpose(2, 0, 1)) * 2 - 1   # (3,H,W)

    lip_mask  = cv2.resize(
        parse_result.masks.get("lips_outer", np.zeros((bgr.shape[:2]), np.uint8)),
        (image_size, image_size)
    )
    skin_mask = cv2.resize(
        parse_result.masks.get("face_oval", np.zeros((bgr.shape[:2]), np.uint8)),
        (image_size, image_size)
    )
    tlm = torch.from_numpy(lip_mask.astype(np.float32)  / 255.0).unsqueeze(0) * 2 - 1
    tsm = torch.from_numpy(skin_mask.astype(np.float32) / 255.0).unsqueeze(0) * 2 - 1

    return torch.cat([tx, tlm, tsm], dim=0).unsqueeze(0)   # (1, 5, H, W)


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) ∈ [-1,1] → BGR uint8."""
    arr = ((tensor.squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    rgb = arr.transpose(1, 2, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     default="result.png")
    p.add_argument("--intensity",  type=float, default=1.0,
                   help="Makeup intensity [0.0 – 1.0]")
    p.add_argument("--size",       type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load image ───────────────────────────────────────────
    bgr = cv2.imread(args.input)
    if bgr is None:
        logger.error(f"Could not read image: {args.input}")
        sys.exit(1)
    original_size = (bgr.shape[1], bgr.shape[0])

    # ── Parse face ───────────────────────────────────────────
    with FaceParser() as parser:
        parse = parser.parse(cv2.resize(bgr, (args.size, args.size)))

    if parse is None:
        logger.error("No face detected in the image.")
        sys.exit(1)

    # ── Model ────────────────────────────────────────────────
    G = load_model(args.checkpoint, device)
    x = preprocess(bgr, parse, args.size).to(device)
    intensity = torch.tensor([args.intensity], device=device)

    with torch.no_grad():
        out = G(x, intensity)

    result_bgr = postprocess(out)

    # Resize back to original dimensions if needed
    if args.size != original_size[0] or args.size != original_size[1]:
        result_bgr = cv2.resize(result_bgr, original_size)

    cv2.imwrite(args.output, result_bgr)
    logger.info(f"✓  Saved result → {args.output}")

    # Side-by-side comparison
    compare_path = Path(args.output).with_stem(
        Path(args.output).stem + "_compare"
    )
    orig_resized = cv2.resize(bgr, (args.size, args.size))
    comparison   = np.hstack([orig_resized, result_bgr
                               if result_bgr.shape == orig_resized.shape
                               else cv2.resize(result_bgr, (args.size, args.size))])
    cv2.imwrite(str(compare_path), comparison)
    logger.info(f"✓  Comparison  → {compare_path}")


if __name__ == "__main__":
    main()
