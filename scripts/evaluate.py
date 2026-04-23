#!/usr/bin/env python3
"""
scripts/evaluate.py
────────────────────
Full evaluation suite: FID, L1, SSIM, LPIPS, inference time.

Usage:
    python scripts/evaluate.py \
        --checkpoint runs/run_0001/ckpt_epoch_0150.pt \
        --data       data/synthetic \
        --output     runs/run_0001/eval_results.json
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.generator    import MaskConditionedUNet
from src.training.dataset    import MakeupPairDataset
from src.evaluation.metrics  import evaluate, LPIPSMetric, compute_fid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate makeup transfer model")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       default="data/synthetic")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--output",     default="eval_results.json")
    p.add_argument("--fid",        action="store_true", help="Compute FID (slow)")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def save_generated_images(generator, loader, device, out_dir: Path, image_size: int):
    """Save generated images to disk for FID computation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = out_dir / "real"
    fake_dir = out_dir / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)

    generator.eval()
    idx = 0
    with torch.no_grad():
        for batch in loader:
            x    = batch["x"].to(device)
            y    = batch["y"].to(device)
            inten= batch["intensity"].to(device)
            fake = generator(x, inten)

            for i in range(y.size(0)):
                def t2bgr(t):
                    arr = ((t.clamp(-1,1)+1)/2*255).byte().cpu().numpy()
                    return cv2.cvtColor(arr.transpose(1,2,0), cv2.COLOR_RGB2BGR)

                cv2.imwrite(str(real_dir / f"{idx:05d}.png"), t2bgr(y[i]))
                cv2.imwrite(str(fake_dir / f"{idx:05d}.png"), t2bgr(fake[i]))
                idx += 1
    return real_dir, fake_dir


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────
    mc   = cfg["model"]
    ckpt = torch.load(args.checkpoint, map_location=device)
    G = MaskConditionedUNet(
        in_channels   = mc["in_channels"],
        out_channels  = mc["out_channels"],
        base_features = mc["base_features"],
        style_dim     = mc["style_dim"],
    )
    G.load_state_dict(ckpt["G_state"])
    G.eval().to(device)
    trained_epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded checkpoint (epoch {trained_epoch})")

    # ── Val loader ───────────────────────────────────────────
    image_size = cfg["training"]["image_size"]
    val_ds = MakeupPairDataset(
        args.data, split="val", image_size=image_size, augment=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    logger.info(f"Validation set: {len(val_ds)} samples")

    # ── Core metrics ─────────────────────────────────────────
    lpips_metric = LPIPSMetric(device)
    logger.info("Computing L1 / SSIM / LPIPS / inference time …")
    results = evaluate(G, val_loader, device, lpips_metric, image_size)

    # ── FID (optional — requires saving all images) ───────────
    if args.fid:
        logger.info("Saving images for FID computation …")
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir, fake_dir = save_generated_images(
                G, val_loader, device, Path(tmpdir), image_size
            )
            logger.info("Computing FID …")
            fid_score = compute_fid(real_dir, fake_dir, device=device.type)
        results["fid"] = fid_score
        logger.info(f"FID: {fid_score:.2f}")

    results["checkpoint"] = args.checkpoint
    results["epoch"]      = trained_epoch

    # ── Print & save ─────────────────────────────────────────
    print("\n" + "="*45)
    print("  Evaluation Results")
    print("="*45)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<18} {v:.4f}")
        else:
            print(f"  {k:<18} {v}")
    print("="*45 + "\n")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
