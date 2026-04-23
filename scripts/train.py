#!/usr/bin/env python3
"""
scripts/train.py
─────────────────
Train the Mask-Conditioned cGAN.

Usage:
    python scripts/train.py \
        --config configs/config.yaml \
        --data   data/synthetic \
        [--resume checkpoints/ckpt_epoch_0050.pt] \
        [--no-wandb]
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.generator      import MaskConditionedUNet
from src.models.discriminator  import PatchGANDiscriminator
from src.models.losses         import MakeupTransferLoss
from src.training.dataset      import MakeupPairDataset
from src.training.trainer      import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train makeup transfer cGAN")
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--data",     default="data/synthetic")
    p.add_argument("--resume",   default=None, help="Path to checkpoint to resume from")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Config ───────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name()}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Datasets ─────────────────────────────────────────────
    tc = cfg["training"]
    mc = cfg["model"]

    train_ds = MakeupPairDataset(args.data, split="train",
                                  image_size=tc["image_size"])
    val_ds   = MakeupPairDataset(args.data, split="val",
                                  image_size=tc["image_size"], augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Models ───────────────────────────────────────────────
    G = MaskConditionedUNet(
        in_channels   = mc["in_channels"],
        out_channels  = mc["out_channels"],
        base_features = mc["base_features"],
        style_dim     = mc["style_dim"],
    )
    D = PatchGANDiscriminator(
        in_channels  = mc["in_channels"],
        out_channels = mc["out_channels"],
    )
    loss_fn = MakeupTransferLoss(
        lambda_gan        = tc["lambda_gan"],
        lambda_l1         = tc["lambda_l1"],
        lambda_perceptual = tc["lambda_perceptual"],
        device            = device,
    )

    n_params_G = sum(p.numel() for p in G.parameters()) / 1e6
    n_params_D = sum(p.numel() for p in D.parameters()) / 1e6
    logger.info(f"Generator:     {n_params_G:.2f}M parameters")
    logger.info(f"Discriminator: {n_params_D:.2f}M parameters")

    # ── Run dir ──────────────────────────────────────────────
    run_name = args.run_name or f"run_{torch.randint(1000, (1,)).item():04d}"
    run_dir  = Path(cfg["paths"]["runs"]) / run_name

    # ── Trainer ──────────────────────────────────────────────
    trainer = Trainer(
        generator     = G,
        discriminator = D,
        loss_fn       = loss_fn,
        train_loader  = train_loader,
        val_loader    = val_loader,
        cfg           = cfg,
        device        = device,
        run_dir       = run_dir,
        use_wandb     = not args.no_wandb,
        resume_ckpt   = args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
