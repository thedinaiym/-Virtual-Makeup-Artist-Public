"""
src/training/trainer.py
────────────────────────
Full training loop for Mask-Conditioned cGAN.

Changes vs v1:
  • Passes skin_rgb + face_shape from dataset to generator (StyleEncoder v2)
  • Larger validation grid (shows input | pred | gt side by side)
  • Better logging: per-loss breakdown every log_interval steps
  • Gradient clipping for training stability
  • Best-checkpoint saving based on val_l1
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─── LR Scheduler ────────────────────────────────────────────

class LinearDecayLR:
    """Linearly decay LR from initial value to 0 over (total - start) epochs."""

    def __init__(self, optimizer, total_epochs: int, decay_start: int):
        self.opt         = optimizer
        self.total       = total_epochs
        self.decay_start = decay_start
        self.base_lrs    = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.decay_start:
            return
        frac = (epoch - self.decay_start) / max(1, self.total - self.decay_start)
        for base_lr, pg in zip(self.base_lrs, self.opt.param_groups):
            pg["lr"] = base_lr * max(0.0, 1.0 - frac)


# ─── Trainer ─────────────────────────────────────────────────

class Trainer:
    """
    Parameters
    ----------
    generator      : MaskConditionedUNet  (with StyleEncoder v2)
    discriminator  : PatchGANDiscriminator
    loss_fn        : MakeupTransferLoss
    train_loader   : DataLoader
    val_loader     : DataLoader
    cfg            : dict  (from yaml config)
    device         : torch.device
    run_dir        : directory for checkpoints + sample images
    use_wandb      : bool
    resume_ckpt    : optional path to resume from
    """

    def __init__(
        self,
        generator,
        discriminator,
        loss_fn,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        cfg:           dict,
        device:        torch.device,
        run_dir:       str | Path = "runs/exp",
        use_wandb:     bool = True,
        resume_ckpt:   Optional[str] = None,
    ):
        self.G       = generator.to(device)
        self.D       = discriminator.to(device)
        self.loss_fn = loss_fn

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg    = cfg
        self.device = device

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)

        tc = cfg["training"]
        self.epochs        = tc["epochs"]
        self.log_interval  = tc["log_interval"]
        self.save_interval = tc["save_interval"]
        self.val_samples   = tc.get("val_samples", 8)
        self.decay_epoch   = tc["decay_epoch"]

        # ── Optimisers ───────────────────────────────────────
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=tc["lr_g"], betas=(tc["beta1"], tc["beta2"])
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=tc["lr_d"], betas=(tc["beta1"], tc["beta2"])
        )
        self.sched_G = LinearDecayLR(self.opt_G, self.epochs, self.decay_epoch)
        self.sched_D = LinearDecayLR(self.opt_D, self.epochs, self.decay_epoch)

        self.start_epoch = 1
        self.best_val_l1 = float("inf")

        if resume_ckpt:
            self._load_checkpoint(resume_ckpt)

        # ── W&B ──────────────────────────────────────────────
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=cfg["project"]["wandb_project"],
                config=cfg,
                resume="allow",
            )
            wandb.watch(self.G, log="gradients", log_freq=200)

    # ─────────────────────────────────────────────────────────
    def train(self):
        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()

            train_logs = self._train_epoch(epoch)
            val_logs   = self._validate(epoch)

            self.sched_G.step(epoch)
            self.sched_D.step(epoch)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch:03d}/{self.epochs}] "
                f"G={train_logs['g_total']:.4f} "
                f"(gan={train_logs['g_gan']:.3f} "
                f"l1={train_logs['g_l1']:.3f} "
                f"perc={train_logs['g_perc']:.3f})  "
                f"D={train_logs['d_loss']:.4f}  "
                f"val_L1={val_logs.get('val_l1', 0):.4f}  "
                f"lr={self.opt_G.param_groups[0]['lr']:.2e}  "
                f"({elapsed:.0f}s)"
            )

            if self.use_wandb:
                wandb.log({
                    **train_logs,
                    **{k: v for k, v in val_logs.items() if not isinstance(v, wandb.Image)},
                    "epoch": epoch,
                    "lr_g":  self.opt_G.param_groups[0]["lr"],
                    "lr_d":  self.opt_D.param_groups[0]["lr"],
                })
                if "val_images" in val_logs:
                    wandb.log({"val_images": val_logs["val_images"], "epoch": epoch})

            # Save periodic checkpoint
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)

            # Save best checkpoint
            val_l1 = val_logs.get("val_l1", float("inf"))
            if val_l1 < self.best_val_l1:
                self.best_val_l1 = val_l1
                self._save_checkpoint(epoch, name="ckpt_best.pt")
                logger.info(f"  ★ New best val_L1={val_l1:.4f} — saved ckpt_best.pt")

        logger.info("Training complete.")

    # ─────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> dict:
        self.G.train()
        self.D.train()

        running = {k: 0.0 for k in
                   ("g_total", "g_gan", "g_l1", "g_perc", "d_loss")}
        n = 0

        for step, batch in enumerate(self.train_loader):
            x          = batch["x"].to(self.device)           # (B,5,H,W)
            y          = batch["y"].to(self.device)           # (B,3,H,W)
            intensity  = batch["intensity"].to(self.device)   # (B,)
            skin_rgb   = batch["skin_rgb"].to(self.device)    # (B,3)
            face_shape = batch["face_shape"].to(self.device)  # (B,5)

            # ── Discriminator update ──────────────────────
            self.opt_D.zero_grad()
            with torch.no_grad():
                fake_y = self.G(x, intensity,
                                skin_rgb=skin_rgb,
                                face_shape=face_shape)

            real_pred = self.D(x, y)
            fake_pred = self.D(x, fake_y.detach())
            d_loss    = self.loss_fn.discriminator_loss(real_pred, fake_pred)
            d_loss.backward()
            nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
            self.opt_D.step()

            # ── Generator update ──────────────────────────
            self.opt_G.zero_grad()
            fake_y    = self.G(x, intensity,
                               skin_rgb=skin_rgb,
                               face_shape=face_shape)
            fake_pred = self.D(x, fake_y)
            g_losses  = self.loss_fn.generator_loss(fake_pred, fake_y, y)
            g_losses["total"].backward()
            nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            self.opt_G.step()

            # Accumulate
            running["g_total"] += g_losses["total"].item()
            running["g_gan"]   += g_losses["gan"].item()
            running["g_l1"]    += g_losses["l1"].item()
            running["g_perc"]  += g_losses["perceptual"].item()
            running["d_loss"]  += d_loss.item()
            n += 1

            if step % self.log_interval == 0:
                logger.debug(
                    f"  [ep{epoch} step{step}] "
                    f"G={g_losses['total'].item():.4f}  "
                    f"D={d_loss.item():.4f}"
                )

        return {k: v / max(n, 1) for k, v in running.items()}

    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.G.eval()

        l1_total = 0.0
        n        = 0
        visuals  = []   # list of (input_rgb, pred, gt) tensors

        for batch in self.val_loader:
            x          = batch["x"].to(self.device)
            y          = batch["y"].to(self.device)
            intensity  = batch["intensity"].to(self.device)
            skin_rgb   = batch["skin_rgb"].to(self.device)
            face_shape = batch["face_shape"].to(self.device)

            fake = self.G(x, intensity,
                          skin_rgb=skin_rgb,
                          face_shape=face_shape)

            l1_total += F.l1_loss(fake, y).item()
            n += 1

            # Collect visuals (RGB slice of x, prediction, ground truth)
            if len(visuals) < self.val_samples:
                for i in range(min(x.size(0), self.val_samples - len(visuals))):
                    visuals.append((
                        x[i, :3],    # input RGB  (3,H,W)
                        fake[i],     # prediction  (3,H,W)
                        y[i],        # ground truth(3,H,W)
                    ))

        logs = {"val_l1": l1_total / max(n, 1)}

        # Save sample grid to disk
        if visuals:
            grid_rows = []
            for inp, pred, gt in visuals:
                row = torch.stack([inp, pred, gt], dim=0)   # (3,3,H,W)
                grid_rows.append(row)
            grid_tensor = torch.cat(grid_rows, dim=0)        # (N*3,3,H,W)
            grid = make_grid(
                (grid_tensor.clamp(-1, 1) + 1) / 2,
                nrow=3, padding=4, pad_value=1.0
            )
            sample_path = self.run_dir / "samples" / f"epoch_{epoch:04d}.png"
            save_image(grid, str(sample_path))

            if self.use_wandb:
                logs["val_images"] = wandb.Image(
                    grid,
                    caption="input RGB | prediction | ground truth"
                )

        return logs

    # ─────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, name: Optional[str] = None):
        fname = name or f"ckpt_epoch_{epoch:04d}.pt"
        path  = self.run_dir / fname
        torch.save({
            "epoch":       epoch,
            "G_state":     self.G.state_dict(),
            "D_state":     self.D.state_dict(),
            "opt_G_state": self.opt_G.state_dict(),
            "opt_D_state": self.opt_D.state_dict(),
            "best_val_l1": self.best_val_l1,
        }, path)
        logger.info(f"  Saved checkpoint → {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        self.opt_G.load_state_dict(ckpt["opt_G_state"])
        self.opt_D.load_state_dict(ckpt["opt_D_state"])
        self.start_epoch  = ckpt["epoch"] + 1
        self.best_val_l1  = ckpt.get("best_val_l1", float("inf"))
        logger.info(f"Resumed from epoch {ckpt['epoch']}  "
                    f"(best_val_l1={self.best_val_l1:.4f})")