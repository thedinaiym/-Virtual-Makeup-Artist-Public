"""
src/models/losses.py
─────────────────────
Composite loss:
  L_total = λ1·L_cGAN + λ2·L_L1 + λ3·L_Perceptual

Perceptual loss uses VGG-16 features (relu2_2, relu3_3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─── Perceptual (VGG) Loss ───────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG-16 intermediate features.
    Extracts relu2_2 (layer 9) and relu3_3 (layer 16).
    """

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Freeze VGG weights
        for p in vgg.parameters():
            p.requires_grad_(False)

        features = vgg.features
        self.slice1 = features[:9 ].to(device).eval()   # relu2_2
        self.slice2 = features[9:16].to(device).eval()  # relu3_3

        # ImageNet normalisation
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406],
                                  device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225],
                                  device=device).view(1, 3, 1, 1)
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1] → [0, 1] → ImageNet normalised
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self._normalise(pred)
        t = self._normalise(target)

        f1_p = self.slice1(p)
        f1_t = self.slice1(t)
        f2_p = self.slice2(f1_p)
        f2_t = self.slice2(f1_t)

        loss = F.l1_loss(f1_p, f1_t) + F.l1_loss(f2_p, f2_t)
        return loss


# ─── GAN Losses ─────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    Wrapper supporting vanilla GAN (BCEWithLogits) or LSGAN (MSE).

    Usage:
        loss_fn = GANLoss(mode='lsgan')
        d_real  = loss_fn(disc(real),  target_is_real=True)
        d_fake  = loss_fn(disc(fake),  target_is_real=False)
        g_loss  = loss_fn(disc(fake),  target_is_real=True)
    """

    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        self.mode = mode
        if mode == "vanilla":
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown GAN mode: {mode}")

    def _make_label(
        self, pred: torch.Tensor, is_real: bool
    ) -> torch.Tensor:
        val = 1.0 if is_real else 0.0
        return torch.full_like(pred, val)

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        label = self._make_label(pred, target_is_real)
        return self.criterion(pred, label)


# ─── Combined Loss ───────────────────────────────────────────

class MakeupTransferLoss(nn.Module):
    """
    Aggregates all losses with configurable weights.

    Parameters
    ----------
    lambda_gan, lambda_l1, lambda_perceptual : loss weights
    device : for VGG
    gan_mode : 'lsgan' (default) or 'vanilla'
    """

    def __init__(
        self,
        lambda_gan:         float = 1.0,
        lambda_l1:          float = 100.0,
        lambda_perceptual:  float = 10.0,
        device:             torch.device = torch.device("cpu"),
        gan_mode:           str = "lsgan",
    ):
        super().__init__()
        self.lambda_gan        = lambda_gan
        self.lambda_l1         = lambda_l1
        self.lambda_perceptual = lambda_perceptual

        self.gan_loss        = GANLoss(mode=gan_mode)
        self.l1_loss         = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss(device)

    def generator_loss(
        self,
        fake_pred:  torch.Tensor,   # D(X, G(X))
        fake_image: torch.Tensor,   # G(X)  i.e. predicted
        real_image: torch.Tensor,   # Y     i.e. ground truth
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with 'total', 'gan', 'l1', 'perceptual'.
        """
        l_gan  = self.gan_loss(fake_pred, target_is_real=True)
        l_l1   = self.l1_loss(fake_image, real_image)
        l_perc = self.perceptual_loss(fake_image, real_image)

        total = (self.lambda_gan        * l_gan
               + self.lambda_l1         * l_l1
               + self.lambda_perceptual * l_perc)

        return {"total": total, "gan": l_gan, "l1": l_l1, "perceptual": l_perc}

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,   # D(X, Y)
        fake_pred: torch.Tensor,   # D(X, G(X))  — detached
    ) -> torch.Tensor:
        l_real = self.gan_loss(real_pred, target_is_real=True)
        l_fake = self.gan_loss(fake_pred, target_is_real=False)
        return (l_real + l_fake) * 0.5
