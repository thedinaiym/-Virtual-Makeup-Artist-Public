"""
src/models/discriminator.py
────────────────────────────
PatchGAN Discriminator — classifies 70×70 local patches as real/fake.

Input : concatenation of (condition, image) → (B, 5+3, H, W)
          • condition = masked input X  (5 channels)
          • image     = real Y  or  generated Ŷ  (3 channels)
Output: (B, 1, H', W') — patch-level logits (no sigmoid; use BCEWithLogitsLoss)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    Standard PatchGAN from Pix2Pix, conditioned on the input X.

    Parameters
    ----------
    in_channels   : input X channels (5)
    out_channels  : image channels   (3)
    base_features : 64
    n_layers      : 3 → ~70×70 receptive field at 256 input
    """

    def __init__(
        self,
        in_channels:   int = 5,
        out_channels:  int = 3,
        base_features: int = 64,
        n_layers:      int = 3,
    ):
        super().__init__()

        # Total input = condition + image
        total_in = in_channels + out_channels

        def _block(ich, och, stride, use_bn=True):
            layers = [nn.Conv2d(ich, och, 4, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(och))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # First layer: no BN
        sequence = _block(total_in, base_features, stride=2, use_bn=False)
        nf = base_features
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += _block(nf_prev, nf, stride=2)

        # One more conv at stride=1 before output
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += _block(nf_prev, nf, stride=1)

        # Output: 1-channel patch map
        sequence.append(nn.Conv2d(nf, 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*sequence)
        self._init_weights()

    def forward(
        self,
        condition: torch.Tensor,   # (B, 5, H, W)
        image:     torch.Tensor,   # (B, 3, H, W)
    ) -> torch.Tensor:
        x = torch.cat([condition, image], dim=1)
        return self.model(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)
