"""
src/models/generator.py
────────────────────────
Mask-Conditioned U-Net with Adaptive Instance Normalisation (AdaIN).

Changes vs v1:
  • StyleEncoder v2: accepts intensity(1) + skin_rgb(3) + face_shape(5) = 9 inputs
    This gives the network explicit knowledge of who it is applying makeup to.
  • Generator.forward() accepts optional skin_rgb and face_shape tensors.
  • AdaIN applied BEFORE skip-connection cat (fixes channel mismatch bug).

Architecture:
  Input  : (B, 5, H, W)  — RGB(3) + lip_mask(1) + skin_mask(1)
  Encoder: 6 downsampling blocks  (64 → 512 feature maps)
  Bottleneck: AdaIN injects style vector
  Decoder: 6 upsampling blocks with skip-connections + AdaIN
  Output : (B, 3, H, W) in [-1, 1]  (tanh)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─── AdaIN ───────────────────────────────────────────────────

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalisation.
    Projects style vector → (scale γ, shift β) per channel.
    """

    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm       = nn.InstanceNorm2d(num_features, affine=False)
        self.style_proj = nn.Linear(style_dim, num_features * 2)
        nn.init.ones_ (self.style_proj.weight)
        nn.init.zeros_(self.style_proj.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x     : (B, C, H, W)
        # style : (B, style_dim)
        B, C, H, W = x.shape
        params = self.style_proj(style)               # (B, 2C)
        gamma  = params[:, :C].view(B, C, 1, 1)
        beta   = params[:, C:].view(B, C, 1, 1)
        return gamma * self.norm(x) + beta


# ─── Encoder block ───────────────────────────────────────────

class DownBlock(nn.Module):
    """Conv(stride=2) → BN → LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─── Decoder block ───────────────────────────────────────────

class UpBlock(nn.Module):
    """
    ConvTranspose → AdaIN → ReLU → (Dropout) → cat(skip)

    AdaIN is applied BEFORE concatenation with the skip connection so
    that channel counts always match the initialised AdaIN(out_ch).
    Style conditions the generated features; skip carries original details.
    """

    def __init__(
        self,
        in_ch:     int,
        out_ch:    int,
        style_dim: int,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm = AdaIN(out_ch, style_dim)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x:     torch.Tensor,   # decoder feature
        skip:  torch.Tensor,   # encoder skip connection
        style: torch.Tensor,   # (B, style_dim)
    ) -> torch.Tensor:
        x = self.up(x)                              # (B, out_ch, H, W)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        # AdaIN + activation BEFORE cat — channel count matches out_ch ✓
        x = self.drop(self.act(self.norm(x, style)))
        x = torch.cat([x, skip], dim=1)            # (B, out_ch + skip_ch, H, W)
        return x


# ─── StyleEncoder v2 ─────────────────────────────────────────

class StyleEncoder(nn.Module):
    """
    Transforms three conditioning signals into a style vector:

      intensity  (1)  : makeup strength [0.0 → 1.0]
      skin_rgb   (3)  : mean cheek RGB normalised to [0, 1]
      face_shape (5)  : one-hot [oval, round, square, heart, oblong]

    Total input dim = 9  →  style vector of size style_dim (128)

    At inference, skin_rgb and face_shape can be omitted — sensible
    defaults (neutral medium skin, oval face) are used automatically.
    """

    FACE_SHAPES = ["oval", "round", "square", "heart", "oblong"]

    def __init__(self, style_dim: int = 128):
        super().__init__()
        self.style_dim = style_dim
        # 1 (intensity) + 3 (skin RGB) + 5 (face shape) = 9
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(
        self,
        intensity:  torch.Tensor,                    # (B,) or (B,1)
        skin_rgb:   Optional[torch.Tensor] = None,   # (B,3) in [0,1]
        face_shape: Optional[torch.Tensor] = None,   # (B,5) one-hot
    ) -> torch.Tensor:
        B      = intensity.shape[0]
        device = intensity.device

        if intensity.dim() == 1:
            intensity = intensity.unsqueeze(-1)      # (B,1)

        # Default: neutral medium skin
        if skin_rgb is None:
            skin_rgb = torch.full((B, 3), 0.6, device=device)

        # Default: oval face
        if face_shape is None:
            face_shape = torch.zeros(B, 5, device=device)
            face_shape[:, 0] = 1.0

        x = torch.cat([intensity, skin_rgb, face_shape], dim=-1)  # (B,9)
        return self.net(x.float())

    # ── Convenience helpers for inference ────────────────────
    @staticmethod
    def encode_face_shape(shape_name: str) -> torch.Tensor:
        """'round' → tensor([0,1,0,0,0])"""
        shapes = ["oval", "round", "square", "heart", "oblong"]
        idx = shapes.index(shape_name) if shape_name in shapes else 0
        one_hot = torch.zeros(5)
        one_hot[idx] = 1.0
        return one_hot

    @staticmethod
    def encode_skin_rgb(rgb) -> torch.Tensor:
        """[R,G,B] uint8-range → normalised float32 (3,)"""
        import numpy as np
        arr = np.array(rgb, dtype=np.float32)
        return torch.from_numpy(arr / 255.0).clamp(0.0, 1.0)


# ─── Generator ───────────────────────────────────────────────

class MaskConditionedUNet(nn.Module):
    """
    Mask-Conditioned U-Net generator with AdaIN style control.

    Parameters
    ----------
    in_channels   : 5  (RGB + lip_mask + skin_mask)
    out_channels  : 3  (RGB output)
    base_features : 64
    num_downs     : 6  (256×256 → 4×4 bottleneck)
    style_dim     : 128
    """

    def __init__(
        self,
        in_channels:   int = 5,
        out_channels:  int = 3,
        base_features: int = 64,
        num_downs:     int = 6,
        style_dim:     int = 128,
    ):
        super().__init__()
        self.style_enc = StyleEncoder(style_dim)

        nf = base_features

        # ── Encoder ─────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        self.enc_blocks.append(DownBlock(in_channels, nf, use_bn=False))
        ch = nf
        for _ in range(1, num_downs):
            out_ch = min(ch * 2, 512)
            self.enc_blocks.append(DownBlock(ch, out_ch))
            ch = out_ch
        # ch == 512 at bottleneck

        # ── Bottleneck ───────────────────────────────────────
        self.bottleneck_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bottleneck_norm = AdaIN(ch, style_dim)
        self.bottleneck_act  = nn.ReLU(inplace=True)

        # ── Decoder ─────────────────────────────────────────
        self.dec_blocks   = nn.ModuleList()
        enc_channels      = self._enc_channels(in_channels, base_features, num_downs)
        dec_ch            = ch
        dropout_until     = 3   # first 3 decoder blocks use dropout

        for i in range(num_downs):
            skip_ch = enc_channels[-(i + 1)]
            out_ch  = max(skip_ch // 2, base_features) if i < num_downs - 1 else base_features
            drop    = 0.5 if i < dropout_until else 0.0
            self.dec_blocks.append(UpBlock(dec_ch, out_ch, style_dim, dropout=drop))
            dec_ch = out_ch + skip_ch   # after cat with skip

        # ── Output head ──────────────────────────────────────
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(dec_ch, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _enc_channels(in_ch, base_f, num_downs):
        channels = []
        ch = in_ch
        for i in range(num_downs):
            out = base_f if i == 0 else min(channels[-1] * 2, 512)
            channels.append(out)
            ch = out
        return channels

    # ─────────────────────────────────────────────────────────
    def forward(
        self,
        x:          torch.Tensor,                    # (B, 5, H, W)
        intensity:  Optional[torch.Tensor] = None,   # (B,)  in [0,1]
        skin_rgb:   Optional[torch.Tensor] = None,   # (B,3) in [0,1]
        face_shape: Optional[torch.Tensor] = None,   # (B,5) one-hot
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : 5-channel input (RGB + lip mask + skin mask)
        intensity  : makeup intensity [0=no makeup, 1=full makeup]
        skin_rgb   : normalised mean cheek RGB from face parser
        face_shape : one-hot face shape from face parser

        Returns
        -------
        (B, 3, H, W) in [-1, 1]
        """
        B = x.size(0)
        if intensity is None:
            intensity = torch.ones(B, device=x.device)

        # Build rich style vector from all conditioning signals
        style = self.style_enc(intensity, skin_rgb, face_shape)   # (B, style_dim)

        # ── Encoder ──────────────────────────────────────────
        skips = []
        feat  = x
        for block in self.enc_blocks:
            feat = block(feat)
            skips.append(feat)

        # ── Bottleneck ───────────────────────────────────────
        feat = self.bottleneck_act(
            self.bottleneck_norm(self.bottleneck_conv(feat), style)
        )

        # ── Decoder ──────────────────────────────────────────
        for i, block in enumerate(self.dec_blocks):
            skip = skips[-(i + 1)]
            feat = block(feat, skip, style)

        return self.out_conv(feat)