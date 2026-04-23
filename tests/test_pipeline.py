"""
tests/test_pipeline.py
───────────────────────
Quick sanity checks — no real data needed; uses synthetic numpy arrays.
Run with:  python -m pytest tests/ -v
"""

import numpy as np
import cv2
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Expert System ───────────────────────────────────────────

from src.pipeline.expert_system import (
    ExpertSystem, detect_undertone, skin_lightness
)

class TestExpertSystem:

    def test_undertone_warm(self):
        skin = np.array([220.0, 170.0, 140.0])   # red >> blue
        assert detect_undertone(skin) == "warm"

    def test_undertone_cool(self):
        skin = np.array([180.0, 160.0, 195.0])   # blue ≥ red
        assert detect_undertone(skin) == "cool"

    def test_undertone_neutral(self):
        skin = np.array([200.0, 170.0, 195.0])   # small diff
        assert detect_undertone(skin) == "neutral"

    def test_lightness_bins(self):
        assert skin_lightness(np.array([230., 220., 215.])) == "light"
        assert skin_lightness(np.array([140., 130., 120.])) == "medium"
        assert skin_lightness(np.array([70.,  60.,  50.])) == "dark"

    def test_get_plan_returns_makeup_plan(self):
        es   = ExpertSystem()
        skin = np.array([200., 160., 130.])
        plan = es.get_plan(skin, "oval")
        assert hasattr(plan, "lip_color")
        assert hasattr(plan, "contour_zones")
        assert isinstance(plan.lip_color, tuple)
        assert len(plan.lip_color) == 3

    @pytest.mark.parametrize("shape", ["oval","round","square","heart","oblong"])
    def test_all_face_shapes(self, shape):
        es   = ExpertSystem()
        skin = np.array([180., 140., 120.])
        plan = es.get_plan(skin, shape)
        assert plan is not None


# ─── Generator ───────────────────────────────────────────────

from src.models.generator import MaskConditionedUNet, StyleEncoder

class TestGenerator:

    def test_style_encoder_output_shape(self):
        enc   = StyleEncoder(style_dim=128)
        inten = torch.rand(4)
        style = enc(inten)
        assert style.shape == (4, 128)

    def test_generator_forward_shape(self):
        G = MaskConditionedUNet(
            in_channels=5, out_channels=3,
            base_features=16, num_downs=4, style_dim=64
        )
        G.eval()
        x     = torch.randn(2, 5, 64, 64)
        inten = torch.ones(2)
        with torch.no_grad():
            out = G(x, inten)
        assert out.shape == (2, 3, 64, 64), f"Got {out.shape}"

    def test_generator_output_range(self):
        G = MaskConditionedUNet(base_features=16, num_downs=4, style_dim=64)
        G.eval()
        x = torch.randn(1, 5, 64, 64)
        with torch.no_grad():
            out = G(x)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <=  1.0 + 1e-5

    def test_intensity_changes_output(self):
        G = MaskConditionedUNet(base_features=16, num_downs=4, style_dim=64)
        G.eval()
        x = torch.randn(1, 5, 64, 64)
        with torch.no_grad():
            out_full = G(x, torch.tensor([1.0]))
            out_low  = G(x, torch.tensor([0.1]))
        # Outputs should differ when intensity changes
        assert not torch.allclose(out_full, out_low, atol=1e-4)


# ─── Discriminator ───────────────────────────────────────────

from src.models.discriminator import PatchGANDiscriminator

class TestDiscriminator:

    def test_discriminator_output_shape(self):
        D = PatchGANDiscriminator(in_channels=5, out_channels=3,
                                   base_features=16, n_layers=2)
        D.eval()
        cond  = torch.randn(2, 5, 64, 64)
        image = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = D(cond, image)
        # Output should be (B, 1, H', W') — a patch map
        assert out.shape[0] == 2
        assert out.shape[1] == 1


# ─── Loss Functions ──────────────────────────────────────────

from src.models.losses import GANLoss, MakeupTransferLoss

class TestLosses:

    def test_gan_loss_real_higher_than_fake(self):
        loss_fn = GANLoss(mode="lsgan")
        pred = torch.ones(4, 1, 8, 8)
        l_real = loss_fn(pred, target_is_real=True)
        l_fake = loss_fn(pred, target_is_real=False)
        # real label=1.0 vs pred=1.0 → loss≈0; fake label=0.0 → loss high
        assert l_real.item() < l_fake.item()

    def test_makeup_loss_returns_dict(self):
        device  = torch.device("cpu")
        loss_fn = MakeupTransferLoss(device=device)
        fake_pred  = torch.zeros(2, 1, 8, 8)
        fake_image = torch.randn(2, 3, 32, 32)
        real_image = torch.randn(2, 3, 32, 32)
        losses = loss_fn.generator_loss(fake_pred, fake_image, real_image)
        for key in ("total", "gan", "l1", "perceptual"):
            assert key in losses
            assert losses[key].item() >= 0


# ─── Dataset ─────────────────────────────────────────────────

from src.training.dataset import MakeupPairDataset
import tempfile
import os

class TestDataset:

    @pytest.fixture
    def fake_data_dir(self, tmp_path):
        """Create a minimal fake dataset directory."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        sz = 64
        for i in range(5):
            stem = str(i + 1).zfill(5)
            for suffix in ["_X.png", "_Y.png"]:
                img = (np.random.rand(sz, sz, 3) * 255).astype(np.uint8)
                cv2.imwrite(str(img_dir / f"{stem}{suffix}"), img)
            for suffix in ["_lip_mask.png", "_skin_mask.png"]:
                mask = (np.random.rand(sz, sz) * 255).astype(np.uint8)
                cv2.imwrite(str(img_dir / f"{stem}{suffix}"), mask)
        return tmp_path

    def test_dataset_length(self, fake_data_dir):
        ds = MakeupPairDataset(fake_data_dir, split="train",
                                image_size=32, val_frac=0.2)
        assert len(ds) >= 1

    def test_dataset_item_shapes(self, fake_data_dir):
        ds   = MakeupPairDataset(fake_data_dir, split="train",
                                  image_size=32, val_frac=0.2, augment=False)
        item = ds[0]
        assert item["x"].shape == (5, 32, 32)
        assert item["y"].shape == (3, 32, 32)
        assert item["intensity"].item() == 1.0

    def test_input_range(self, fake_data_dir):
        ds   = MakeupPairDataset(fake_data_dir, split="train",
                                  image_size=32, val_frac=0.2, augment=False)
        item = ds[0]
        assert item["x"].min() >= -1.0 - 1e-5
        assert item["x"].max() <=  1.0 + 1e-5
