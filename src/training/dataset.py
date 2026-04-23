"""
src/training/dataset.py
────────────────────────
PyTorch Dataset for synthetic paired makeup images.

Each sample:
  x_tensor   : (5, H, W) float32 — RGB + lip_mask + skin_mask, normalised [-1,1]
  y_tensor   : (3, H, W) float32 — makeup image (ground truth), normalised [-1,1]
  intensity  : scalar float32 1.0 (training always uses full makeup)
  skin_rgb   : (3,)  float32 in [0,1] — mean cheek RGB normalised
  face_shape : (5,)  float32 one-hot  — [oval, round, square, heart, oblong]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─── Face shape encoding ─────────────────────────────────────
FACE_SHAPES = ["oval", "round", "square", "heart", "oblong"]


def encode_face_shape(shape_name: str) -> torch.Tensor:
    """'oval' → tensor([1,0,0,0,0])"""
    idx = FACE_SHAPES.index(shape_name) if shape_name in FACE_SHAPES else 0
    one_hot = torch.zeros(len(FACE_SHAPES))
    one_hot[idx] = 1.0
    return one_hot


def encode_skin_rgb(rgb: list | np.ndarray) -> torch.Tensor:
    """[R,G,B] uint8-range → normalised float32 tensor (3,)"""
    arr = np.array(rgb, dtype=np.float32)
    return torch.from_numpy(arr / 255.0).clamp(0.0, 1.0)


class MakeupPairDataset(Dataset):
    """
    Loads (X, Y, lip_mask, skin_mask) pairs produced by DatasetGenerator.
    Also loads per-sample metadata (skin_rgb, face_shape) from metadata.json.

    Parameters
    ----------
    root       : path to data/synthetic/
    split      : 'train' | 'val'
    val_frac   : fraction of data reserved for validation
    image_size : resize target
    augment    : apply random horizontal flip + colour jitter
    """

    def __init__(
        self,
        root:       str | Path,
        split:      str = "train",
        val_frac:   float = 0.1,
        image_size: int = 256,
        augment:    bool = True,
    ):
        self.root       = Path(root)
        self.img_dir    = self.root / "images"
        self.image_size = image_size
        self.augment    = augment and (split == "train")

        # ── Load metadata ────────────────────────────────────
        meta_path = self.root / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

        # ── Find all pairs ───────────────────────────────────
        x_files = sorted(self.img_dir.glob("*_X.png"))
        indices  = [f.stem.replace("_X", "") for f in x_files]

        # Train / val split (deterministic)
        n_val = max(1, int(len(indices) * val_frac))
        if split == "train":
            self.indices = indices[n_val:]
        else:
            self.indices = indices[:n_val]

        # ── Augmentation ─────────────────────────────────────
        self._color_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03
        )

        # Default style fallbacks (used when metadata is missing)
        self._default_skin_rgb   = encode_skin_rgb([180, 150, 130])
        self._default_face_shape = encode_face_shape("oval")

    # ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.indices)

    # ─────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        stem = self.indices[idx]
        p    = self.img_dir

        # ── Load images ──────────────────────────────────────
        bgr_x     = cv2.imread(str(p / f"{stem}_X.png"))
        bgr_y     = cv2.imread(str(p / f"{stem}_Y.png"))
        lip_mask  = cv2.imread(str(p / f"{stem}_lip_mask.png"),  cv2.IMREAD_GRAYSCALE)
        skin_mask = cv2.imread(str(p / f"{stem}_skin_mask.png"), cv2.IMREAD_GRAYSCALE)

        # Resize
        sz = self.image_size
        bgr_x     = cv2.resize(bgr_x,     (sz, sz), interpolation=cv2.INTER_AREA)
        bgr_y     = cv2.resize(bgr_y,     (sz, sz), interpolation=cv2.INTER_AREA)
        lip_mask  = cv2.resize(lip_mask,  (sz, sz), interpolation=cv2.INTER_NEAREST)
        skin_mask = cv2.resize(skin_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

        # BGR → RGB float [0, 1]
        rgb_x = cv2.cvtColor(bgr_x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_y = cv2.cvtColor(bgr_y, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Masks → float [0, 1]
        lm = lip_mask.astype(np.float32)  / 255.0
        sm = skin_mask.astype(np.float32) / 255.0

        # ── Geometry augmentation (same flip for all) ────────
        if self.augment and np.random.rand() > 0.5:
            rgb_x = np.fliplr(rgb_x).copy()
            rgb_y = np.fliplr(rgb_y).copy()
            lm    = np.fliplr(lm).copy()
            sm    = np.fliplr(sm).copy()

        # ── To tensors, normalise to [-1, 1] ─────────────────
        tx  = torch.from_numpy(rgb_x.transpose(2, 0, 1)) * 2 - 1   # (3,H,W)
        ty  = torch.from_numpy(rgb_y.transpose(2, 0, 1)) * 2 - 1   # (3,H,W)
        tlm = torch.from_numpy(lm).unsqueeze(0) * 2 - 1             # (1,H,W)
        tsm = torch.from_numpy(sm).unsqueeze(0) * 2 - 1             # (1,H,W)

        # ── Colour jitter on X only ───────────────────────────
        if self.augment:
            tx_pil = transforms.ToPILImage()(((tx + 1) / 2).clamp(0, 1))
            tx_pil = self._color_jitter(tx_pil)
            tx     = transforms.ToTensor()(tx_pil) * 2 - 1

        # ── 5-channel input tensor ────────────────────────────
        x_input = torch.cat([tx, tlm, tsm], dim=0)   # (5, H, W)

        # ── Metadata: skin_rgb + face_shape ──────────────────
        # metadata.json uses zero-padded numeric keys: "00001", "00002" ...
        meta = self._meta.get(stem, {})

        skin_rgb_raw = meta.get("skin_rgb", None)
        if skin_rgb_raw is not None:
            skin_rgb = encode_skin_rgb(skin_rgb_raw)
        else:
            skin_rgb = self._default_skin_rgb.clone()

        shape_name = meta.get("face_shape", "oval")
        face_shape = encode_face_shape(shape_name)

        return {
            "x":          x_input,                  # (5, H, W)
            "y":          ty,                        # (3, H, W)
            "intensity":  torch.tensor(1.0),         # scalar
            "skin_rgb":   skin_rgb,                  # (3,)
            "face_shape": face_shape,                # (5,)
            "stem":       stem,
        }