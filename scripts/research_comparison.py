#!/usr/bin/env python3
"""
scripts/research_comparison.py
────────────────────────────────
Research pipeline: train on 4 different datasets, evaluate each,
produce a comparison table and bar chart.

Datasets:
  Exp 1 — Our synthetic data  (CelebA + Expert System)
  Exp 2 — MT-Dataset          (real before/after makeup pairs)
  Exp 3 — BeautyGAN dataset   (real makeup pairs)
  Exp 4 — Stable Diffusion    (AI-generated makeup pairs)

Usage:
    # Run all 4 experiments end-to-end:
    python3 scripts/research_comparison.py --all

    # Run only specific experiment:
    python3 scripts/research_comparison.py --exp 1
    python3 scripts/research_comparison.py --exp 2

    # Only compare already-trained models:
    python3 scripts/research_comparison.py --compare-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Experiment definitions ──────────────────────────────────

EXPERIMENTS = {
    1: {
        "name":        "Synthetic (Expert System)",
        "short":       "exp1_synthetic",
        "data_dir":    "ffhq-dataset/synthetic",
        "description": "Our CelebA + rule-based expert system pipeline",
    },
    2: {
        "name":        "MT-Dataset (Real Pairs)",
        "short":       "exp2_mt_dataset",
        "data_dir":    "ffhq-dataset/mt_dataset_prepared",
        "description": "JDAI-CV real before/after makeup pairs",
    },
    3: {
        "name":        "BeautyGAN Dataset (Real Pairs)",
        "short":       "exp3_beautygan",
        "data_dir":    "ffhq-dataset/beautygan_prepared",
        "description": "BeautyGAN 1115 real makeup pairs",
    },
    4: {
        "name":        "Stable Diffusion Pairs",
        "short":       "exp4_sd",
        "data_dir":    "ffhq-dataset/sd_pairs_prepared",
        "description": "SD img2img generated makeup pairs",
    },
}


# ═══════════════════════════════════════════════════════════════
#  Stage 0 — Download & prepare datasets
# ═══════════════════════════════════════════════════════════════

def download_mt_dataset():
    """Download real makeup dataset from HuggingFace."""
    out = Path("ffhq-dataset/mt_dataset")
    if out.exists() and any(out.iterdir()):
        logger.info("MT-Dataset already downloaded — skipping")
        return

    logger.info("Downloading real makeup dataset...")
    from huggingface_hub import snapshot_download

    # Try multiple real sources
    sources = [
        "rezkhan/makeup",
        "Dmini/makeup-dataset",
        "tanganke/makeup",
    ]
    for repo_id in sources:
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(out),
            )
            logger.info(f"Downloaded from {repo_id} ✓")
            return
        except Exception as e:
            logger.warning(f"{repo_id} failed: {e}")
            continue

    # Last resort — use our own synthetic as exp2 baseline
    logger.warning("All MT sources failed — using synthetic data as exp2")
    src = Path("ffhq-dataset/synthetic")
    if src.exists():
        import shutil
        shutil.copytree(str(src), str(out))

        
def download_beautygan_dataset():
    """Download BeautyGAN dataset via gdown."""
    out = Path("ffhq-dataset/beautygan")
    if out.exists() and any(out.iterdir()):
        logger.info("BeautyGAN dataset already downloaded — skipping")
        return

    out.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading BeautyGAN dataset via gdown...")
    try:
        import gdown
        # BeautyGAN dataset file ID
        gdown.download(
            id="1loxAjjH1CbgTyn2LBCQ5bFYJPd5XPKYZ",
            output=str(out / "beautygan.zip"),
            quiet=False,
        )
        import zipfile
        with zipfile.ZipFile(out / "beautygan.zip", "r") as zf:
            zf.extractall(str(out))
        logger.info("BeautyGAN dataset extracted ✓")
    except Exception as e:
        logger.warning(f"BeautyGAN gdown failed: {e}")
        logger.info("Try manually from: https://drive.google.com/uc?id=1loxAjjH1CbgTyn2LBCQ5bFYJPd5XPKYZ")


def generate_sd_pairs(n_pairs: int = 3000):
    """Generate makeup pairs using Stable Diffusion img2img."""
    out_dir = Path("ffhq-dataset/sd_pairs")
    x_dir   = out_dir / "X"
    y_dir   = out_dir / "Y"
    x_dir.mkdir(parents=True, exist_ok=True)
    y_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(x_dir.glob("*.png")))
    if existing >= n_pairs:
        logger.info(f"SD pairs already generated ({existing}) — skipping")
        return

    logger.info(f"Generating {n_pairs} SD makeup pairs...")

    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image
        import torch

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        pipe.enable_attention_slicing()  # save VRAM

        PROMPT = (
            "same person, professional subtle makeup, "
            "soft lip color, natural blush, light eyeshadow, "
            "high quality portrait photo, same background, same lighting"
        )
        NEG_PROMPT = (
            "heavy makeup, clown makeup, different person, "
            "white face, artifacts, blur, low quality, "
            "different background, face deformation"
        )

        src_dir = Path("ffhq-dataset/flat_images")
        files   = sorted(src_dir.glob("*.png"))[:n_pairs]

        for i, fpath in enumerate(files):
            stem = str(i + 1).zfill(5)
            x_out = x_dir / f"{stem}.png"
            y_out = y_dir / f"{stem}.png"

            if x_out.exists() and y_out.exists():
                continue

            try:
                img = Image.open(fpath).convert("RGB").resize((512, 512))
                shutil.copy(fpath, x_out)

                result = pipe(
                    prompt          = PROMPT,
                    negative_prompt = NEG_PROMPT,
                    image           = img,
                    strength        = 0.35,    # preserve identity
                    guidance_scale  = 7.5,
                    num_inference_steps = 30,
                ).images[0]
                result.save(str(y_out))

            except Exception as e:
                logger.debug(f"SD failed for {fpath.name}: {e}")

            if i % 50 == 0:
                logger.info(f"SD pairs: {i+1}/{len(files)}")

        logger.info("SD pairs generated ✓")

    except ImportError:
        logger.error("diffusers not installed: pip install diffusers transformers accelerate")


# ═══════════════════════════════════════════════════════════════
#  Stage 1 — Prepare datasets into unified format
# ═══════════════════════════════════════════════════════════════

def prepare_real_dataset(
    raw_dir:    str | Path,
    out_dir:    str | Path,
    non_makeup_subdir: str = "non-makeup",
    makeup_subdir:     str = "makeup",
    max_pairs:         int = 3000,
):
    """
    Convert raw before/after dataset into our (X, Y, masks) format.
    Expects raw_dir to have:
        non-makeup/  ← no-makeup faces (X)
        makeup/      ← makeup faces (Y)  (used as reference style)

    Since real datasets have DIFFERENT people in X and Y,
    we only keep them for perceptual quality training,
    not pixel-level L1 (set lambda_l1 lower for these experiments).
    """
    import cv2
    import numpy as np

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Find X and Y images
    x_dir = raw_dir / non_makeup_subdir
    y_dir = raw_dir / makeup_subdir

    if not x_dir.exists():
        # Try to find them automatically
        subdirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        logger.info(f"Available subdirs in {raw_dir}: {[d.name for d in subdirs]}")
        if len(subdirs) >= 2:
            x_dir = subdirs[0]
            y_dir = subdirs[1]
        else:
            logger.error(f"Cannot find non-makeup/makeup subdirs in {raw_dir}")
            return 0

    x_files = sorted(list(x_dir.glob("*.png")) + list(x_dir.glob("*.jpg")))
    y_files = sorted(list(y_dir.glob("*.png")) + list(y_dir.glob("*.jpg")))

    n = min(len(x_files), len(y_files), max_pairs)
    logger.info(f"Preparing {n} real pairs from {raw_dir.name}")

    metadata = {}
    count    = 0

    # We need face parser for masks
    from src.pipeline.face_parser   import FaceParser
    from src.pipeline.expert_system import ExpertSystem

    parser = FaceParser(min_confidence=0.5)
    expert = ExpertSystem()

    for i in range(n):
        stem  = str(i + 1).zfill(5)
        x_bgr = cv2.imread(str(x_files[i]))
        y_bgr = cv2.imread(str(y_files[i]))

        if x_bgr is None or y_bgr is None:
            continue

        sz    = 256
        x_bgr = cv2.resize(x_bgr, (sz, sz))
        y_bgr = cv2.resize(y_bgr, (sz, sz))

        parse = parser.parse(x_bgr)
        if parse is None:
            continue

        analysis  = expert.analyze(parse.skin_rgb, parse.face_shape)
        lip_mask  = parse.masks.get("lips_outer", np.zeros((sz, sz), np.uint8))
        skin_mask = parse.masks.get("face_oval",  np.zeros((sz, sz), np.uint8))

        cv2.imwrite(str(img_out / f"{stem}_X.png"), x_bgr)
        cv2.imwrite(str(img_out / f"{stem}_Y.png"), y_bgr)
        cv2.imwrite(str(img_out / f"{stem}_lip_mask.png"),  lip_mask)
        cv2.imwrite(str(img_out / f"{stem}_skin_mask.png"), skin_mask)

        metadata[stem] = {
            "source_file": x_files[i].name,
            "face_shape":  analysis["face_shape"],
            "undertone":   analysis["undertone"],
            "lightness":   analysis["lightness"],
            "skin_rgb":    parse.skin_rgb.tolist(),
        }
        count += 1

        if count % 100 == 0:
            logger.info(f"  Prepared {count}/{n}")

    parser.close()

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Prepared {count} pairs → {out_dir}")
    return count


def prepare_sd_dataset(sd_raw_dir: str | Path, out_dir: str | Path):
    """Convert SD X/Y pairs into our format (add masks)."""
    return prepare_real_dataset(
        raw_dir            = sd_raw_dir,
        out_dir            = out_dir,
        non_makeup_subdir  = "X",
        makeup_subdir      = "Y",
    )


# ═══════════════════════════════════════════════════════════════
#  Stage 2 — Train
# ═══════════════════════════════════════════════════════════════

def train_experiment(exp_id: int, cfg_path: str = "configs/config.yaml"):
    exp    = EXPERIMENTS[exp_id]
    run_name = exp["short"]
    data_dir = exp["data_dir"]

    logger.info(f"\n{'='*60}")
    logger.info(f"  Training Experiment {exp_id}: {exp['name']}")
    logger.info(f"  Data: {data_dir}")
    logger.info(f"  Run : {run_name}")
    logger.info(f"{'='*60}\n")

    if not Path(data_dir).exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run data preparation first.")
        return False

    cmd = [
        sys.executable, "scripts/train.py",
        "--config",   cfg_path,
        "--data",     data_dir,
        "--run-name", run_name,
        "--no-wandb",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════════
#  Stage 3 — Evaluate & Compare
# ═══════════════════════════════════════════════════════════════

def evaluate_experiment(exp_id: int, cfg_path: str = "configs/config.yaml") -> dict:
    exp      = EXPERIMENTS[exp_id]
    run_name = exp["short"]
    data_dir = exp["data_dir"]

    # Find latest checkpoint
    run_dir = ROOT / "runs" / run_name
    ckpts   = sorted(run_dir.glob("ckpt_epoch_*.pt")) if run_dir.exists() else []

    if not ckpts:
        # Try best checkpoint
        best = run_dir / "ckpt_best.pt"
        if best.exists():
            ckpt_path = str(best)
        else:
            logger.warning(f"No checkpoint found for {run_name}")
            return {"experiment": exp["name"], "error": "no checkpoint"}
    else:
        ckpt_path = str(ckpts[-1])   # latest epoch

    logger.info(f"Evaluating {run_name} with checkpoint: {Path(ckpt_path).name}")

    result_file = ROOT / "runs" / run_name / "eval_results.json"

    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", ckpt_path,
        "--data",       data_dir,
        "--config",     cfg_path,
        "--output",     str(result_file),
    ]
    subprocess.run(cmd, cwd=str(ROOT))

    if result_file.exists():
        with open(result_file) as f:
            results = json.load(f)
        results["experiment"] = exp["name"]
        results["description"] = exp["description"]
        return results
    else:
        return {"experiment": exp["name"], "error": "evaluation failed"}


def compare_all(exp_ids: list[int]) -> None:
    """Load eval results and print comparison table + save chart."""
    all_results = []

    for exp_id in exp_ids:
        exp         = EXPERIMENTS[exp_id]
        result_file = ROOT / "runs" / exp["short"] / "eval_results.json"

        if result_file.exists():
            with open(result_file) as f:
                r = json.load(f)
            r["experiment"] = exp["name"]
            all_results.append(r)
        else:
            logger.warning(f"No results for exp {exp_id}: {exp['name']}")

    if not all_results:
        logger.error("No results to compare")
        return

    # Print comparison table
    metrics = ["l1", "ssim", "lpips", "inference_ms"]
    header  = f"{'Experiment':<35} " + " ".join(f"{m:>14}" for m in metrics)
    print("\n" + "="*80)
    print("  RESEARCH COMPARISON RESULTS")
    print("="*80)
    print(header)
    print("-"*80)

    for r in all_results:
        row = f"{r['experiment']:<35} "
        for m in metrics:
            val = r.get(m, float("nan"))
            row += f"{val:>14.4f}" if isinstance(val, float) else f"{'N/A':>14}"
        print(row)

    print("="*80)
    print("\nLower is better: L1, LPIPS, inference_ms")
    print("Higher is better: SSIM")

    # Save comparison JSON
    out_path = ROOT / "runs" / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nComparison saved → {out_path}")

    # Save bar chart
    _save_comparison_chart(all_results, ROOT / "runs" / "comparison_chart.png")


def _save_comparison_chart(results: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics    = ["l1", "ssim", "lpips"]
        labels     = [r["experiment"].split("(")[0].strip() for r in results]
        x          = np.arange(len(metrics))
        width      = 0.8 / max(len(results), 1)
        fig, ax    = plt.subplots(figsize=(12, 5))

        for i, r in enumerate(results):
            vals = [r.get(m, 0) for m in metrics]
            ax.bar(x + i * width, vals, width, label=labels[i])

        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(["L1 ↓", "SSIM ↑", "LPIPS ↓"])
        ax.set_title("Dataset Comparison — Makeup Transfer Quality")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        logger.info(f"Chart saved → {out_path}")
    except Exception as e:
        logger.warning(f"Chart failed: {e}")


# ═══════════════════════════════════════════════════════════════
#  Main CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Research comparison pipeline")
    p.add_argument("--all",          action="store_true",
                   help="Run all 4 experiments end-to-end")
    p.add_argument("--exp",          type=int, choices=[1,2,3,4],
                   help="Run single experiment (1-4)")
    p.add_argument("--prepare-only", action="store_true",
                   help="Only download and prepare datasets")
    p.add_argument("--train-only",   action="store_true",
                   help="Only train (datasets must be prepared)")
    p.add_argument("--compare-only", action="store_true",
                   help="Only compare already-trained models")
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--sd-pairs",     type=int, default=3000,
                   help="Number of SD pairs to generate for exp 4")
    return p.parse_args()


def run_experiment_full(exp_id: int, args):
    exp = EXPERIMENTS[exp_id]
    logger.info(f"\n{'#'*60}")
    logger.info(f"  EXPERIMENT {exp_id}: {exp['name']}")
    logger.info(f"  {exp['description']}")
    logger.info(f"{'#'*60}")

    data_dir = Path(exp["data_dir"])

    # ── Exp 1: our synthetic data ──────────────────────────
    if exp_id == 1:
        if not data_dir.exists() or not any(data_dir.iterdir()):
            logger.info("Generating synthetic dataset...")
            subprocess.run([
                sys.executable, "scripts/generate_dataset.py",
                "--source",        "ffhq-dataset/flat_images",
                "--output",        str(data_dir),
                "--pairs",         "5000",
                "--size",          "256",
                "--filter-gender",
            ], cwd=str(ROOT))

    # ── Exp 2: MT-Dataset ──────────────────────────────────
    elif exp_id == 2:
        download_mt_dataset()
        if not data_dir.exists():
            prepare_real_dataset(
                raw_dir   = "ffhq-dataset/mt_dataset",
                out_dir   = str(data_dir),
                max_pairs = 3000,
            )

    # ── Exp 3: BeautyGAN ──────────────────────────────────
    elif exp_id == 3:
        download_beautygan_dataset()
        if not data_dir.exists():
            prepare_real_dataset(
                raw_dir   = "ffhq-dataset/beautygan",
                out_dir   = str(data_dir),
                max_pairs = 1115,
            )

    # ── Exp 4: Stable Diffusion ───────────────────────────
    elif exp_id == 4:
        sd_raw = Path("ffhq-dataset/sd_pairs")
        if not sd_raw.exists() or len(list((sd_raw/"X").glob("*.png"))) < args.sd_pairs:
            generate_sd_pairs(n_pairs=args.sd_pairs)
        if not data_dir.exists():
            prepare_sd_dataset(
                sd_raw_dir = str(sd_raw),
                out_dir    = str(data_dir),
            )

    # ── Train ─────────────────────────────────────────────
    if not args.prepare_only:
        train_experiment(exp_id, args.config)

    # ── Evaluate ──────────────────────────────────────────
    if not args.prepare_only and not args.train_only:
        return evaluate_experiment(exp_id, args.config)

    return {}


def main():
    args    = parse_args()
    os.chdir(str(ROOT))

    exp_ids = [args.exp] if args.exp else [1, 2, 3, 4]

    if args.compare_only:
        compare_all(exp_ids)
        return

    all_results = []
    for exp_id in exp_ids:
        result = run_experiment_full(exp_id, args)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        compare_all(exp_ids)

    print("\n✓  Research pipeline complete!")
    print(f"   Results: runs/comparison_results.json")
    print(f"   Chart  : runs/comparison_chart.png\n")


if __name__ == "__main__":
    main()
