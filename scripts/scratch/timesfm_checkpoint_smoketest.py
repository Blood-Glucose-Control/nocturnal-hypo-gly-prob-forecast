#!/usr/bin/env python3
"""Smoketest: TimesFM checkpoint round-trip and checkpoint-N promotion.

Tests:
  1. Load existing model.pt -> predict -> _save_checkpoint -> _load_checkpoint
     -> predict again: assert predictions are numerically identical (max abs diff < 1e-4).
  2. Promote an existing HF Trainer checkpoint-N dir to model.pt format,
     load it, and assert predictions are non-NaN and the right length.

Run from project root:
  source .timesfm-venv/bin/activate  # or whichever venv has transformers
  python scripts/scratch/timesfm_checkpoint_smoketest.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.timesfm.config import TimesFMConfig  # noqa: E402
from src.models.timesfm.model import TimesFMForecaster  # noqa: E402

# -------------------------------------------------------------------
# Paths to an existing trained artifact (contains model.pt/ and checkpoint-* dirs)
# -------------------------------------------------------------------
ARTIFACT_DIR = (
    PROJECT_ROOT
    / "trained_models/artifacts/timesfm"
    / "2026-04-22_10:11_RID20260422_101119_gpu1_625_holdout_workflow"
)
MODEL_PT_DIR = ARTIFACT_DIR / "model.pt"


def make_context_df(series_length: int = 600, seed: int = 99) -> pd.DataFrame:
    """Tiny synthetic BG series for calling _predict()."""
    rng = np.random.default_rng(seed)
    bg = 7.0 + np.cumsum(rng.normal(0, 0.05, series_length))
    bg = np.clip(bg, 3.0, 22.0).astype(np.float32)
    dates = pd.date_range("2025-06-01", periods=series_length, freq="5min")
    return pd.DataFrame({"bg_mM": bg}, index=dates)


# -------------------------------------------------------------------
# Test 1: save/load round-trip on a trained model
# -------------------------------------------------------------------
def test_round_trip() -> None:
    print("\n=== Test 1: Save/Load Round-Trip ===")

    if not MODEL_PT_DIR.exists():
        print(f"  SKIP: model.pt not found at {MODEL_PT_DIR}")
        return

    # Initialize with base pretrained weights (cached); _load_checkpoint will
    # override with our fine-tuned hf_model/ weights.
    config = TimesFMConfig(
        context_length=512,
        forecast_length=96,
        torch_dtype="bfloat16",
    )

    # Load the trained model
    print(f"  Loading from {MODEL_PT_DIR} ...")
    forecaster = TimesFMForecaster(config)
    forecaster._load_checkpoint(str(MODEL_PT_DIR))

    context_df = make_context_df(series_length=600)
    pred_original = forecaster._predict(context_df, prediction_length=96)
    print(f"  Original forecast (first 5): {pred_original[:5]}")

    with tempfile.TemporaryDirectory() as tmpdir:
        resave_dir = os.path.join(tmpdir, "resaved_model.pt")

        # Save to a fresh location
        os.makedirs(resave_dir)
        forecaster._save_checkpoint(resave_dir)
        print(f"  Saved to {resave_dir}")
        print(f"  Contents: {os.listdir(resave_dir)}")
        print(f"  hf_model/: {os.listdir(os.path.join(resave_dir, 'hf_model'))}")

        # Load into a brand-new forecaster instance (base weights, then override)
        config2 = TimesFMConfig(
            context_length=512,
            forecast_length=96,
            torch_dtype="bfloat16",
        )
        fresh = TimesFMForecaster(config2)
        fresh._load_checkpoint(resave_dir)
        print(f"  Loaded from {resave_dir}")

        pred_loaded = fresh._predict(context_df, prediction_length=96)
        print(f"  Loaded forecast (first 5):   {pred_loaded[:5]}")

        max_diff = float(
            np.max(np.abs(pred_original.astype(float) - pred_loaded.astype(float)))
        )
        print(f"  Max abs diff: {max_diff:.2e}")
        assert max_diff < 1e-4, f"FAIL: predictions diverge by {max_diff:.2e}"

    print("  PASS: predictions match within 1e-4")


# -------------------------------------------------------------------
# Test 2: promote an existing checkpoint-N dir to model.pt format
# -------------------------------------------------------------------
def test_promote_checkpoint_n() -> None:
    print("\n=== Test 2: Promote checkpoint-N to model.pt format ===")

    if not ARTIFACT_DIR.exists():
        print(f"  SKIP: artifact dir not found at {ARTIFACT_DIR}")
        return

    ckpt_dirs = sorted(ARTIFACT_DIR.glob("checkpoint-*"))
    if not ckpt_dirs:
        print(f"  SKIP: no checkpoint-* dirs in {ARTIFACT_DIR}")
        return

    ckpt_dir = ckpt_dirs[-1]  # latest retained checkpoint
    run_hf_model = ARTIFACT_DIR / "hf_model"
    print(f"  Using checkpoint: {ckpt_dir.name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        promoted = Path(tmpdir) / "promoted_model.pt"
        hf_model_dir = promoted / "hf_model"
        hf_model_dir.mkdir(parents=True)

        # Copy the checkpoint's fine-tuned weights + the run's config.json
        # (checkpoint-N has model.safetensors at root; config.json lives in hf_model/)
        shutil.copy2(ckpt_dir / "model.safetensors", hf_model_dir / "model.safetensors")
        shutil.copy2(run_hf_model / "config.json", hf_model_dir / "config.json")

        # Read the epoch from trainer_state.json for informational purposes
        trainer_state_path = ckpt_dir / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path) as f:
                state = json.load(f)
            epoch = state.get("epoch", "?")
            step = ckpt_dir.name.split("-")[-1]
            print(f"  Checkpoint epoch: {epoch}, step: {step}")

        # Write a minimal timesfm_config.json
        timesfm_meta = {
            "checkpoint_path": None,
            "context_length": 512,
            "horizon_length": 96,
            "use_cpu": False,
            "is_finetuned": True,
        }
        with open(promoted / "timesfm_config.json", "w") as f:
            json.dump(timesfm_meta, f, indent=2)

        print(f"  Promoted dir contents: {[p.name for p in promoted.iterdir()]}")

        # Load and predict (base weights initialized, then overridden by _load_checkpoint)
        config = TimesFMConfig(
            context_length=512,
            forecast_length=96,
            torch_dtype="bfloat16",
        )
        forecaster = TimesFMForecaster(config)
        forecaster._load_checkpoint(str(promoted))

        context_df = make_context_df(series_length=600)
        pred = forecaster._predict(context_df, prediction_length=96)

        print(
            f"  Forecast length: {len(pred)}, range [{pred.min():.3f}, {pred.max():.3f}]"
        )
        assert len(pred) == 96, f"FAIL: expected 96 steps, got {len(pred)}"
        assert not np.isnan(pred).any(), "FAIL: predictions contain NaN"

    print("  PASS: checkpoint-N promoted and evaluated successfully")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    test_round_trip()
    test_promote_checkpoint_n()
    print("\n✓ All smoketests passed!")
