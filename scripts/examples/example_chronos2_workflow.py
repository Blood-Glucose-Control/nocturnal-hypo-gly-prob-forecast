#!/usr/bin/env python3
"""
Minimal Chronos-2 workflow example.

Demonstrates the complete Chronos2Forecaster lifecycle:
1. Create config
2. Create model
3. Load data
4. Zero-shot evaluation
5. Fine-tune
6. Save/load roundtrip
7. Final evaluation

Usage:
    python scripts/examples/example_chronos2_workflow.py
"""

from pathlib import Path

import numpy as np

from src.data.diabetes_datasets.data_loader import get_loader
from src.models.chronos import Chronos2Config, Chronos2Forecaster

PROJECT_ROOT = Path(__file__).parent.parent.parent


def compute_rmse(predictions, episodes):
    """Compute mean RMSE across episodes."""
    rmse_list = []
    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:04d}"
        if item_id not in predictions.index.get_level_values(0):
            continue
        pred = predictions.loc[item_id]["mean"].values
        actual = ep["target_bg"][: len(pred)]
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


def main():
    print("=" * 60)
    print("CHRONOS-2 WORKFLOW EXAMPLE")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("\n[1] Loading data...")
    loader = get_loader("brown_2019", dataset_type="train", use_cached=True)
    print(f"    Train patients: {len(loader.train_data)}")
    print(f"    Val patients: {len(loader.validation_data)}")

    # =========================================================================
    # STEP 2: Zero-shot evaluation
    # =========================================================================
    print("\n[2] Zero-shot evaluation...")

    config_zs = Chronos2Config(
        training_mode="zero_shot",
        context_length=512,
        forecast_length=72,
        max_train_episodes=500,
        max_val_episodes=500,
    )
    model_zs = Chronos2Forecaster(config_zs)

    output_dir_zs = str(PROJECT_ROOT / "models/chronos2_example_zeroshot")
    model_zs.fit(
        train_data=loader.train_data,
        output_dir=output_dir_zs,
        val_data=loader.validation_data,
    )

    # Evaluate
    val_episodes = model_zs._val_episodes
    ts_val, _ = model_zs._format_for_autogluon(val_episodes)
    predictions = model_zs.predictor.predict(ts_val)
    rmse_zs = compute_rmse(predictions, val_episodes)

    print(f"    Zero-shot RMSE: {rmse_zs:.4f}")

    # =========================================================================
    # STEP 3: Fine-tune model
    # =========================================================================
    print("\n[3] Fine-tuning model...")

    config_ft = Chronos2Config(
        training_mode="fine_tune",
        fine_tune_steps=500,  # Quick test
        fine_tune_lr=1e-4,
        context_length=512,
        forecast_length=72,
        max_train_episodes=1000,
        max_val_episodes=500,
    )
    model_ft = Chronos2Forecaster(config_ft)

    output_dir = str(PROJECT_ROOT / "models/chronos2_example")
    model_ft.fit(
        train_data=loader.train_data,
        output_dir=output_dir,
        val_data=loader.validation_data,
    )

    # Evaluate fine-tuned
    val_episodes = model_ft._val_episodes
    ts_val, _ = model_ft._format_for_autogluon(val_episodes)
    predictions = model_ft.predictor.predict(ts_val)
    rmse_ft = compute_rmse(predictions, val_episodes)

    print(f"    Fine-tuned RMSE: {rmse_ft:.4f}")
    print(f"    Model saved to: {output_dir}")

    # =========================================================================
    # STEP 4: Load model and verify
    # =========================================================================
    print("\n[4] Load model from checkpoint...")

    # load() is a classmethod - pass config to ensure Chronos2-specific params are preserved
    model_loaded = Chronos2Forecaster.load(output_dir, config_ft)

    # Rebuild episodes for loaded model (it doesn't have them cached)
    val_episodes = model_loaded._build_episodes(loader.validation_data)
    if len(val_episodes) > 500:
        np.random.seed(42)
        indices = np.random.choice(len(val_episodes), 500, replace=False)
        val_episodes = [val_episodes[i] for i in sorted(indices)]

    ts_val, _ = model_loaded._format_for_autogluon(val_episodes)
    predictions = model_loaded.predictor.predict(ts_val)
    rmse_loaded = compute_rmse(predictions, val_episodes)

    print(f"    Loaded model RMSE: {rmse_loaded:.4f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Zero-shot RMSE:    {rmse_zs:.4f}")
    print(f"  Fine-tuned RMSE:   {rmse_ft:.4f}")
    print(f"  Loaded model RMSE: {rmse_loaded:.4f}")
    print(f"  Improvement:       {(rmse_zs - rmse_ft) / rmse_zs * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
