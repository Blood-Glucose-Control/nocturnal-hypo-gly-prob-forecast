#!/usr/bin/env python3
"""
Evaluate a saved Chronos-2 model on holdout data.

This script loads a previously fine-tuned Chronos2Forecaster and evaluates
it on Brown 2019 holdout data using midnight-anchored episodes. This is the
second half of the parity test: the model was trained via the model class
in example_chronos2_finetune.py, and now we verify it produces the same
RMSE as the original experiment script (validated: 1.890 RMSE).

Usage:
    # Evaluate a saved model
    python scripts/examples/example_chronos2_evaluate.py --model-dir models/chronos2_brown/20260218_123456

    # Via SLURM
    sbatch --export=MODEL_DIR=models/chronos2_brown/20260218_123456 scripts/training/slurm/chronos2_evaluate.sh
"""

import argparse
import logging
from pathlib import Path

from src.data.versioning.dataset_registry import DatasetRegistry
from src.models.chronos2 import Chronos2Forecaster
from src.models.chronos2.visualization import plot_evaluation_episodes

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose internal logging
logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)

# Validated baseline from the experiment script (Phase 1, 15K steps, IOB)
VALIDATED_RMSE = 1.890


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chronos-2 on holdout data")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to saved Chronos-2 model (from example_chronos2_finetune.py)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_5pct",
        help="Holdout config directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="brown_2019",
        help="Dataset name to evaluate on",
    )
    args = parser.parse_args()

    # =========================================================================
    # Step 1: Load the saved model
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Load saved model")
    print("=" * 60)

    # load() reads config.json + metadata.json from disk, then calls
    # _load_checkpoint(). Don't use _load_checkpoint() directly — that
    # would bypass config restoration and use fresh defaults instead.
    model = Chronos2Forecaster.load(args.model_dir)

    print(f"  Loaded from: {args.model_dir}")
    print(f"  is_fitted: {model.is_fitted}")
    print(f"  predictor: {model.predictor is not None}")
    print(f"  Config covariates: {model.config.covariate_cols}")

    # =========================================================================
    # Step 2: Load holdout data
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 2: Load holdout data")
    print("=" * 60)

    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_df = registry.load_holdout_data_only(args.dataset)

    print(f"  Dataset: {args.dataset}")
    print(f"  Holdout: {holdout_df.shape} ({holdout_df['p_num'].nunique()} patients)")

    # =========================================================================
    # Step 3: Evaluate
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 3: Evaluate on holdout (midnight-anchored episodes)")
    print("=" * 60)

    eval_result = model.evaluate(holdout_df, return_predictions=True)

    rmse = eval_result["rmse"]
    n_episodes = eval_result["n_episodes"]

    print(f"  RMSE: {rmse:.4f}")
    print(f"  Episodes: {n_episodes}")

    # =========================================================================
    # Step 4: Compare against validated baseline
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 4: Parity check")
    print("=" * 60)

    delta = rmse - VALIDATED_RMSE
    pct_diff = (delta / VALIDATED_RMSE) * 100

    print(f"  Model class RMSE:   {rmse:.4f}")
    print(f"  Validated baseline:  {VALIDATED_RMSE:.3f}")
    print(f"  Delta:               {delta:+.4f} ({pct_diff:+.1f}%)")

    # Allow +-5% tolerance for non-deterministic training
    tolerance = 0.05
    if abs(pct_diff) < tolerance * 100:
        print(f"  PARITY ACHIEVED (within {tolerance*100:.0f}% tolerance)")
    else:
        print(f"  WARNING: Outside {tolerance*100:.0f}% tolerance — investigate")

    # =========================================================================
    # Step 5: Per-episode statistics (if predictions available)
    # =========================================================================
    if "predictions" in eval_result:
        print()
        print("=" * 60)
        print("STEP 5: Prediction statistics")
        print("=" * 60)

        predictions = eval_result["predictions"]
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Prediction columns: {list(predictions.columns)}")

        # Show per-item stats if available
        items = predictions.index.get_level_values(0).unique()
        print(f"  Unique items (episodes): {len(items)}")

    # =========================================================================
    # Step 6: Generate evaluation plots (best/worst episodes)
    # =========================================================================
    if "episodes" in eval_result and "per_episode" in eval_result:
        print()
        print("=" * 60)
        print("STEP 6: Generate evaluation plots")
        print("=" * 60)

        plot_dir = str(Path(args.model_dir) / "eval_plots")
        plot_paths = plot_evaluation_episodes(
            episodes=eval_result["episodes"],
            per_episode=eval_result["per_episode"],
            output_dir=plot_dir,
            model_label=f"Chronos-2 ({args.dataset})",
            forecast_length=model.config.forecast_length,
            interval_mins=model.config.interval_mins,
            covariate_cols=model.config.covariate_cols,
        )

        for name, path in plot_paths.items():
            print(f"  {name}: {path}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
