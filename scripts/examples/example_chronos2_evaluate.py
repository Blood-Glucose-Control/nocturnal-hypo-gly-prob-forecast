#!/usr/bin/env python3
"""
Evaluate a fine-tuned Chronos-2 model using evaluate_nocturnal_forecasting().

Validates that the new evaluation infrastructure (episode_builders +
holdout_eval) produces the same results as the original manual evaluation.
Expected: ~1.890 RMSE for the 15K-step Brown 2019 model.

Usage:
    python scripts/examples/example_chronos2_evaluate.py --model-dir models/chronos2_brown/<timestamp>
"""

import argparse
import logging

from src.data.versioning.dataset_registry import DatasetRegistry
from src.models.chronos2 import Chronos2Forecaster

# Import from the new evaluation infrastructure
from src.evaluation.nocturnal import evaluate_nocturnal_forecasting

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chronos-2 model")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to saved model directory",
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
        help="Dataset name",
    )
    args = parser.parse_args()

    # Step 1: Load model
    print("=" * 60)
    print("STEP 1: Load model")
    print("=" * 60)
    model = Chronos2Forecaster.load(args.model_dir)
    config = model.config
    print(f"  Model loaded from: {args.model_dir}")
    print(f"  Context: {config.context_length}, Forecast: {config.forecast_length}")
    print(f"  Covariates: {config.covariate_cols}")

    # Step 2: Load holdout data
    print()
    print("=" * 60)
    print("STEP 2: Load holdout data")
    print("=" * 60)
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_df = registry.load_holdout_data_only(args.dataset)
    print(f"  Holdout: {holdout_df.shape} ({holdout_df['p_num'].nunique()} patients)")

    # Step 3: Evaluate using evaluate_nocturnal_forecasting
    print()
    print("=" * 60)
    print("STEP 3: Evaluate (midnight-anchored)")
    print("=" * 60)
    results = evaluate_nocturnal_forecasting(
        model=model,
        holdout_data=holdout_df,
        context_length=config.context_length,
        forecast_length=config.forecast_length,
        target_col=config.target_col,
        covariate_cols=config.covariate_cols,
        interval_mins=config.interval_mins,
    )

    # Step 4: Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Overall RMSE: {results['overall_rmse']:.4f}")
    print(f"  Total episodes: {results['total_episodes']}")
    print()
    print("  Per-patient:")
    for p in results["per_patient"]:
        print(
            f"    Patient {p['patient_id']}: "
            f"RMSE={p['rmse']:.3f}, MAE={p['mae']:.3f}, "
            f"episodes={p['episodes']}"
        )

    print()
    print("  Target RMSE: ~1.890 (validated baseline)")
    print(f"  Actual RMSE: {results['overall_rmse']:.4f}")
    diff_pct = abs(results["overall_rmse"] - 1.890) / 1.890 * 100
    print(f"  Difference: {diff_pct:.1f}%")


if __name__ == "__main__":
    main()
