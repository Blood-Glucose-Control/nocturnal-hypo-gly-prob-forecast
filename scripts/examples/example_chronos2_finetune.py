#!/usr/bin/env python3
"""
Fine-tune Chronos-2 on Brown 2019 using the model class + registry.

This script demonstrates the Chronos2Forecaster workflow end-to-end:
1. Load training data from the DatasetRegistry (with holdout split)
2. Configure the model via Chronos2Config
3. Fine-tune via model.fit() (gap handling + sliding windows + IOB)
4. Save the model for later evaluation

The goal is to replicate the validated experiment result (1.890 RMSE)
using the model class interface instead of raw AutoGluon calls.

Usage:
    # On watgpu (GPU required for fine-tuning)
    python scripts/examples/example_chronos2_finetune.py

    # Custom steps / output
    python scripts/examples/example_chronos2_finetune.py --steps 5000 --output-dir models/chronos2_quick

    # Via SLURM
    sbatch scripts/training/slurm/chronos2_finetune.sh
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.data.versioning.dataset_registry import DatasetRegistry
from src.models.chronos2 import Chronos2Config, Chronos2Forecaster

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose internal logging
logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 on Brown 2019")
    parser.add_argument(
        "--steps",
        type=int,
        default=15000,
        help="Number of fine-tuning steps (validated best: 15000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="AutoGluon time limit in seconds (None = unlimited)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/chronos2_brown",
        help="Directory to save the fine-tuned model",
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
        help="Dataset name to train on",
    )
    args = parser.parse_args()

    # =========================================================================
    # Step 1: Load data from registry
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Load data from registry")
    print("=" * 60)

    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    train_df = registry.load_training_data_only(args.dataset)

    print(f"  Dataset: {args.dataset}")
    print(f"  Train: {train_df.shape} ({train_df['p_num'].nunique()} patients)")

    # =========================================================================
    # Step 2: Configure the model
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 2: Configure Chronos-2")
    print("=" * 60)

    config = Chronos2Config(
        # Model
        model_path="autogluon/chronos-2",
        training_mode="fine_tune",
        # Architecture (must match Chronos-2 base model)
        context_length=512,  # ~42.7 hours at 5-min intervals
        forecast_length=72,  # 6 hours at 5-min intervals
        # Fine-tuning
        fine_tune_steps=args.steps,
        fine_tune_lr=args.lr,
        time_limit=args.time_limit,
        # Covariates — Brown 2019 has IOB from Hovorka model.
        # For Aleppo/BrisT1D, use ["iob", "cob"].
        covariate_cols=["iob"],
        # Gap handling — applied inside model._prepare_training_data()
        imputation_threshold_mins=45,  # interpolate gaps up to 45 min
        # min_segment_length auto-computed as context_length + forecast_length = 584
    )

    print(f"  Steps: {config.fine_tune_steps}")
    print(f"  LR: {config.fine_tune_lr}")
    print(f"  Context: {config.context_length} ({config.context_length * 5 / 60:.1f}h)")
    print(
        f"  Forecast: {config.forecast_length} ({config.forecast_length * 5 / 60:.1f}h)"
    )
    print(f"  Covariates: {config.covariate_cols}")
    print(f"  Min segment length: {config.min_segment_length}")
    print(f"  AutoGluon hyperparams: {config.get_autogluon_hyperparameters()}")

    # =========================================================================
    # Step 3: Create model and fine-tune
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 3: Fine-tune")
    print("=" * 60)

    # Timestamped output directory to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{timestamp}"
    print(f"  Output: {output_dir}")

    model = Chronos2Forecaster(config)
    results = model.fit(train_df, output_dir=output_dir)

    print()
    print("  Training complete!")
    print(f"  Predictor path: {results['train_metrics'].get('predictor_path', 'N/A')}")

    # =========================================================================
    # Step 4: Save model
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 4: Save model")
    print("=" * 60)

    model.save(output_dir)
    print(f"  Saved to: {output_dir}")
    print(f"  Contents: {[p.name for p in Path(output_dir).iterdir()]}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Model saved to: {output_dir}")
    print("  To evaluate, run:")
    print(
        f"    python scripts/examples/example_chronos2_evaluate.py --model-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
