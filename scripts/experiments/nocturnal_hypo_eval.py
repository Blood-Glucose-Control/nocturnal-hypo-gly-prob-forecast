#!/usr/bin/env python3
r"""
Nocturnal Hypoglycemia Evaluation Script.

Evaluates models on the midnight-anchored nocturnal forecasting task — a clinically
meaningful evaluation where context ends at midnight and the model forecasts 6 hours
of overnight blood glucose when hypoglycemia risk is highest.

This is fundamentally different from sliding-window evaluation (sliding_window_eval.py),
which measures general forecast accuracy across all times of day. These two evaluation
modes produce DIFFERENT RMSE numbers and must never be compared on the same leaderboard.

Usage:
    # Zero-shot:
    python scripts/experiments/nocturnal_hypo_eval.py --model chronos2 --dataset brown_2019

    # Fine-tuned with checkpoint:
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model chronos2 --dataset tamborlane_2008 \
        --checkpoint trained_models/artifacts/chronos2/.../model.pt

    # Full options:
    python scripts/experiments/nocturnal_hypo_eval.py \
        --model chronos2 --dataset brown_2019 \
        --config-dir configs/data/holdout_10pct \
        --context-length 512 --forecast-length 96 \
        --cuda-device 0 --covariate-cols iob

    # Exact commands for past experiments are saved in each run's
    # experiment_config.json under "reproducibility_command".
"""

import argparse
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.utils import get_patient_column
from src.evaluation.nocturnal import (
    evaluate_nocturnal_forecasting,
    plot_best_worst_episodes,
    STEPS_PER_HOUR,
)
from src.evaluation.storage import write_nocturnal_results
from src.models import create_model_and_config
from src.utils import get_git_commit_hash, setup_file_logging, load_yaml_config

# Configure root logger for console output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_output_directory(
    model_name: str,
    dataset_name: str,
    context_length: int,
    forecast_length: int,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Create and return output directory path."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        mode = "finetuned" if checkpoint else "zeroshot"
        output_dir = (
            f"./experiments/nocturnal_forecasting/"
            f"{context_length}ctx_{forecast_length}fh/{model_name}/"
            f"{timestamp}_{dataset_name}_{mode}"
        )

    output_path = Path(output_dir)
    output_path = _resolve_output_dir_collision(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _resolve_output_dir_collision(output_path: Path) -> Path:
    """Avoid overwriting previous eval artifacts.

    If *output_path* already contains nocturnal evaluation outputs, append a
    deterministic ``_rerunNN`` suffix and return the first available path.
    """
    marker_files = (
        "nocturnal_results.json",
        "experiment_config.json",
        "nocturnal_evaluation.log",
    )

    if not output_path.exists():
        return output_path

    if not any((output_path / marker).exists() for marker in marker_files):
        return output_path

    idx = 1
    while True:
        candidate = Path(f"{output_path}_rerun{idx:02d}")
        if not candidate.exists():
            logger.warning(
                "Output directory %s already contains results; writing to %s",
                output_path,
                candidate,
            )
            return candidate
        idx += 1


def save_experiment_config(
    args: argparse.Namespace,
    model_config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save experiment configuration for reproducibility."""
    config = {
        "evaluation_type": "nocturnal_hypoglycemia",
        "cli_args": vars(args),
        "model_config": model_config,
        "environment": {
            "git_commit": get_git_commit_hash(),
            "python_version": sys.version.split()[0],
            "timestamp": datetime.now().isoformat(),
        },
        "reproducibility_command": shlex.join(sys.argv),
    }

    config_file = output_path / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Experiment config saved to: {config_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Nocturnal hypoglycemia evaluation (midnight-anchored forecasting)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ttm",
        choices=[
            "sundial",
            "ttm",
            "chronos2",
            "moirai",
            "moment",
            "timegrad",
            "timesfm",
            "tide",
            "toto",
            "naive_baseline",
            "statistical",
            "deepar",
            "patchtst",
            "tft",
        ],
        help="Model type to use for evaluation",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="brown_2019",
        help="Dataset name",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_10pct",
        help="Holdout config directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context window length in steps (default: 512)",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=72,
        help="Forecast horizon in steps (default: 72 = 6 hours)",
    )
    parser.add_argument(
        "--covariate-cols",
        type=str,
        nargs="*",
        default=None,
        help="Covariate column names (e.g., iob cob)",
    )
    parser.add_argument(
        "--patients",
        type=str,
        nargs="+",
        default=None,
        help="Subset of patient IDs to evaluate (default: all holdout patients)",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=1,
        help="CUDA device ID to use (default: 1)",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        default=False,
        help="Use predict_quantiles() and compute WQL + Brier@3.9 "
        "(model must support probabilistic forecasting)",
    )
    parser.add_argument(
        "--no-dilate",
        action="store_true",
        default=False,
        help="Skip DILATE (Soft-DTW shape) metrics. Useful for large runs "
        "where the O(n_episodes * forecast_length^2) cost is prohibitive.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set CUDA device to avoid DataParallel issues
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    # Load config from file if provided
    config_dict = load_yaml_config(args.model_config) if args.model_config else {}

    # Prepare model kwargs — pop model_type to avoid collision with the
    # positional argument in create_model_and_config()
    model_kwargs = {
        **config_dict,
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
    }
    model_kwargs.pop("model_type", None)

    # Initialize model
    logger.info(f"\n--- Initializing {args.model.upper()} ---")
    model, config = create_model_and_config(
        args.model, checkpoint=args.checkpoint, **model_kwargs
    )

    # Early check: ensure model supports probabilistic forecasting if requested
    if args.probabilistic and not model.supports_probabilistic_forecast:
        logger.error(
            "%s does not support probabilistic forecasting. "
            "Remove --probabilistic or use a model that supports it "
            "(e.g., chronos2).",
            args.model,
        )
        raise ValueError("Model does not support probabilistic forecasting")

    context_length = args.context_length
    forecast_length = args.forecast_length
    mode = "Fine-tuned" if args.checkpoint else "Zero-shot"

    # Setup output directory and file logging BEFORE logging config,
    # so everything goes to both console and file in one pass.
    output_path = setup_output_directory(
        args.model,
        args.dataset,
        context_length,
        forecast_length,
        args.checkpoint,
        args.output_dir,
    )
    log_file = setup_file_logging(output_path, "nocturnal_evaluation.log")

    # Log configuration (captured by both console and file handlers)
    logger.info("=" * 60)
    logger.info("NOCTURNAL HYPOGLYCEMIA EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model} ({mode})")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(
        f"Context: {context_length} steps ({context_length / STEPS_PER_HOUR:.1f} hours)"
    )
    logger.info(
        f"Forecast: {forecast_length} steps ({forecast_length / STEPS_PER_HOUR:.1f} hours)"
    )
    logger.info(f"Output: {output_path}")
    logger.info(f"Log file: {log_file}")

    # Load holdout data
    logger.info("\n--- Loading Holdout Data ---")
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = get_patient_column(holdout_data)
    if args.patients:
        requested = set(args.patients)
        available = set(holdout_data[patient_col].unique())
        missing = requested - available
        if missing:
            logger.warning(
                f"Requested patients not found in holdout data: {sorted(missing)}"
            )
        holdout_data = holdout_data[holdout_data[patient_col].isin(args.patients)]
    patients = holdout_data[patient_col].unique()
    if args.patients:
        logger.info(
            f"Filtered to {len(patients)} of {len(args.patients)} requested patients: {list(patients)}"
        )
    logger.info(f"Evaluating patients: {list(patients)}")
    logger.info(f"Total samples: {len(holdout_data):,}")

    # Auto-detect covariates from model config if not explicitly specified.
    # Fine-tuned models (e.g., Chronos-2 with IOB) need the same columns at
    # predict time as were present during training.
    covariate_cols = args.covariate_cols
    if (
        covariate_cols is None
        and hasattr(config, "covariate_cols")
        and config.covariate_cols
    ):
        covariate_cols = config.covariate_cols
        logger.info("Using covariates from model config: %s", covariate_cols)

    # Build resolved config dict once (used in experiment_config.json and results)
    resolved_config = {
        **config_dict,
        "context_length": context_length,
        "forecast_length": forecast_length,
        "covariate_cols": covariate_cols,
    }

    # Save experiment configuration
    save_experiment_config(args, resolved_config, output_path)

    # Auto-detect covariates from model config if not explicitly specified.
    # Fine-tuned models (e.g., Chronos-2 with IOB) need the same columns at
    # predict time as were present during training.
    covariate_cols = args.covariate_cols
    if (
        covariate_cols is None
        and hasattr(config, "covariate_cols")
        and config.covariate_cols
    ):
        covariate_cols = config.covariate_cols
        logger.info("Using covariates from model config: %s", covariate_cols)

    # Run nocturnal evaluation
    logger.info("\n--- Running Nocturnal Evaluation ---")
    results = evaluate_nocturnal_forecasting(
        model=model,
        holdout_data=holdout_data,
        context_length=context_length,
        forecast_length=forecast_length,
        covariate_cols=covariate_cols,
        probabilistic=args.probabilistic,
        compute_dilate=not args.no_dilate,
    )

    # Log overall results
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall RMSE: {results['overall_rmse']:.4f}")
    if "overall_wql" in results:
        logger.info(f"Overall WQL:  {results['overall_wql']:.4f}")
        logger.info(f"Overall Brier@3.9: {results['overall_brier']:.4f}")
        logger.info(f"Overall MACE: {results['overall_mace']:.4f}")
        for lvl in (50, 80, 90, 95):
            key = f"overall_coverage_{lvl}"
            if key in results:
                logger.info(f"Coverage {lvl}%%: {results[key]:.3f}")
        for lvl in (50, 80, 90, 95):
            key = f"overall_sharpness_{lvl}"
            if key in results:
                logger.info(f"Sharpness {lvl}%%: {results[key]:.3f}")
    logger.info(f"Total midnight episodes: {results['total_episodes']}")

    # Save results (3-tier storage)
    tier_metadata = {
        "evaluation_type": "nocturnal_hypoglycemia",
        "model": args.model,
        "mode": mode.lower(),
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": resolved_config,
    }
    written = write_nocturnal_results(results, output_path, tier_metadata)
    for tier_name, tier_path in written.items():
        logger.info(f"  {tier_name}: {tier_path}")

    # Generate best/worst episode plots
    logger.info("\n--- Generating Plots ---")
    plot_best_worst_episodes(
        per_episode=results["per_episode"],
        output_path=output_path,
        model_name=args.model,
        dataset_name=args.dataset,
        is_finetuned=bool(args.checkpoint),
    )

    logger.info("\n")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
