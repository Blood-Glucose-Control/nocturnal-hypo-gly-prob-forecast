#!/usr/bin/env python3
"""
Sliding Window Evaluation Script.

Standardized benchmarking script for fair model comparison using sliding window
episode generation. Evaluates general forecast accuracy across all times of day.

For nocturnal hypoglycemia-specific evaluation (midnight-anchored), use
nocturnal_hypo_eval.py instead.

Usage:
    python scripts/experiments/sliding_window_eval.py --model sundial --dataset brown_2019
    python scripts/experiments/sliding_window_eval.py --model ttm --dataset brown_2019
    python scripts/experiments/sliding_window_eval.py --model sundial --context-length 512 --forecast-length 72
    python scripts/experiments/sliding_window_eval.py --model sundial --model-config configs/models/sundial.yaml
    python scripts/experiments/sliding_window_eval.py --model sundial --checkpoint path/to/checkpoint
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.utils import get_patient_column
from src.evaluation.metrics import compute_regression_metrics
from src.models import create_model_and_config
from src.models.base import BaseTimeSeriesFoundationModel
from src.utils import get_git_commit_hash, setup_file_logging, load_yaml_config

# Constants
SAMPLING_INTERVAL_MINUTES = 5
STEPS_PER_HOUR = 60 // SAMPLING_INTERVAL_MINUTES
HYPO_THRESHOLD_MMOL = 3.9
DEFAULT_CONTEXT_HOURS_PLOT = 3
MAX_PLOT_EXAMPLES = 8

# Configure root logger for console output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def iter_episodes(patient_df: pd.DataFrame, context_length: int, forecast_length: int):
    """Yield fixed-size (context_df, target_values) episodes via sliding window.

    Args:
        patient_df: DataFrame with 'bg_mM' column
        context_length: Number of steps for context window
        forecast_length: Number of steps to forecast

    Yields:
        Tuple of (context_df, target_values) where:
            - context_df: DataFrame with context_length rows
            - target_values: numpy array of forecast_length values
    """
    total_length = context_length + forecast_length
    # Sliding window with stride = forecast_length (non-overlapping forecasts)
    for start in range(0, len(patient_df) - total_length + 1, forecast_length):
        episode = patient_df.iloc[start : start + total_length]
        context = episode.iloc[:context_length]
        target = episode["bg_mM"].values[context_length:]
        yield context, target


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.

    Wrapper around shared metrics utility for backwards compatibility.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        Dictionary with rmse, mae, mape, mse
    """
    return compute_regression_metrics(predictions, targets)


def compute_weighted_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute episode-weighted overall metrics.

    Args:
        results: List of per-patient result dictionaries

    Returns:
        Dictionary with overall rmse, mae, mape, mse
    """
    if not results:
        return {}

    total_episodes = sum(r["episodes"] for r in results)
    overall_rmse = np.sqrt(
        sum(r["rmse"] ** 2 * r["episodes"] for r in results) / total_episodes
    )
    overall_mae = sum(r["mae"] * r["episodes"] for r in results) / total_episodes
    overall_mape = sum(r["mape"] * r["episodes"] for r in results) / total_episodes

    return {
        "rmse": overall_rmse,
        "mae": overall_mae,
        "mape": overall_mape,
        "mse": overall_rmse**2,
    }


def plot_forecast_examples(
    examples: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    dataset_name: str,
    context_length: int,
    forecast_length: int,
    is_finetuned: bool,
    context_hours_to_show: int = DEFAULT_CONTEXT_HOURS_PLOT,
):
    """Generate forecast visualization plots.

    Args:
        examples: List of example dicts with 'patient', 'context', 'target', 'pred'
        output_path: Directory to save plot
        model_name: Name of the model
        dataset_name: Name of the dataset
        context_length: Context window length
        forecast_length: Forecast horizon length
        is_finetuned: Whether using fine-tuned or zero-shot model
        context_hours_to_show: Hours of context to display in plot
    """
    if not examples:
        return

    n = len(examples)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    context_steps_to_show = context_hours_to_show * STEPS_PER_HOUR

    for i, ex in enumerate(examples):
        ax = axes[i]
        ctx_full = ex["context"]
        tgt = ex["target"]
        pred = ex["pred"]

        # Truncate context to last N hours for visualization
        ctx = (
            ctx_full[-context_steps_to_show:]
            if len(ctx_full) > context_steps_to_show
            else ctx_full
        )

        # Time axis in hours (0 = forecast start)
        t_ctx = (np.arange(len(ctx)) - len(ctx)) / STEPS_PER_HOUR
        t_pred = np.arange(len(tgt)) / STEPS_PER_HOUR

        # Plot BG
        ax.plot(t_ctx, ctx, "b-", lw=1.5, label="BG (context)")
        ax.plot(t_pred, tgt, "g-", lw=2, label="BG (actual)")
        ax.plot(t_pred, pred, "r--", lw=2, alpha=0.8, label="BG (forecast)")

        ax.axvline(0, color="gray", ls=":", lw=1.5, label="Forecast start")
        ax.axhline(
            HYPO_THRESHOLD_MMOL,
            color="crimson",
            ls="--",
            alpha=0.4,
            lw=1,
            label="Hypo threshold",
        )

        ax.set_ylabel("BG (mmol/L)", fontsize=9)
        ax.set_ylim(0, 18)

        rmse = compute_metrics(pred, tgt)["rmse"]
        ax.set_title(f"{ex['patient']}: RMSE={rmse:.2f} mmol/L", fontsize=10)
        ax.set_xlabel("Hours", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    # Hide extra subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Legend for first plot
    if n > 0:
        axes[0].legend(fontsize=7, loc="upper right")

    mode_str = "Fine-tuned" if is_finetuned else "Zero-Shot"
    fig.suptitle(
        f"{model_name.upper()} {mode_str} Forecasts - {dataset_name}\n"
        f"(Context: {context_length} steps, Forecast: {forecast_length} steps)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    plot_file = output_path / "forecasts.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to: {plot_file}")


def setup_output_directory(
    model_name: str,
    dataset_name: str,
    context_length: int,
    forecast_length: int,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Create and return output directory path.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        context_length: Context window length in steps
        forecast_length: Forecast horizon in steps
        checkpoint: Optional checkpoint path
        output_dir: Optional custom output directory

    Returns:
        Path object for output directory
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        mode = "finetuned" if checkpoint else "zeroshot"
        output_dir = (
            f"./experiments/standard_forecasting/"
            f"{context_length}ctx_{forecast_length}fh/{model_name}/"
            f"{timestamp}_{dataset_name}_{mode}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_experiment_config(
    args: argparse.Namespace,
    model_config: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save experiment configuration for reproducibility.

    Args:
        args: Parsed command line arguments
        model_config: Model-specific configuration dictionary
        output_path: Output directory path
    """
    # Build reproducibility command
    cmd_parts = ["python", "scripts/experiments/sliding_window_eval.py"]
    cmd_parts.extend(["--model", args.model])
    cmd_parts.extend(["--dataset", args.dataset])
    cmd_parts.extend(["--config-dir", args.config_dir])
    cmd_parts.extend(["--context-length", str(args.context_length)])
    cmd_parts.extend(["--forecast-length", str(args.forecast_length)])
    if args.checkpoint:
        cmd_parts.extend(["--checkpoint", args.checkpoint])
    if args.model_config:
        cmd_parts.extend(["--model-config", args.model_config])

    config = {
        "cli_args": {
            "model": args.model,
            "dataset": args.dataset,
            "config_dir": args.config_dir,
            "checkpoint": args.checkpoint,
            "context_length": args.context_length,
            "forecast_length": args.forecast_length,
            "model_config": args.model_config,
            "output_dir": args.output_dir,
        },
        "model_config": model_config,
        "environment": {
            "git_commit": get_git_commit_hash(),
            "python_version": sys.version.split()[0],
            "timestamp": datetime.now().isoformat(),
        },
        "reproducibility_command": " ".join(cmd_parts),
    }

    config_file = output_path / "experiment_configs.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Experiment config saved to: {config_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Model-agnostic holdout evaluation for standardized benchmarking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sundial",
        choices=["sundial", "ttm", "chronos", "moirai"],
        help="Model type to use for evaluation",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config YAML file (optional, CLI args override)",
    )
    parser.add_argument(
        "--dataset", type=str, default="brown_2019", help="Dataset name"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_5pct",
        help="Holdout config directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint (if None, uses zero-shot)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context window length in steps (default: from checkpoint or 512)",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=None,
        help="Forecast horizon in steps (default: from checkpoint or 96)",
    )
    return parser.parse_args()


def evaluate_patient(
    patient_id: str,
    patient_df: pd.DataFrame,
    model: BaseTimeSeriesFoundationModel,
    context_length: int,
    forecast_length: int,
    collect_examples: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate model on a single patient's data.

    Args:
        patient_id: Patient identifier
        patient_df: DataFrame with patient's time series
        model: Forecasting model
        context_length: Context window length
        forecast_length: Forecast horizon length
        collect_examples: Whether to collect example for plotting

    Returns:
        Tuple of (patient_results, examples_list)
    """
    patient_preds = []
    patient_targets = []
    examples = []

    for context_df, target in iter_episodes(
        patient_df, context_length, forecast_length
    ):
        context_values = context_df["bg_mM"].values

        # Skip episodes with NaN values
        if np.isnan(context_values).any() or np.isnan(target).any():
            continue

        # Predict using unified interface (forecast_length is set in model config)
        pred = model.predict(context_df)

        # Handle multi-dimensional predictions: (batch, forecast_length, channels)
        # Extract only the target channel (first channel = bg_mM) and flatten
        if pred.ndim == 3:
            # Shape (batch, forecast_length, channels) -> (forecast_length,)
            pred = pred[0, :, 0]  # First batch, all timesteps, first channel
        elif pred.ndim == 2:
            # Shape (batch, forecast_length) -> (forecast_length,)
            pred = pred.flatten()

        patient_preds.append(pred)
        patient_targets.append(target)

        # Store first valid example for plotting
        if collect_examples and len(patient_preds) == 1:
            examples.append(
                {
                    "patient": patient_id,
                    "context": context_values,
                    "target": target,
                    "pred": pred,
                }
            )

    if not patient_preds:
        logger.warning(f"  {patient_id}: No valid windows")
        return None, examples

    # Aggregate patient metrics
    preds = np.concatenate(patient_preds)
    targets = np.concatenate(patient_targets)
    metrics = compute_metrics(preds, targets)

    results = {
        "patient": patient_id,
        "episodes": len(patient_preds),
        **metrics,
    }

    logger.info(
        f" Pid {patient_id}: RMSE={metrics['rmse']:.3f}, "
        f"MAE={metrics['mae']:.3f} ({len(patient_preds)} episodes)"
    )

    return results, examples


def save_results(
    results: Dict[str, Any], all_results: List[Dict[str, Any]], output_path: Path
):
    """Save evaluation results to JSON and CSV.

    Args:
        results: Overall results dictionary
        all_results: List of per-patient results
        output_path: Output directory path
    """
    # Save JSON
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Save CSV
    if all_results:
        pd.DataFrame(all_results).to_csv(
            output_path / "patient_metrics.csv", index=False
        )


def main():
    args = parse_arguments()

    # Load config from file if provided
    config_dict = load_yaml_config(args.model_config) if args.model_config else {}

    # Prepare model kwargs - only include CLI args if explicitly specified
    model_kwargs = {**config_dict}
    if args.context_length is not None:
        model_kwargs["context_length"] = args.context_length
    if args.forecast_length is not None:
        model_kwargs["forecast_length"] = args.forecast_length

    # Initialize model first - when loading checkpoint, saved config is used as base
    # with CLI overrides validated and applied
    logger.info("=" * 60)
    logger.info("MODEL-AGNOSTIC HOLDOUT EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {'Fine-tuned' if args.checkpoint else 'Zero-shot'}")

    if args.model_config:
        logger.info(f"Model config file: {args.model_config}")

    logger.info(f"\n--- Initializing {args.model.upper()} ---")
    model, config = create_model_and_config(
        args.model, checkpoint=args.checkpoint, **model_kwargs
    )

    # Use actual config values (may differ from CLI args if loading checkpoint)
    context_length = config.context_length
    forecast_length = config.forecast_length

    logger.info(f"Dataset: {args.dataset}")
    logger.info(
        f"Context: {context_length} steps "
        f"({context_length / STEPS_PER_HOUR:.1f} hours)"
    )
    logger.info(
        f"Forecast: {forecast_length} steps "
        f"({forecast_length / STEPS_PER_HOUR:.1f} hours)"
    )

    # Setup output directory (after we know actual config values)
    output_path = setup_output_directory(
        args.model,
        args.dataset,
        context_length,
        forecast_length,
        args.checkpoint,
        args.output_dir,
    )

    # Setup file logging to capture all evaluation output
    log_file = setup_file_logging(output_path)
    logger.info(f"Output: {output_path}")
    logger.info(f"Log file: {log_file}")

    # Re-log configuration now that file logging is active
    logger.info("=" * 60)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {'Fine-tuned' if args.checkpoint else 'Zero-shot'}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(
        f"Context: {context_length} steps ({context_length / STEPS_PER_HOUR:.1f} hours)"
    )
    logger.info(
        f"Forecast: {forecast_length} steps ({forecast_length / STEPS_PER_HOUR:.1f} hours)"
    )

    # Log CLI override info
    if args.context_length is not None:
        if args.context_length != context_length:
            logger.info(
                f"CLI context_length={args.context_length} was ignored (using checkpoint value)"
            )
        else:
            logger.info(f"CLI context_length={args.context_length} applied")
    if args.forecast_length is not None:
        if args.forecast_length != forecast_length:
            logger.info(
                f"CLI forecast_length={args.forecast_length} was ignored (using checkpoint value)"
            )
        else:
            logger.info(
                f"CLI forecast_length={args.forecast_length} applied (override from checkpoint)"
            )

    # Load holdout data
    logger.info("\n--- Loading Holdout Data ---")
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()
    logger.info(f"Holdout patients: {list(patients)}")
    logger.info(f"Total samples: {len(holdout_data):,}")

    # Save experiment configuration for reproducibility
    save_experiment_config(
        args,
        {
            "context_length": context_length,
            "forecast_length": forecast_length,
            **config_dict,
        },
        output_path,
    )

    # Evaluate each patient
    logger.info("\n--- Running Evaluation ---")
    all_results = []
    plot_examples = []

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()
        collect_examples = len(plot_examples) < MAX_PLOT_EXAMPLES

        patient_results, examples = evaluate_patient(
            patient_id,
            patient_df,
            model,
            context_length,
            forecast_length,
            collect_examples,
        )

        if patient_results:
            all_results.append(patient_results)

        if examples:
            plot_examples.extend(examples)

    # Compute overall metrics
    overall = compute_weighted_metrics(all_results)

    if overall:
        total_episodes = sum(r["episodes"] for r in all_results)
        logger.info("\n")
        logger.info("=" * 60)
        logger.info("OVERALL RESULTS")
        logger.info("=" * 60)
        logger.info(f"RMSE: {overall['rmse']:.3f}")
        logger.info(f"MAE:  {overall['mae']:.3f}")
        logger.info(f"MAPE: {overall['mape']:.1f}%")
        logger.info(f"Total episodes: {total_episodes}")

    # Prepare results dictionary
    results = {
        "model": args.model,
        "mode": "fine-tuned" if args.checkpoint else "zero-shot",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_file": args.model_config,
            "context_length": context_length,
            "forecast_length": forecast_length,
            **config_dict,
        },
        "overall": overall,
        "per_patient": all_results,
    }

    # Save results
    save_results(results, all_results, output_path)

    # Generate plots
    if plot_examples:
        logger.info("\n--- Generating Plots ---")
        plot_forecast_examples(
            plot_examples,
            output_path,
            args.model,
            args.dataset,
            context_length,
            forecast_length,
            is_finetuned=bool(args.checkpoint),
        )

    logger.info("\n")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Use single GPU to avoid DataParallel issues
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 (Blackwell)
    main()
