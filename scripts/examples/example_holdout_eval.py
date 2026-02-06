#!/usr/bin/env python3
"""
Model-Agnostic Holdout Evaluation Script.

Standardized benchmarking script for fair model comparison.
Uses fixed-size episodes (context_length + forecast_length) for consistent
evaluation across all models.

Usage:
    python scripts/examples/example_holdout_eval.py --model sundial --dataset kaggle_brisT1D
    python scripts/examples/example_holdout_eval.py --model ttm --dataset kaggle_brisT1D
    python scripts/examples/example_holdout_eval.py --model sundial --context-length 512 --forecast-length 72
    python scripts/examples/example_holdout_eval.py --model sundial --config configs/models/sundial.yaml
    python scripts/examples/example_holdout_eval.py --model sundial --checkpoint path/to/checkpoint
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.data.versioning.dataset_registry import DatasetRegistry
from src.models.base import BaseTimeSeriesFoundationModel, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load model config from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary of config parameters
    """
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def create_model_and_config(
    model_type: str,
    checkpoint: str = None,
    **kwargs
) -> Tuple[BaseTimeSeriesFoundationModel, ModelConfig]:
    """Factory function to create model and config based on type.

    Args:
        model_type: One of 'sundial', 'ttm', 'chronos', 'moirai'
        checkpoint: Optional path to fine-tuned checkpoint
        **kwargs: Additional config parameters (e.g., num_samples, forecast_length)

    Returns:
        Tuple of (model, config)
    """
    if model_type == "sundial":
        from src.models.sundial import SundialForecaster, SundialConfig
        config = SundialConfig(
            num_samples=kwargs.get("num_samples", 100)
        )
        model = SundialForecaster(config)
        if checkpoint:
            model._load_checkpoint(checkpoint)
        return model, config

    # Placeholder for future models
    elif model_type == "chronos":
        raise NotImplementedError("Chronos model not yet implemented")

    elif model_type == "moirai":
        raise NotImplementedError("Moirai model not yet implemented")

    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: sundial, ttm, chronos, moirai")


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
        episode = patient_df.iloc[start:start + total_length]
        context = episode.iloc[:context_length]
        target = episode["bg_mM"].values[context_length:]
        yield context, target


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute evaluation metrics.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        Dictionary with rmse, mae, mape, mse
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape, "mse": mse}


def main():
    parser = argparse.ArgumentParser(
        description="Model-agnostic holdout evaluation for standardized benchmarking"
    )
    parser.add_argument(
        "--model", type=str, default="sundial",
        choices=["sundial", "chronos", "moirai"],
        help="Model type to use (ttm not yet supported - use example_holdout_ttm_workflow.py)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to model config YAML file (optional, CLI args override)"
    )
    parser.add_argument(
        "--dataset", type=str, default="kaggle_brisT1D",
        help="Dataset name"
    )
    parser.add_argument(
        "--config-dir", type=str, default="configs/data/holdout_5pct",
        help="Holdout config directory"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to fine-tuned checkpoint (if None, uses zero-shot)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of samples for probabilistic models"
    )
    parser.add_argument(
        "--context-length", type=int, default=512,
        help="Context window length in steps (default: 512)"
    )
    parser.add_argument(
        "--forecast-length", type=int, default=72,
        help="Forecast horizon in steps (default: 72 = 6 hours at 5-min intervals)"
    )
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        mode = "finetuned" if args.checkpoint else "zeroshot"
        args.output_dir = f"./trained_models/artifacts/{args.model}_eval/{timestamp}_{args.dataset}_{mode}"

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    context_length = args.context_length
    forecast_length = args.forecast_length

    logger.info("=" * 60)
    logger.info("MODEL-AGNOSTIC HOLDOUT EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {'Fine-tuned' if args.checkpoint else 'Zero-shot'}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Context: {context_length} steps ({context_length / 12:.1f} hours)")
    logger.info(f"Forecast: {forecast_length} steps ({forecast_length / 12:.1f} hours)")
    logger.info(f"Output: {args.output_dir}")

    # Load holdout data
    logger.info("\n--- Loading Holdout Data ---")
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = "p_num" if "p_num" in holdout_data.columns else "id"
    patients = holdout_data[patient_col].unique()
    logger.info(f"Holdout patients: {list(patients)}")
    logger.info(f"Total samples: {len(holdout_data):,}")

    # Load config from file if provided, then override with CLI args
    config_dict = load_config(args.config) if args.config else {}
    # CLI args override config file values
    if args.num_samples is not None:
        config_dict["num_samples"] = args.num_samples
    config_dict["forecast_length"] = forecast_length

    # Initialize model
    logger.info(f"\n--- Initializing {args.model.upper()} ---")
    if args.config:
        logger.info(f"Config file: {args.config}")
    model, _ = create_model_and_config(
        args.model,
        checkpoint=args.checkpoint,
        **config_dict
    )

    # Evaluate each patient
    logger.info("\n--- Running Evaluation ---")
    all_results = []
    plot_examples = []  # Store examples for plotting

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

        patient_preds = []
        patient_targets = []
        patient_contexts = []

        for context_df, target in iter_episodes(patient_df, context_length, forecast_length):
            context_values = context_df["bg_mM"].values

            # Skip episodes with NaN values
            if np.isnan(context_values).any() or np.isnan(target).any():
                continue

            # Predict using unified interface
            pred = model.predict(context_df, forecast_length)
            patient_preds.append(pred)
            patient_targets.append(target)
            patient_contexts.append(context_values)

            # Store first valid example per patient for plotting
            if len(plot_examples) < 8 and len(patient_preds) == 1:
                plot_examples.append({
                    "patient": patient_id,
                    "context": context_values,
                    "target": target,
                    "pred": pred,
                })

        if not patient_preds:
            logger.warning(f"  {patient_id}: No valid windows")
            continue

        # Aggregate patient metrics
        preds = np.concatenate(patient_preds)
        targets = np.concatenate(patient_targets)
        metrics = compute_metrics(preds, targets)

        all_results.append({
            "patient": patient_id,
            "episodes": len(patient_preds),
            **metrics,
        })

        logger.info(f"  {patient_id}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f} ({len(patient_preds)} episodes)")

    # Overall metrics (weighted by episodes)
    overall = {}
    if all_results:
        total_episodes = sum(r['episodes'] for r in all_results)
        overall_rmse = np.sqrt(sum(r['rmse']**2 * r['episodes'] for r in all_results) / total_episodes)
        overall_mae = sum(r['mae'] * r['episodes'] for r in all_results) / total_episodes
        overall_mape = sum(r['mape'] * r['episodes'] for r in all_results) / total_episodes
        overall = {"rmse": overall_rmse, "mae": overall_mae, "mape": overall_mape, "mse": overall_rmse**2}

        logger.info("\n" + "=" * 60)
        logger.info("OVERALL RESULTS")
        logger.info("=" * 60)
        logger.info(f"RMSE: {overall['rmse']:.3f}")
        logger.info(f"MAE:  {overall['mae']:.3f}")
        logger.info(f"MAPE: {overall['mape']:.1f}%")
        logger.info(f"Total episodes: {total_episodes}")

    # Save results
    results = {
        "model": args.model,
        "mode": "fine-tuned" if args.checkpoint else "zero-shot",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "config_file": args.config,
            "num_samples": config_dict.get("num_samples", args.num_samples),
            "context_length": context_length,
            "forecast_length": forecast_length,
            **{k: v for k, v in config_dict.items() if k not in ["num_samples", "forecast_length"]},
        },
        "overall": overall,
        "per_patient": all_results,
    }

    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Save CSV
    if all_results:
        pd.DataFrame(all_results).to_csv(output_path / "patient_metrics.csv", index=False)

    # Generate plots
    if plot_examples:
        logger.info("\n--- Generating Plots ---")
        n = len(plot_examples)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        # Show last 3 hours of context + forecast
        context_hours_to_show = 3
        context_steps_to_show = context_hours_to_show * 12

        for i, ex in enumerate(plot_examples):
            ax = axes[i]
            ctx_full, tgt, pred = ex["context"], ex["target"], ex["pred"]

            # Truncate context to last N hours for visualization
            ctx = ctx_full[-context_steps_to_show:] if len(ctx_full) > context_steps_to_show else ctx_full

            # Time axis in hours (0 = forecast start)
            t_ctx = (np.arange(len(ctx)) - len(ctx)) / 12
            t_pred = np.arange(len(tgt)) / 12

            # Plot BG
            ax.plot(t_ctx, ctx, "b-", lw=1.5, label="BG (context)")
            ax.plot(t_pred, tgt, "g-", lw=2, label="BG (actual)")
            ax.plot(t_pred, pred, "r--", lw=2, alpha=0.8, label="BG (forecast)")

            ax.axvline(0, color="gray", ls=":", lw=1.5, label="Forecast start")
            ax.axhline(3.9, color="crimson", ls="--", alpha=0.4, lw=1, label="Hypo threshold")

            ax.set_ylabel("BG (mmol/L)", fontsize=9)
            ax.set_ylim(0, 18)

            rmse = compute_metrics(pred, tgt)["rmse"]
            ax.set_title(f"{ex['patient']}: RMSE={rmse:.2f} mmol/L", fontsize=10)
            ax.set_xlabel("Hours", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        # Legend for first plot
        axes[0].legend(fontsize=7, loc="upper right")

        mode_str = "Fine-tuned" if args.checkpoint else "Zero-Shot"
        fig.suptitle(
            f"{args.model.upper()} {mode_str} Forecasts - {args.dataset}\n"
            f"(Context: {context_length} steps, Forecast: {forecast_length} steps)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()

        plot_file = output_path / "forecasts.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved to: {plot_file}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
