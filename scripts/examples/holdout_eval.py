#!/usr/bin/env python3
"""
Model-Agnostic Holdout Evaluation Script.

Standardized benchmarking script for fair model comparison.
Uses fixed-size episodes (context_length + forecast_length) for consistent
evaluation across all models.

Usage:
    python scripts/examples/holdout_eval.py --model sundial --dataset brown_2019
    python scripts/examples/holdout_eval.py --model ttm --dataset brown_2019
    python scripts/examples/holdout_eval.py --model sundial --context-length 512 --forecast-length 72
    python scripts/examples/holdout_eval.py --model sundial --model-config configs/models/sundial.yaml
    python scripts/examples/holdout_eval.py --model sundial --checkpoint path/to/checkpoint
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.data.versioning.dataset_registry import DatasetRegistry
from src.models.base import BaseTimeSeriesFoundationModel, ModelConfig
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.episode_builders import build_midnight_episodes

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


def setup_file_logging(output_path: Path) -> Path:
    """Add file handler to logger for saving logs to output directory.

    Args:
        output_path: Directory to save log file

    Returns:
        Path to the log file
    """
    log_file = output_path / "evaluation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    # Add to root logger so all loggers write to file
    logging.getLogger().addHandler(file_handler)
    return log_file


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
    model_type: str, checkpoint: Optional[str] = None, **kwargs
) -> Tuple[BaseTimeSeriesFoundationModel, ModelConfig]:
    """Factory function to create model and config based on type.

    Args:
        model_type: One of 'sundial', 'ttm', 'chronos', 'tide', 'moirai'
        checkpoint: Optional path to fine-tuned checkpoint
        **kwargs: Additional config parameters (e.g., num_samples, forecast_length)

    Returns:
        Tuple of (model, config)

    Note:
        When checkpoint is provided, the saved config is used as base.
        CLI overrides are validated:
        - batch_size: always allowed (inference-only setting)
        - forecast_length: allowed if <= saved value (truncate predictions)
        - context_length: must match saved value (affects model architecture)
    """
    if model_type == "sundial":
        from src.models.sundial import SundialForecaster, SundialConfig

        if checkpoint:
            # Load model with saved config
            model = SundialForecaster.load(checkpoint)
            config = model.config

            # Apply valid overrides
            if "batch_size" in kwargs:
                config.batch_size = kwargs["batch_size"]
            if "forecast_length" in kwargs:
                requested = kwargs["forecast_length"]
                if requested <= config.forecast_length:
                    logger.info(
                        f"Overriding forecast_length: {config.forecast_length} -> {requested}"
                    )
                    config.forecast_length = requested
                else:
                    logger.warning(
                        f"Cannot increase forecast_length beyond trained value "
                        f"({config.forecast_length}). Using saved value."
                    )
        else:
            config = SundialConfig(
                forecast_length=kwargs.get("forecast_length", 96),
                num_samples=kwargs.get("num_samples", 100),
            )
            model = SundialForecaster(config)
        return model, config

    elif model_type == "ttm":
        from src.models.ttm import TTMForecaster, TTMConfig
        from src.models.base.base_model import ModelConfig
        import dataclasses

        if checkpoint:
            # Load config from training_metadata.json (config.json is overwritten by TSFM)
            metadata_path = os.path.join(checkpoint, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                saved_config = metadata.get("config", {})
            else:
                logger.warning(
                    f"No training_metadata.json found in {checkpoint}, using defaults"
                )
                saved_config = {}

            # Get valid field names from ModelConfig (TTMConfig uses **kwargs)
            base_fields = {f.name for f in dataclasses.fields(ModelConfig)}
            # TTM-specific fields handled by TTMConfig.__init__
            ttm_fields = {
                "scaler_type",
                "input_features",
                "target_features",
                "split_config",
                "fewshot_percent",
                "num_input_channels",
                "num_output_channels",
                "prediction_filter_length",
                "resolution_min",
                "use_tracking_callback",
                "find_optimal_lr",
                "logging_dir",
            }
            valid_fields = base_fields | ttm_fields | {"model_path", "training_mode"}

            # Filter to only known fields
            filtered_config = {
                k: v for k, v in saved_config.items() if k in valid_fields
            }
            filtered_config["model_path"] = (
                checkpoint  # Override to load from checkpoint
            )
            filtered_config["training_mode"] = "fine_tune"

            # Log any ignored fields
            ignored = set(saved_config.keys()) - valid_fields
            if ignored:
                logger.debug(
                    f"Ignoring unknown config fields from checkpoint: {ignored}"
                )

            config = TTMConfig(**filtered_config)

            # Apply valid overrides
            if "batch_size" in kwargs:
                config.batch_size = kwargs["batch_size"]

            if "forecast_length" in kwargs:
                requested = kwargs["forecast_length"]
                if requested <= config.forecast_length:
                    logger.info(
                        f"Overriding forecast_length: {config.forecast_length} -> {requested}"
                    )
                    config.forecast_length = requested
                else:
                    logger.warning(
                        f"Cannot increase forecast_length beyond trained value "
                        f"({config.forecast_length}). Using saved value."
                    )

            if "context_length" in kwargs:
                requested = kwargs["context_length"]
                if requested != config.context_length:
                    logger.warning(
                        f"context_length mismatch: requested {requested}, "
                        f"model trained with {config.context_length}. "
                        f"Using saved value."
                    )

            # Create model with config and load checkpoint
            model = TTMForecaster(config)
            model._load_checkpoint(checkpoint)
            model.is_fitted = True
        else:
            config = TTMConfig(
                model_path=kwargs.get(
                    "model_path", "ibm-granite/granite-timeseries-ttm-r2"
                ),
                context_length=kwargs.get("context_length", 512),
                forecast_length=kwargs.get("forecast_length", 96),
                batch_size=kwargs.get("batch_size", 256),
                training_mode="zero_shot",
                freeze_backbone=True,
            )
            model = TTMForecaster(config)
        return model, config

    elif model_type == "chronos":
        raise NotImplementedError("Chronos model not yet implemented")

    elif model_type == "tide":
        from src.models.tide import TiDEForecaster, TiDEConfig

        if checkpoint:
            model = TiDEForecaster.load(checkpoint)
            config = model.config

            if "batch_size" in kwargs:
                config.batch_size = kwargs["batch_size"]
            if "forecast_length" in kwargs:
                requested = kwargs["forecast_length"]
                if requested <= config.forecast_length:
                    config.forecast_length = requested
                else:
                    logger.warning(
                        f"Cannot increase forecast_length beyond trained value "
                        f"({config.forecast_length}). Using saved value."
                    )
        else:
            config = TiDEConfig(
                context_length=kwargs.get("context_length", 512),
                forecast_length=kwargs.get("forecast_length", 72),
            )
            model = TiDEForecaster(config)
        return model, config

    elif model_type == "moirai":
        raise NotImplementedError("Moirai model not yet implemented")

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: sundial, ttm, chronos, tide, moirai"
        )


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


# ---------------------------------------------------------------------------
# Midnight-Anchored Evaluation (Nocturnal Hypoglycemia Task)
# ---------------------------------------------------------------------------
#
# Sliding-window eval (iter_episodes above) measures general forecast accuracy.
# Midnight-anchored eval measures nocturnal hypo task performance: context ends
# at midnight, 6h forecast covers the overnight window.
# ---------------------------------------------------------------------------


def evaluate_nocturnal_forecasting(
    model: BaseTimeSeriesFoundationModel,
    holdout_data: pd.DataFrame,
    context_length: int,
    forecast_length: int,
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
    interval_mins: int = SAMPLING_INTERVAL_MINUTES,
) -> Dict[str, Any]:
    """Evaluate model on midnight-anchored nocturnal forecasting task.

    Builds midnight episodes per patient, stacks them into a panel DataFrame
    with ``episode_id``, and calls model.predict() once. Covariates (IOB)
    are included in the context window only — no future covariates are passed
    to avoid data leakage.

    Args:
        model: Model implementing predict(data, **kwargs) -> pd.DataFrame.
        holdout_data: Flat DataFrame with all holdout patients.
        context_length: Context window size in steps.
        forecast_length: Forecast horizon in steps.
        target_col: BG column name.
        covariate_cols: Covariate column names (e.g., ["iob"]).
        interval_mins: Sampling interval in minutes.

    Returns:
        Dict with overall_rmse, total_episodes, per_patient, per_episode.
    """
    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()

    # --- Phase 1: Build episodes for all patients ---
    # Collect episodes and track which patient each belongs to.
    episode_metadata = []
    context_dfs = []

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

        # Set DatetimeIndex if not already set
        if not isinstance(patient_df.index, pd.DatetimeIndex):
            time_col = "datetime" if "datetime" in patient_df.columns else None
            if time_col:
                patient_df[time_col] = pd.to_datetime(patient_df[time_col])
                patient_df = patient_df.set_index(time_col).sort_index()
            else:
                logger.warning(
                    "Patient %s: no datetime column or DatetimeIndex, skipping",
                    patient_id,
                )
                continue

        episodes, _stats = build_midnight_episodes(
            patient_df,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            covariate_cols=covariate_cols,
            interval_mins=interval_mins,
        )

        if not episodes:
            logger.info("  Patient %s: no valid midnight episodes", patient_id)
            continue

        for i, ep in enumerate(episodes):
            ep_id = f"{patient_id}_ep{i:03d}"

            ctx = ep["context_df"].copy()
            ctx["episode_id"] = ep_id
            context_dfs.append(ctx)

            episode_metadata.append(
                {
                    "episode_id": ep_id,
                    "patient_id": str(patient_id),
                    "anchor": ep["anchor"],
                    "target_bg": ep["target_bg"],
                    "context_bg": ep["context_df"][target_col].to_numpy(),
                }
            )

    if not episode_metadata:
        logger.warning("No valid midnight episodes found across all patients")
        return {
            "overall_rmse": float("nan"),
            "total_episodes": 0,
            "per_patient": [],
            "per_episode": [],
        }

    # --- Phase 2: Stack and call predict() once ---
    stacked_context = pd.concat(context_dfs)

    logger.info(
        "Calling predict() with %d stacked episodes (%d rows)",
        len(episode_metadata),
        len(stacked_context),
    )

    predictions = model.predict(stacked_context)

    # --- Phase 3: Unpack predictions by episode_id and compute metrics ---
    all_episode_results = []
    patient_episodes = {}  # patient_id -> list of (pred, target) tuples

    # Build lookup: episode_id -> target_bg and patient_id from metadata
    meta_by_ep = {m["episode_id"]: m for m in episode_metadata}

    for ep_id, group in predictions.groupby("episode_id"):
        meta = meta_by_ep[ep_id]
        # Contract: predict() returns a DataFrame with a column named after
        # the target (e.g., "bg_mM"). Models rename internally if needed.
        pred = group[target_col].to_numpy()
        target = meta["target_bg"]

        ep_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

        all_episode_results.append(
            {
                "patient_id": meta["patient_id"],
                "anchor": meta["anchor"].isoformat(),
                "rmse": ep_rmse,
                "pred": pred,
                "target_bg": target,
                "context_bg": meta["context_bg"],
            }
        )

        pid = meta["patient_id"]
        if pid not in patient_episodes:
            patient_episodes[pid] = []
        patient_episodes[pid].append((pred, target))

    # Per-patient aggregate
    all_patient_results = []
    for pid, ep_list in patient_episodes.items():
        preds = np.concatenate([p for p, _ in ep_list])
        targets = np.concatenate([t for _, t in ep_list])
        metrics = compute_metrics(preds, targets)

        all_patient_results.append(
            {
                "patient_id": pid,
                "episodes": len(ep_list),
                **metrics,
            }
        )

        logger.info(
            "  Patient %s: RMSE=%.3f, MAE=%.3f (%d midnight episodes)",
            pid,
            metrics["rmse"],
            metrics["mae"],
            len(ep_list),
        )

    # Overall metrics (concatenated predictions, consistent with per-patient)
    all_preds = np.concatenate([ep["pred"] for ep in all_episode_results])
    all_targets = np.concatenate([ep["target_bg"] for ep in all_episode_results])
    overall_rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    logger.info(
        "Nocturnal evaluation: %.4f RMSE over %d midnight episodes",
        overall_rmse,
        len(all_episode_results),
    )

    return {
        "overall_rmse": overall_rmse,
        "total_episodes": len(all_episode_results),
        "per_patient": all_patient_results,
        "per_episode": all_episode_results,
    }


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


def get_patient_column(df: pd.DataFrame) -> str:
    """Determine the patient identifier column name.

    Args:
        df: DataFrame containing patient data

    Returns:
        Name of the patient column

    Raises:
        ValueError: If neither expected column is found
    """
    if "p_num" in df.columns:
        return "p_num"
    elif "id" in df.columns:
        return "id"
    else:
        raise ValueError(
            f"Expected patient column 'p_num' or 'id' not found. "
            f"Available columns: {df.columns.tolist()}"
        )


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


def get_git_commit_hash() -> str:
    """Get the current git commit hash.

    Returns:
        Short git commit hash, or 'unknown' if not in a git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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
    cmd_parts = ["python", "scripts/examples/holdout_eval.py"]
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
        choices=["sundial", "ttm", "chronos", "tide", "moirai"],
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
    config_dict = load_config(args.model_config) if args.model_config else {}

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
