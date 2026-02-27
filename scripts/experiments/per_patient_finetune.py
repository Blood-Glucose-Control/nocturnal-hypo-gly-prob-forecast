#!/usr/bin/env python3
"""
Per-Patient Fine-Tuning Script (Stage 2).

Implements the two-stage fine-tuning protocol for personalized glucose forecasting:
  - Stage 1: Population model trained via the holdout workflow (external)
  - Stage 2: Per-patient adaptation of the Stage 1 checkpoint (this script)

Uses LOPO (leave-one-patient-out) evaluation: the target patient must be in the
holdout set (excluded from Stage 1 training). Improvement from Stage 2 is then
attributable solely to personalization, not data leakage.

Usage:
    python scripts/experiments/per_patient_finetune.py \\
        --model ttm \\
        --checkpoint experiments/holdout_ttm/output/ \\
        --dataset brown_2019 \\
        --patient-id bro_01 \\
        --val-days 7 \\
        --test-days 7 \\
        --lora-rank 8 \\
        --num-epochs 50

Note on LoRA:
    Models that do not support LoRA will fall back to full fine-tuning.
    Transformer-based models (e.g., Chronos-Bolt) will use LoRA as configured.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.utils import get_patient_column
from src.data.versioning.dataset_registry import DatasetRegistry
from src.evaluation import (
    evaluate_nocturnal_forecasting,
    plot_best_worst_episodes,
    plot_stage_comparison_auto,
)
from src.models import create_model_and_config
from src.models.base.base_model import LoRAConfig
from src.utils import get_git_commit_hash, setup_file_logging

# Constants
STEPS_PER_DAY = 288  # 5-min intervals × 24h
SAMPLING_INTERVAL_MINUTES = 5
STEPS_PER_HOUR = 60 // SAMPLING_INTERVAL_MINUTES
MIN_FINETUNE_DAYS = 14  # Hard minimum for fine-tune window
MIN_TOTAL_DAYS = 21     # Warn if patient has fewer than this

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def temporal_split(
    patient_df: pd.DataFrame,
    val_days: int,
    test_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a single patient's DataFrame chronologically into three windows.

    Uses datetime-based splitting (robust to CGM gaps).

    Args:
        patient_df: Patient data sorted by datetime (DatetimeIndex or datetime column).
        val_days: Days to reserve for validation / early stopping.
        test_days: Days to reserve for final evaluation (never used in training).

    Returns:
        (finetune_df, val_df, test_df)

    Raises:
        ValueError: If the fine-tune window has < MIN_FINETUNE_DAYS of data.
    """
    # Ensure datetime index
    df = patient_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next(
            (c for c in ["datetime", "time", "timestamp"] if c in df.columns), None
        )
        if time_col is None:
            raise ValueError("No datetime column found in patient data")
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df = df.sort_index()

    last_ts = df.index[-1]
    cutoff_test = last_ts - timedelta(days=test_days)
    cutoff_val = last_ts - timedelta(days=test_days + val_days)

    test_df = df[df.index > cutoff_test].reset_index(names="datetime")
    val_df = df[(df.index > cutoff_val) & (df.index <= cutoff_test)].reset_index(names="datetime")
    finetune_df = df[df.index <= cutoff_val].reset_index(names="datetime")

    finetune_days = (df[df.index <= cutoff_val].index[-1] - df.index[0]).total_seconds() / 86400
    if finetune_days < MIN_FINETUNE_DAYS:
        raise ValueError(
            f"Fine-tune window only covers {finetune_days:.1f} days "
            f"(minimum {MIN_FINETUNE_DAYS} days required). "
            f"Reduce --val-days or --test-days, or use a patient with more data."
        )

    logger.info(
        "Temporal split: fine-tune=%d rows (%.1f days), val=%d rows (%d days), test=%d rows (%d days)",
        len(finetune_df),
        finetune_days,
        len(val_df),
        val_days,
        len(test_df),
        test_days,
    )
    return finetune_df, val_df, test_df


def verify_lopo(
    patient_id: Any,
    holdout_data: pd.DataFrame,
    patient_col: str,
) -> None:
    """Verify that the target patient is in the holdout set (LOPO protocol).

    Raises:
        ValueError: If patient not found in holdout data.
    """
    holdout_patients = set(holdout_data[patient_col].unique())
    # Try both str and original type
    if patient_id not in holdout_patients and str(patient_id) not in holdout_patients:
        raise ValueError(
            f"Patient '{patient_id}' is NOT in the holdout set for this dataset.\n"
            f"Holdout patients: {sorted(str(p) for p in holdout_patients)}\n\n"
            "LOPO protocol requires the target patient to be excluded from Stage 1 "
            "training. Either:\n"
            "  1. Choose a patient from the holdout set listed above, or\n"
            "  2. Retrain Stage 1 with this patient explicitly excluded."
        )
    logger.info("LOPO check passed: patient '%s' is in holdout set.", patient_id)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def setup_output_dir(
    model_type: str,
    dataset: str,
    patient_id: Any,
    output_dir: Optional[str],
) -> Path:
    """Create and return the output directory."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        output_dir = (
            f"experiments/per_patient_finetune/"
            f"{model_type}/{dataset}/{patient_id}/{timestamp}"
        )
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_experiment_config(args: argparse.Namespace, output_path: Path) -> None:
    """Save experiment arguments for reproducibility."""
    config = {
        "script": "scripts/experiments/per_patient_finetune.py",
        "cli_args": vars(args),
        "environment": {
            "git_commit": get_git_commit_hash(),
            "python_version": sys.version.split()[0],
            "timestamp": datetime.now().isoformat(),
        },
        "protocol": {
            "name": "two_stage_lopo_finetune",
            "stage1": "population model trained on all holdout patients (external)",
            "stage2": "per-patient fine-tuning on holdout patient data",
        },
    }
    config_file = output_path / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Experiment config saved to: %s", config_file)


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save stage 1 vs stage 2 results JSON."""
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to: %s", results_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-patient Stage 2 fine-tuning on a holdout patient. "
            "Requires a Stage 1 checkpoint trained via the holdout workflow."
        )
    )
    parser.add_argument("--model", type=str, required=True, choices=["ttm", "sundial", "chronos2"],
                        help="Model type (must match checkpoint)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. brown_2019)")
    parser.add_argument("--patient-id", type=str, required=True,
                        help="Target patient ID (must be in holdout set)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    parser.add_argument("--config-dir", type=str, default="configs/data/holdout_5pct",
                        help="Holdout config directory")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (default: 8; skipped for non-transformer models)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                        help="Stage 2 learning rate (default: 3e-5)")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Maximum training epochs for TTM (default: 50 with early stopping)")
    parser.add_argument("--fine-tune-steps", type=int, default=5000,
                        help="Fine-tuning steps for Chronos2 (default: 5000)")
    parser.add_argument("--val-days", type=int, default=7,
                        help="Days reserved for validation (default: 7)")
    parser.add_argument("--test-days", type=int, default=7,
                        help="Days reserved for final test evaluation (default: 7)")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Context window length in steps (uses checkpoint value by default)")
    parser.add_argument("--forecast-length", type=int, default=72,
                        help="Forecast horizon in steps (default: 72 = 6h nocturnal)")
    parser.add_argument("--covariate-cols", type=str, nargs="*", default=None,
                        help="Covariate column names for nocturnal eval (e.g. iob cob)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    # ── Output setup ────────────────────────────────────────────────────────
    output_path = setup_output_dir(
        args.model, args.dataset, args.patient_id, args.output_dir
    )
    log_file = setup_file_logging(output_path, "finetune.log")

    logger.info("=" * 65)
    logger.info("PER-PATIENT STAGE 2 FINE-TUNING")
    logger.info("=" * 65)
    logger.info("Model:       %s", args.model)
    logger.info("Checkpoint:  %s", args.checkpoint)
    logger.info("Dataset:     %s", args.dataset)
    logger.info("Patient:     %s", args.patient_id)
    logger.info("Output:      %s", output_path)
    logger.info("Log:         %s", log_file)

    save_experiment_config(args, output_path)

    # ── Load holdout data + LOPO check ──────────────────────────────────────
    logger.info("\n--- Loading Holdout Data ---")
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = get_patient_column(holdout_data)
    verify_lopo(args.patient_id, holdout_data, patient_col)

    # Filter to target patient (args.patient_id is always str from argparse)
    patient_df = holdout_data[holdout_data[patient_col] == args.patient_id].copy()

    total_days = (
        pd.to_datetime(patient_df["datetime" if "datetime" in patient_df.columns else patient_df.columns[0]]).max()
        - pd.to_datetime(patient_df["datetime" if "datetime" in patient_df.columns else patient_df.columns[0]]).min()
    ).total_seconds() / 86400

    logger.info(
        "Patient '%s': %d rows, %.1f days of data",
        args.patient_id,
        len(patient_df),
        total_days,
    )
    if total_days < MIN_TOTAL_DAYS:
        logger.warning(
            "Patient has only %.1f days of data (< %d). "
            "Fine-tuning results may be unreliable.",
            total_days,
            MIN_TOTAL_DAYS,
        )

    # ── Temporal split ───────────────────────────────────────────────────────
    logger.info("\n--- Temporal Split ---")
    finetune_df, val_df, test_df = temporal_split(
        patient_df, val_days=args.val_days, test_days=args.test_days
    )

    # Build training_input (fine-tune + val) and compute split fractions
    # We pass train+val to fit(); the model's split_config controls where val begins.
    training_input = pd.concat([finetune_df, val_df], ignore_index=True)
    val_frac = len(val_df) / len(training_input)
    train_frac = 1.0 - val_frac
    stage2_split_config = {"train": round(train_frac, 4), "val": round(val_frac, 4), "test": 0.0}
    logger.info(
        "Stage 2 split config: train=%.3f (%.0f rows), val=%.3f (%.0f rows), test excluded",
        train_frac,
        len(finetune_df),
        val_frac,
        len(val_df),
    )

    # ── Load Stage 1 checkpoint ──────────────────────────────────────────────
    logger.info("\n--- Loading Stage 1 Checkpoint ---")
    kwargs: Dict[str, Any] = {}
    if args.context_length is not None:
        kwargs["context_length"] = args.context_length
    if args.forecast_length is not None:
        kwargs["forecast_length"] = args.forecast_length

    model, config = create_model_and_config(
        args.model, checkpoint=args.checkpoint, **kwargs
    )
    context_length = config.context_length
    forecast_length = config.forecast_length

    logger.info("Loaded %s checkpoint", args.model.upper())
    logger.info("  context_length:  %d steps (%.1fh)", context_length, context_length / STEPS_PER_HOUR)
    logger.info("  forecast_length: %d steps (%.1fh)", forecast_length, forecast_length / STEPS_PER_HOUR)

    # ── Stage 1 baseline evaluation ──────────────────────────────────────────
    # Build eval DataFrame with patient ID column (required by evaluate_nocturnal_forecasting).
    # Reuse for both Stage 1 and Stage 2 so the test window is identical.
    eval_test_df = test_df.copy()
    eval_test_df["p_num"] = str(args.patient_id)

    logger.info("\n--- Stage 1 Baseline Evaluation (on test window) ---")
    stage1_results = evaluate_nocturnal_forecasting(
        model=model,
        holdout_data=eval_test_df,
        context_length=context_length,
        forecast_length=forecast_length,
        covariate_cols=args.covariate_cols,
    )
    stage1_rmse = stage1_results.get("overall_rmse", float("nan"))
    logger.info(
        "Stage 1 nocturnal RMSE: %.4f mmol/L (%d midnight episodes)",
        stage1_rmse,
        stage1_results["total_episodes"],
    )

    # ── Configure Stage 2 ────────────────────────────────────────────────────
    logger.info("\n--- Configuring Stage 2 Fine-Tuning ---")

    # Set LoRA on the model instance (applied inside fit() via _enable_lora()).
    # Only enable for models that declare LoRA support — TTM is MLP-based
    # (supports_lora = False) and enabling LoRA on unsupported models would
    # cause a TypeError in _enable_lora() due to a property/call mismatch.
    if model.supports_lora:
        model.lora_config = LoRAConfig(
            enabled=True,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=0.1,
            auto_detect_modules=True,
        )
        logger.info("LoRA configured: rank=%d, alpha=%d", args.lora_rank, args.lora_alpha)
    else:
        logger.info(
            "Note: %s does not support LoRA (not a transformer model). "
            "Proceeding with full fine-tuning of all parameters.",
            model.__class__.__name__,
        )

    # Update model config for Stage 2
    # Chronos2 uses fine_tune_lr / fine_tune_steps (AutoGluon hyperparams);
    # TTM and others use learning_rate / num_epochs (base class fields).
    model.config.training_mode = "fine_tune"
    if hasattr(model.config, "fine_tune_lr"):
        model.config.fine_tune_lr = args.learning_rate
        model.config.fine_tune_steps = args.fine_tune_steps
        logger.info("Stage 2 LR:     %g", args.learning_rate)
        logger.info("Stage 2 steps:  %d", args.fine_tune_steps)
    else:
        model.config.learning_rate = args.learning_rate
        model.config.num_epochs = args.num_epochs
        logger.info("Stage 2 LR:     %g", args.learning_rate)
        logger.info("Stage 2 epochs: %d (with early stopping)", args.num_epochs)

    # Update split config if the model uses one (TTM reads split_config to
    # partition train+val internally). Chronos2 ignores this — AutoGluon
    # handles validation via num_val_windows (last prediction_length steps).
    if hasattr(model.config, "split_config"):
        model.config.split_config = stage2_split_config
        logger.info("Updated split_config: %s", stage2_split_config)
    else:
        logger.info(
            "Model does not use split_config; backend handles validation internally"
        )

    # ── Stage 2 fine-tuning ──────────────────────────────────────────────────
    logger.info("\n--- Stage 2 Fine-Tuning ---")
    finetune_checkpoint_dir = str(output_path / "stage2_checkpoint")

    finetune_metrics = model.fit(
        train_data=training_input,
        output_dir=finetune_checkpoint_dir,
    )
    logger.info("Fine-tuning complete. Train metrics: %s", finetune_metrics.get("train_metrics", {}))

    # ── Save Stage 2 checkpoint ──────────────────────────────────────────────
    logger.info("\n--- Saving Stage 2 Checkpoint ---")
    model.save(finetune_checkpoint_dir)
    logger.info("Stage 2 checkpoint saved to: %s", finetune_checkpoint_dir)

    # ── Stage 2 evaluation ───────────────────────────────────────────────────
    logger.info("\n--- Stage 2 Evaluation (on test window) ---")
    stage2_results = evaluate_nocturnal_forecasting(
        model=model,
        holdout_data=eval_test_df,
        context_length=context_length,
        forecast_length=forecast_length,
        covariate_cols=args.covariate_cols,
    )
    stage2_rmse = stage2_results.get("overall_rmse", float("nan"))
    logger.info(
        "Stage 2 nocturnal RMSE: %.4f mmol/L (%d midnight episodes)",
        stage2_rmse,
        stage2_results["total_episodes"],
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    rmse_delta = stage1_rmse - stage2_rmse  # positive = improvement
    logger.info("\n")
    logger.info("=" * 65)
    logger.info("RESULTS SUMMARY  — patient: %s", args.patient_id)
    logger.info("=" * 65)
    logger.info("  Stage 1 RMSE (population):   %.4f mmol/L", stage1_rmse)
    logger.info("  Stage 2 RMSE (personalized): %.4f mmol/L", stage2_rmse)
    if not np.isnan(rmse_delta):
        direction = "improvement" if rmse_delta > 0 else "regression"
        logger.info(
            "  Delta:                       %.4f mmol/L (%s)",
            abs(rmse_delta),
            direction,
        )

    # Plots
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    if stage1_results.get("per_episode") and stage2_results.get("per_episode"):
        plot_stage_comparison_auto(
            stage1_per_episode=stage1_results["per_episode"],
            stage2_per_episode=stage2_results["per_episode"],
            stage1_rmse=stage1_rmse,
            stage2_rmse=stage2_rmse,
            output_path=plots_dir,
            model_name=args.model,
            dataset_name=args.dataset,
            patient_id=args.patient_id,
        )

        plot_best_worst_episodes(
            per_episode=stage2_results["per_episode"],
            output_path=plots_dir,
            model_name=args.model,
            dataset_name=args.dataset,
            is_finetuned=True,
        )

    # Save consolidated results JSON
    full_results = {
        "patient_id": args.patient_id,
        "dataset": args.dataset,
        "model": args.model,
        "stage1_checkpoint": args.checkpoint,
        "stage2_checkpoint": finetune_checkpoint_dir,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "context_length": context_length,
            "forecast_length": forecast_length,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "val_days": args.val_days,
            "test_days": args.test_days,
            "split_config": stage2_split_config,
        },
        "data_split": {
            "finetune_rows": len(finetune_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
        "stage1": {
            "overall_rmse": stage1_rmse,
            "total_episodes": stage1_results.get("total_episodes", 0),
            "per_patient": stage1_results.get("per_patient", []),
        },
        "stage2": {
            "overall_rmse": stage2_rmse,
            "total_episodes": stage2_results.get("total_episodes", 0),
            "per_patient": stage2_results.get("per_patient", []),
            "finetune_metrics": finetune_metrics,
        },
        "delta": {
            "rmse": rmse_delta,
            "direction": "improvement" if rmse_delta > 0 else ("regression" if rmse_delta < 0 else "neutral"),
        },
    }
    save_results(full_results, output_path)

    logger.info("=" * 65)
    logger.info("DONE — output: %s", output_path)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
