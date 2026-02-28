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
    # Single patient:
    python scripts/experiments/per_patient_finetune.py \\
        --model chronos2 \\
        --checkpoint models/chronos2_stage1/20260224_234112 \\
        --dataset brown_2019 \\
        --patient-id bro_92 \\
        --config-dir configs/data/holdout_10pct \\
        --fine-tune-steps 5000

    # All holdout patients (batch mode):
    python scripts/experiments/per_patient_finetune.py \\
        --model chronos2 \\
        --checkpoint models/chronos2_stage1/20260224_234112 \\
        --dataset brown_2019 \\
        --all-patients \\
        --config-dir configs/data/holdout_10pct \\
        --fine-tune-steps 5000

Note on LoRA:
    Models that do not support LoRA will fall back to full fine-tuning.
    Transformer-based models (e.g., Chronos-Bolt) will use LoRA as configured.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """Create and return the output directory for a single patient."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        output_dir = (
            f"experiments/per_patient_finetune/"
            f"{model_type}/{dataset}/{patient_id}/{timestamp}"
        )
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_batch_output_dir(
    model_type: str,
    output_dir: Optional[str],
) -> Path:
    """Create output directory for batch mode, following holdout workflow convention."""
    if output_dir is not None:
        path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        run_id = f"RID{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        path = Path(
            f"trained_models/artifacts/{model_type}/"
            f"{timestamp}_{run_id}_per_patient_finetune"
        )
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


def write_summary_csv(summary_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Write batch results summary CSV."""
    df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("Summary CSV saved to: %s", csv_path)

    # Print summary table to console
    logger.info("\n%s", df.to_string(index=False))


# ---------------------------------------------------------------------------
# Core: single-patient fine-tuning
# ---------------------------------------------------------------------------

def run_single_patient(
    args: argparse.Namespace,
    patient_id: str,
    holdout_data: pd.DataFrame,
    patient_col: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Run Stage 1 eval + Stage 2 fine-tuning + Stage 2 eval for one patient.

    Loads a fresh Stage 1 checkpoint (fine-tuning mutates the model in-place),
    evaluates it, fine-tunes on the patient's temporal split, then re-evaluates.

    Args:
        args: CLI arguments.
        patient_id: Target patient ID.
        holdout_data: Full holdout DataFrame (all patients).
        patient_col: Patient column name in holdout_data.
        output_path: Output directory for this patient.

    Returns:
        Summary dict with patient_id, stage1_rmse, stage2_rmse, delta, etc.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = setup_file_logging(output_path, "finetune.log")

    logger.info("=" * 65)
    logger.info("PER-PATIENT STAGE 2 FINE-TUNING — %s", patient_id)
    logger.info("=" * 65)
    logger.info("Model:       %s", args.model)
    logger.info("Checkpoint:  %s", args.checkpoint)
    logger.info("Dataset:     %s", args.dataset)
    logger.info("Patient:     %s", patient_id)
    logger.info("Output:      %s", output_path)
    logger.info("Log:         %s", log_file)

    save_experiment_config(args, output_path)

    # ── Verify LOPO ─────────────────────────────────────────────────────────
    verify_lopo(patient_id, holdout_data, patient_col)

    # ── Filter to target patient ────────────────────────────────────────────
    patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

    time_col = "datetime" if "datetime" in patient_df.columns else patient_df.columns[0]
    total_days = (
        pd.to_datetime(patient_df[time_col]).max()
        - pd.to_datetime(patient_df[time_col]).min()
    ).total_seconds() / 86400

    logger.info(
        "Patient '%s': %d rows, %.1f days of data",
        patient_id, len(patient_df), total_days,
    )
    if total_days < MIN_TOTAL_DAYS:
        logger.warning(
            "Patient has only %.1f days of data (< %d). "
            "Fine-tuning results may be unreliable.",
            total_days, MIN_TOTAL_DAYS,
        )

    # ── Temporal split ──────────────────────────────────────────────────────
    logger.info("\n--- Temporal Split ---")
    finetune_df, val_df, test_df = temporal_split(
        patient_df, val_days=args.val_days, test_days=args.test_days
    )

    training_input = pd.concat([finetune_df, val_df], ignore_index=True)
    val_frac = len(val_df) / len(training_input)
    train_frac = 1.0 - val_frac
    stage2_split_config = {"train": round(train_frac, 4), "val": round(val_frac, 4), "test": 0.0}
    logger.info(
        "Stage 2 split config: train=%.3f (%.0f rows), val=%.3f (%.0f rows), test excluded",
        train_frac, len(finetune_df), val_frac, len(val_df),
    )

    # ── Load Stage 1 checkpoint (fresh copy per patient) ────────────────────
    logger.info("\n--- Loading Stage 1 Checkpoint ---")
    model_kwargs: Dict[str, Any] = {}
    if args.context_length is not None:
        model_kwargs["context_length"] = args.context_length
    if args.forecast_length is not None:
        model_kwargs["forecast_length"] = args.forecast_length

    model, config = create_model_and_config(
        args.model, checkpoint=args.checkpoint, **model_kwargs
    )
    context_length = config.context_length
    forecast_length = config.forecast_length

    logger.info("Loaded %s checkpoint", args.model.upper())
    logger.info("  context_length:  %d steps (%.1fh)", context_length, context_length / STEPS_PER_HOUR)
    logger.info("  forecast_length: %d steps (%.1fh)", forecast_length, forecast_length / STEPS_PER_HOUR)

    # Auto-detect covariates from model config if not explicitly specified
    # (same logic as nocturnal_hypo_eval.py — required because AutoGluon
    # expects the same columns at predict time as were present during training)
    covariate_cols = args.covariate_cols
    if covariate_cols is None and hasattr(config, "covariate_cols") and config.covariate_cols:
        covariate_cols = config.covariate_cols
        logger.info("Using covariates from model config: %s", covariate_cols)

    # ── Stage 1 baseline evaluation ─────────────────────────────────────────
    eval_test_df = test_df.copy()
    eval_test_df["p_num"] = str(patient_id)

    # Pass return_quantiles for models that support it (e.g. Chronos2)
    extra_predict_kwargs = {}
    if args.model in ("chronos2", "chronos"):
        extra_predict_kwargs["return_quantiles"] = True

    logger.info("\n--- Stage 1 Baseline Evaluation (on test window) ---")
    stage1_results = evaluate_nocturnal_forecasting(
        model=model,
        holdout_data=eval_test_df,
        context_length=context_length,
        forecast_length=forecast_length,
        covariate_cols=covariate_cols,
        extra_predict_kwargs=extra_predict_kwargs if extra_predict_kwargs else None,
    )
    stage1_rmse = stage1_results.get("overall_rmse", float("nan"))
    logger.info(
        "Stage 1 nocturnal RMSE: %.4f mmol/L (%d midnight episodes)",
        stage1_rmse, stage1_results["total_episodes"],
    )

    # ── Configure Stage 2 ──────────────────────────────────────────────────
    logger.info("\n--- Configuring Stage 2 Fine-Tuning ---")

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

    if hasattr(model.config, "split_config"):
        model.config.split_config = stage2_split_config
        logger.info("Updated split_config: %s", stage2_split_config)
    else:
        logger.info(
            "Model does not use split_config; backend handles validation internally"
        )

    # ── Stage 2 fine-tuning ────────────────────────────────────────────────
    logger.info("\n--- Stage 2 Fine-Tuning ---")
    finetune_checkpoint_dir = str(output_path / "stage2_checkpoint")

    finetune_metrics = model.fit(
        train_data=training_input,
        output_dir=finetune_checkpoint_dir,
    )
    logger.info("Fine-tuning complete. Train metrics: %s", finetune_metrics.get("train_metrics", {}))

    # ── Save Stage 2 checkpoint ────────────────────────────────────────────
    logger.info("\n--- Saving Stage 2 Checkpoint ---")
    model.save(finetune_checkpoint_dir)
    logger.info("Stage 2 checkpoint saved to: %s", finetune_checkpoint_dir)

    # ── Stage 2 evaluation ─────────────────────────────────────────────────
    logger.info("\n--- Stage 2 Evaluation (on test window) ---")
    stage2_results = evaluate_nocturnal_forecasting(
        model=model,
        holdout_data=eval_test_df,
        context_length=context_length,
        forecast_length=forecast_length,
        covariate_cols=covariate_cols,
        extra_predict_kwargs=extra_predict_kwargs if extra_predict_kwargs else None,
    )
    stage2_rmse = stage2_results.get("overall_rmse", float("nan"))
    logger.info(
        "Stage 2 nocturnal RMSE: %.4f mmol/L (%d midnight episodes)",
        stage2_rmse, stage2_results["total_episodes"],
    )

    # ── Summary ────────────────────────────────────────────────────────────
    rmse_delta = stage1_rmse - stage2_rmse  # positive = improvement
    logger.info("\n")
    logger.info("=" * 65)
    logger.info("RESULTS SUMMARY  — patient: %s", patient_id)
    logger.info("=" * 65)
    logger.info("  Stage 1 RMSE (population):   %.4f mmol/L", stage1_rmse)
    logger.info("  Stage 2 RMSE (personalized): %.4f mmol/L", stage2_rmse)
    if not np.isnan(rmse_delta):
        direction = "improvement" if rmse_delta > 0 else "regression"
        logger.info(
            "  Delta:                       %.4f mmol/L (%s)",
            abs(rmse_delta), direction,
        )

    # ── Plots ──────────────────────────────────────────────────────────────
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
            patient_id=patient_id,
        )

        plot_best_worst_episodes(
            per_episode=stage2_results["per_episode"],
            output_path=plots_dir,
            model_name=args.model,
            dataset_name=args.dataset,
            is_finetuned=True,
        )

    # ── Save consolidated results JSON (with per_episode for re-plotting) ──
    full_results = {
        "patient_id": patient_id,
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
            "per_episode": stage1_results.get("per_episode", []),
        },
        "stage2": {
            "overall_rmse": stage2_rmse,
            "total_episodes": stage2_results.get("total_episodes", 0),
            "per_patient": stage2_results.get("per_patient", []),
            "per_episode": stage2_results.get("per_episode", []),
            "finetune_metrics": finetune_metrics,
        },
        "delta": {
            "rmse": rmse_delta,
            "direction": "improvement" if rmse_delta > 0 else ("regression" if rmse_delta < 0 else "neutral"),
        },
    }
    save_results(full_results, output_path)

    logger.info("=" * 65)
    logger.info("DONE — patient: %s, output: %s", patient_id, output_path)
    logger.info("=" * 65)

    return {
        "patient_id": patient_id,
        "stage1_rmse": stage1_rmse,
        "stage2_rmse": stage2_rmse,
        "delta": rmse_delta,
        "direction": "improvement" if rmse_delta > 0 else ("regression" if rmse_delta < 0 else "neutral"),
        "n_episodes": stage1_results.get("total_episodes", 0),
        "status": "OK",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-patient Stage 2 fine-tuning on holdout patient(s). "
            "Requires a Stage 1 checkpoint trained via the holdout workflow."
        )
    )
    parser.add_argument("--model", type=str, required=True, choices=["ttm", "sundial", "chronos2"],
                        help="Model type (must match checkpoint)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. brown_2019)")

    # Patient selection: single or all
    patient_group = parser.add_mutually_exclusive_group(required=True)
    patient_group.add_argument("--patient-id", type=str, default=None,
                               help="Target patient ID (must be in holdout set)")
    patient_group.add_argument("--all-patients", action="store_true",
                               help="Run for ALL holdout patients (batch mode)")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not set)")
    parser.add_argument("--config-dir", type=str, default="configs/data/holdout_10pct",
                        help="Holdout config directory (default: holdout_10pct)")
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

    # ── Load holdout data (shared across all patients) ──────────────────────
    logger.info("\n--- Loading Holdout Data ---")
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)

    patient_col = get_patient_column(holdout_data)

    # Determine patient list
    if args.all_patients:
        # Get patient list from holdout config (not from data, which includes
        # temporal holdout rows for ALL patients in hybrid mode).
        holdout_config = registry.get_holdout_config(args.dataset)
        if holdout_config and holdout_config.patient_config:
            patient_ids = sorted(holdout_config.patient_config.holdout_patients)
        else:
            patient_ids = sorted(holdout_data[patient_col].unique().astype(str))
        logger.info("Batch mode: %d holdout patients", len(patient_ids))
    else:
        patient_ids = [args.patient_id]

    # ── Setup output directories ────────────────────────────────────────────
    if args.all_patients:
        batch_root = setup_batch_output_dir(args.model, args.output_dir)
        log_file = setup_file_logging(batch_root, "batch_finetune.log")
        logger.info("Batch output root: %s", batch_root)
        logger.info("Batch log: %s", log_file)
    else:
        batch_root = None

    # ── Run per-patient fine-tuning ─────────────────────────────────────────
    summary_rows: List[Dict[str, Any]] = []

    for idx, patient_id in enumerate(patient_ids):
        logger.info("\n")
        logger.info("*" * 65)
        logger.info("PATIENT %d/%d: %s", idx + 1, len(patient_ids), patient_id)
        logger.info("*" * 65)

        # Determine per-patient output directory
        if batch_root is not None:
            patient_output = batch_root / patient_id
        else:
            patient_output = setup_output_dir(
                args.model, args.dataset, patient_id, args.output_dir
            )

        try:
            result = run_single_patient(
                args=args,
                patient_id=patient_id,
                holdout_data=holdout_data,
                patient_col=patient_col,
                output_path=patient_output,
            )
            summary_rows.append(result)
        except Exception as e:
            logger.error("FAILED for patient %s: %s", patient_id, e, exc_info=True)
            summary_rows.append({
                "patient_id": patient_id,
                "stage1_rmse": float("nan"),
                "stage2_rmse": float("nan"),
                "delta": float("nan"),
                "direction": "error",
                "n_episodes": 0,
                "status": f"FAILED: {e}",
            })

    # ── Write summary CSV (batch mode or single patient) ────────────────────
    output_root = batch_root if batch_root is not None else patient_output
    write_summary_csv(summary_rows, output_root)

    # ── Final summary ───────────────────────────────────────────────────────
    logger.info("\n")
    logger.info("=" * 65)
    logger.info("ALL PATIENTS COMPLETE")
    logger.info("=" * 65)
    ok_count = sum(1 for r in summary_rows if r["status"] == "OK")
    logger.info("  Succeeded: %d / %d", ok_count, len(summary_rows))
    logger.info("  Output:    %s", output_root)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
