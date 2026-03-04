#!/usr/bin/env python3
"""
TTM holdout workflow -- direct, no factory indirection.

7-step validation workflow for TTM (TinyTimeMixer) using the model class
directly. Serves as a template for model-specific holdout scripts.

Steps:
  1. Check holdout configs exist
  2. Validate holdout configs (no data leakage)
  3. Load & combine training data
  4. Zero-shot evaluation (midnight-anchored, all holdout patients)
  5. Fine-tune model
  6. Load checkpoint (verify save/load)
  7. Resume training from checkpoint

Usage:
    # Zero-shot only (skip training)
    python scripts/examples/ttm_holdout_workflow.py \\
        --datasets brown_2019 --skip-training

    # Full workflow with training
    python scripts/examples/ttm_holdout_workflow.py \\
        --datasets brown_2019 --epochs 10

    # Use YAML config for TTM params
    python scripts/examples/ttm_holdout_workflow.py \\
        --datasets brown_2019 \\
        --model-config configs/models/ttm/fine_tune.yaml \\
        --epochs 25

    # Skip specific steps
    python scripts/examples/ttm_holdout_workflow.py \\
        --datasets brown_2019 --skip-steps 4 7
"""

import argparse
import json
import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.versioning import holdout_utils
from src.data.preprocessing.dataset_combiner import (
    combine_datasets_for_training,
)
from src.data.utils import get_patient_column
from src.evaluation.episode_builders import build_midnight_episodes
from src.evaluation.metrics import compute_regression_metrics
from src.models.ttm import TTMForecaster, TTMConfig
from src.models.ttm.config import (
    create_default_ttm_config,
    create_ttm_zero_shot_config,
)
from src.utils import load_yaml_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL = "bg_mM"
INTERVAL_MINS = 5
STEPS_PER_HOUR = 60 // INTERVAL_MINS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy sub-loggers
logging.getLogger("src.data.preprocessing").setLevel(logging.WARNING)
logging.getLogger("src.data.diabetes_datasets").setLevel(logging.WARNING)
logging.getLogger("src.models").setLevel(logging.WARNING)
logging.getLogger("src.utils").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------
def evaluate_midnight(
    model: TTMForecaster,
    holdout_df: pd.DataFrame,
    context_length: int,
    forecast_length: int,
    phase_name: str,
    output_dir: str,
    covariate_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run midnight-anchored evaluation on all holdout patients.

    Builds midnight episodes per patient, calls model.predict()
    per episode, computes RMSE + discontinuity.

    Returns:
        Dict with overall_rmse, discontinuity, total_episodes, per_patient.
    """
    patient_col = get_patient_column(holdout_df)
    patients = holdout_df[patient_col].unique()

    all_preds, all_targets = [], []
    all_discs = []
    per_patient = []

    for patient_id in tqdm(patients, desc=f"[{phase_name}] Evaluating", unit="pt"):
        patient_df = holdout_df[holdout_df[patient_col] == patient_id].copy()

        # Set DatetimeIndex
        if not isinstance(patient_df.index, pd.DatetimeIndex):
            if "datetime" in patient_df.columns:
                patient_df["datetime"] = pd.to_datetime(patient_df["datetime"])
                patient_df = patient_df.set_index("datetime").sort_index()
            else:
                continue

        episodes, stats = build_midnight_episodes(
            patient_df,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=TARGET_COL,
            covariate_cols=covariate_cols,
        )
        if not episodes:
            continue

        patient_preds, patient_targets = [], []
        for ep in episodes:
            # Build context DataFrame that model.predict() expects
            ctx = ep["context_df"].reset_index(names="datetime")
            ctx[patient_col] = patient_id

            pred = np.asarray(model.predict(ctx))
            if pred.ndim == 3:
                pred = pred[0, :, 0]
            elif pred.ndim == 2:
                pred = pred.flatten()
            target = ep["target_bg"]

            if len(pred) > len(target):
                pred = pred[: len(target)]

            patient_preds.append(pred)
            patient_targets.append(target)

            # Discontinuity: |last context BG - first forecast BG|
            last_ctx = ep["context_df"][TARGET_COL].iloc[-1]
            disc = abs(float(last_ctx) - float(pred[0]))
            all_discs.append(disc)

        p_pred = np.concatenate(patient_preds)
        p_tgt = np.concatenate(patient_targets)
        metrics = compute_regression_metrics(p_pred, p_tgt)
        per_patient.append(
            {"patient_id": str(patient_id), "episodes": len(episodes), **metrics}
        )
        all_preds.append(p_pred)
        all_targets.append(p_tgt)

    if not all_preds:
        logger.warning(f"[{phase_name}] No valid episodes found")
        return {"overall_rmse": float("nan"), "total_episodes": 0, "per_patient": []}

    concat_pred = np.concatenate(all_preds)
    concat_tgt = np.concatenate(all_targets)
    overall_rmse = float(np.sqrt(np.mean((concat_pred - concat_tgt) ** 2)))
    mean_disc = float(np.mean(all_discs))

    # Save results
    results = {
        "phase": phase_name,
        "overall_rmse": overall_rmse,
        "mean_discontinuity": mean_disc,
        "total_episodes": sum(p["episodes"] for p in per_patient),
        "total_patients": len(per_patient),
        "per_patient": per_patient,
    }
    out_path = Path(output_dir) / f"eval_{phase_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(
        f"[{phase_name}] RMSE={overall_rmse:.4f}  Disc={mean_disc:.3f}  "
        f"Episodes={results['total_episodes']}  Patients={results['total_patients']}"
    )
    return results


def _eval_all_holdouts(
    model: TTMForecaster,
    holdout_datasets: Dict[str, pd.DataFrame],
    context_length: int,
    forecast_length: int,
    phase_name: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Run evaluate_midnight on each dataset's holdout data.

    When multiple datasets are provided, results are saved per-dataset
    with the dataset name appended to the phase (e.g., eval_0_zero_shot_brown_2019.json).
    """
    # Get covariate columns the model expects (from preprocessor or config)
    covariate_cols = None
    if model.preprocessor is not None and hasattr(
        model.preprocessor, "observable_columns"
    ):
        covariate_cols = list(model.preprocessor.observable_columns)
    elif model.config.input_features:
        covariate_cols = list(model.config.input_features)

    all_results = {}
    multi = len(holdout_datasets) > 1

    for ds_name, holdout_df in holdout_datasets.items():
        ds_phase = f"{phase_name}_{ds_name}" if multi else phase_name
        all_results[ds_name] = evaluate_midnight(
            model,
            holdout_df,
            context_length,
            forecast_length,
            phase_name=ds_phase,
            output_dir=output_dir,
            covariate_cols=covariate_cols,
        )

    return all_results


# ---------------------------------------------------------------------------
# Step 1: Check holdout configs exist
# ---------------------------------------------------------------------------
def step1_check_configs(config_dir: str, datasets: List[str], output_dir: str) -> bool:
    config_path = Path(config_dir)
    out_path = Path(output_dir) / "configs"
    out_path.mkdir(parents=True, exist_ok=True)

    ok = True
    for ds in datasets:
        cfg_file = config_path / f"{ds}.yaml"
        if cfg_file.exists():
            shutil.copy2(cfg_file, out_path / cfg_file.name)
            logger.info(f"  [Step 1] {ds}: config OK")
        else:
            logger.error(f"  [Step 1] {ds}: config MISSING at {cfg_file}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Step 2: Validate holdout configs
# ---------------------------------------------------------------------------
def step2_validate_configs(datasets: List[str], registry: DatasetRegistry) -> bool:
    ok = True
    for ds in datasets:
        try:
            result = holdout_utils.validate_holdout_config(ds, registry, verbose=False)
            status = "PASS" if not result.get("errors") else "FAIL"
            logger.info(f"  [Step 2] {ds}: {status}")
            if status == "FAIL":
                ok = False
        except Exception as e:
            logger.error(f"  [Step 2] {ds}: ERROR - {e}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Step 3: Load training data
# ---------------------------------------------------------------------------
def step3_load_data(
    datasets: List[str], registry: DatasetRegistry, config_dir: str
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load and combine training data, also return holdout for eval.

    Returns:
        (train_data, holdout_datasets) where holdout_datasets maps
        dataset_name -> holdout DataFrame for each dataset.
    """
    # Load combined training data
    combined_data, _column_info = combine_datasets_for_training(
        datasets, registry, config_dir
    )
    logger.info(f"  [Step 3] Combined training data: {len(combined_data):,} rows")

    # BG NaN gaps are handled by segment_all_patients() inside model.fit().
    # Covariates (iob, insulin_availability) are already clean from data loaders.
    nan_count = combined_data[TARGET_COL].isna().sum()
    if nan_count > 0:
        logger.info(
            f"  [Step 3] {nan_count:,} BG NaN values (handled by gap segmentation in model.fit)"
        )

    # Load holdout data for evaluation from ALL datasets
    holdout_datasets: Dict[str, pd.DataFrame] = {}
    for ds in datasets:
        holdout_df = registry.load_holdout_data_only(ds)
        patient_col = get_patient_column(holdout_df)
        n_patients = holdout_df[patient_col].nunique()
        logger.info(
            f"  [Step 3] Holdout ({ds}): {len(holdout_df):,} rows, "
            f"{n_patients} patients"
        )
        holdout_datasets[ds] = holdout_df

    return combined_data, holdout_datasets


# ---------------------------------------------------------------------------
# Step 4: Zero-shot evaluation
# ---------------------------------------------------------------------------
def step4_zero_shot(
    zs_config: TTMConfig,
    holdout_datasets: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Dict[str, Any]:
    logger.info("  [Step 4] Creating zero-shot TTM model...")
    logger.info(f"  [Step 4] Resolved config: {zs_config.to_dict()}")
    model = TTMForecaster(zs_config)

    return _eval_all_holdouts(
        model,
        holdout_datasets,
        zs_config.context_length,
        zs_config.forecast_length,
        phase_name="0_zero_shot",
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Step 5: Train model
# ---------------------------------------------------------------------------
def step5_train(
    ft_config: TTMConfig,
    train_data: pd.DataFrame,
    holdout_datasets: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Tuple[TTMForecaster, Path, Dict[str, Any]]:
    logger.info("  [Step 5] Creating TTM model for fine-tuning...")
    logger.info(f"  [Step 5] Resolved config: {ft_config.to_dict()}")
    model = TTMForecaster(ft_config)

    logger.info(
        f"  [Step 5] Training for {ft_config.num_epochs} epochs, "
        f"batch_size={ft_config.batch_size}, lr={ft_config.learning_rate}"
    )
    results = model.fit(train_data=train_data, output_dir=output_dir)
    logger.info(f"  [Step 5] Training complete: {results.get('train_metrics', {})}")

    model_path = Path(output_dir) / "model.pt"
    model.save(str(model_path))
    logger.info(f"  [Step 5] Model saved to {model_path}")

    eval_results = _eval_all_holdouts(
        model,
        holdout_datasets,
        ft_config.context_length,
        ft_config.forecast_length,
        phase_name="1_after_training",
        output_dir=output_dir,
    )
    return model, model_path, eval_results


# ---------------------------------------------------------------------------
# Step 6: Load checkpoint
# ---------------------------------------------------------------------------
def step6_load_checkpoint(
    ft_config: TTMConfig,
    model_path: Path,
    holdout_datasets: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Tuple[Optional[TTMForecaster], Optional[Dict]]:
    if not model_path.exists():
        logger.error(f"  [Step 6] Checkpoint not found: {model_path}")
        return None, None

    logger.info(f"  [Step 6] Loading checkpoint from {model_path}...")
    loaded_model = TTMForecaster.load(str(model_path), ft_config)

    eval_results = _eval_all_holdouts(
        loaded_model,
        holdout_datasets,
        ft_config.context_length,
        ft_config.forecast_length,
        phase_name="2_after_loading",
        output_dir=output_dir,
    )
    return loaded_model, eval_results


# ---------------------------------------------------------------------------
# Step 7: Resume training
# ---------------------------------------------------------------------------
def step7_resume_training(
    model: TTMForecaster,
    train_data: pd.DataFrame,
    holdout_datasets: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Tuple[TTMForecaster, Path, Dict[str, Any]]:
    resumed_dir = Path(output_dir) / "resumed_training"
    logger.info(f"  [Step 7] Resuming training for {model.config.num_epochs} epochs...")

    # Log pre-resume state for comparison
    pre_history = getattr(model, "training_history", {})
    logger.info(
        f"  [Step 7] Pre-resume training history entries: {len(pre_history) if isinstance(pre_history, list) else 'dict'}"
    )

    results = model.fit(train_data=train_data, output_dir=str(resumed_dir))
    logger.info(f"  [Step 7] Resume complete: {results.get('train_metrics', {})}")

    post_history = getattr(model, "training_history", {})
    best = getattr(model, "best_metrics", {})
    logger.info(
        f"  [Step 7] Post-resume history entries: {len(post_history) if isinstance(post_history, list) else 'dict'}"
    )
    if best:
        logger.info(f"  [Step 7] Best metrics: {best}")

    model_path = resumed_dir / "model.pt"
    model.save(str(model_path))

    eval_results = _eval_all_holdouts(
        model,
        holdout_datasets,
        model.config.context_length,
        model.config.forecast_length,
        phase_name="3_after_resumed_training",
        output_dir=output_dir,
    )
    return model, model_path, eval_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTM holdout workflow (direct, no factory)")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["brown_2019"],
        help="Datasets to train on (first is used for eval)",
    )
    p.add_argument(
        "--config-dir",
        default="configs/data/holdout_10pct",
        help="Holdout config directory",
    )
    p.add_argument(
        "--model-config",
        default=None,
        help="YAML config file for TTM hyperparameters",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    p.add_argument("--learning-rate", type=float, default=None, help="Override lr")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--forecast-length", type=int, default=96)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (auto-generated if not set)",
    )
    p.add_argument(
        "--skip-steps",
        nargs="+",
        type=int,
        default=[],
        help="Steps to skip (e.g., --skip-steps 4 7)",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip steps 5, 6, 7 (zero-shot only)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ----- Resolve config overrides: CLI > YAML > defaults (done ONCE) -----
    config_overrides: Dict[str, Any] = {
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
    }
    if args.model_config:
        yaml_config = load_yaml_config(args.model_config)
        config_overrides.update(yaml_config)
    # CLI args override YAML
    if args.epochs is not None:
        config_overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config_overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config_overrides["learning_rate"] = args.learning_rate

    # ----- Create all objects upfront -----
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)

    zs_config = create_ttm_zero_shot_config(**config_overrides)
    ft_overrides = {k: v for k, v in config_overrides.items() if k != "training_mode"}
    ft_config = create_default_ttm_config(training_mode="fine_tune", **ft_overrides)

    skip = set(args.skip_steps)
    if args.skip_training:
        skip.update({5, 6, 7})

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = args.output_dir or (
        f"trained_models/artifacts/ttm_workflow/{timestamp}_{'_'.join(args.datasets)}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save run config for reproducibility
    run_config = {
        "datasets": args.datasets,
        "config_dir": args.config_dir,
        "model_config": args.model_config,
        "config_overrides": {k: str(v) for k, v in config_overrides.items()},
        "skip_steps": list(skip),
        "timestamp": timestamp,
    }
    with open(Path(output_dir) / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Copy model-config YAML to output dir for reproducibility
    if args.model_config and Path(args.model_config).exists():
        shutil.copy2(args.model_config, Path(output_dir) / Path(args.model_config).name)

    logger.info("=" * 60)
    logger.info("TTM HOLDOUT WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Datasets:  {args.datasets}")
    logger.info(f"Output:    {output_dir}")
    logger.info(
        f"Context:   {zs_config.context_length} steps ({zs_config.context_length / STEPS_PER_HOUR:.1f}h)"
    )
    logger.info(
        f"Forecast:  {zs_config.forecast_length} steps ({zs_config.forecast_length / STEPS_PER_HOUR:.1f}h)"
    )
    logger.info(f"Skip:      {sorted(skip) if skip else 'none'}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Check configs
    # ------------------------------------------------------------------
    if 1 not in skip:
        logger.info("\n--- Step 1: Check holdout configs ---")
        if not step1_check_configs(args.config_dir, args.datasets, output_dir):
            logger.error("Step 1 FAILED. Aborting.")
            return

    # ------------------------------------------------------------------
    # Step 2: Validate configs
    # ------------------------------------------------------------------
    if 2 not in skip:
        logger.info("\n--- Step 2: Validate holdout configs ---")
        if not step2_validate_configs(args.datasets, registry):
            logger.error("Step 2 FAILED. Aborting.")
            return

    # ------------------------------------------------------------------
    # Step 3: Load data (train + holdout loaded ONCE here)
    # ------------------------------------------------------------------
    logger.info("\n--- Step 3: Load training + holdout data ---")
    train_data, holdout_datasets = step3_load_data(
        args.datasets, registry, args.config_dir
    )

    # ------------------------------------------------------------------
    # Step 4: Zero-shot evaluation
    # ------------------------------------------------------------------
    if 4 not in skip:
        logger.info("\n--- Step 4: Zero-shot evaluation ---")
        try:
            step4_zero_shot(zs_config, holdout_datasets, output_dir)
        except Exception as e:
            logger.error(f"Step 4 FAILED: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Step 5: Train
    # ------------------------------------------------------------------
    model, model_path = None, None
    if 5 not in skip:
        logger.info("\n--- Step 5: Fine-tune model ---")
        try:
            model, model_path, _ = step5_train(
                ft_config, train_data, holdout_datasets, output_dir
            )
        except Exception as e:
            logger.error(f"Step 5 FAILED: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Step 6: Load checkpoint
    # ------------------------------------------------------------------
    if 6 not in skip:
        # Fall back to output_dir/model.pt if step 5 was skipped
        if model_path is None:
            model_path = Path(output_dir) / "model.pt"
            logger.info(
                f"  [Step 6] No model from step 5, trying fallback: {model_path}"
            )
        logger.info("\n--- Step 6: Load checkpoint ---")
        try:
            model, _ = step6_load_checkpoint(
                ft_config, model_path, holdout_datasets, output_dir
            )
        except Exception as e:
            logger.error(f"Step 6 FAILED: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Step 7: Resume training
    # ------------------------------------------------------------------
    if 7 not in skip and model is not None:
        logger.info("\n--- Step 7: Resume training ---")
        try:
            step7_resume_training(model, train_data, holdout_datasets, output_dir)
        except Exception as e:
            logger.error(f"Step 7 FAILED: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("WORKFLOW COMPLETE")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Workflow failed: {e}\n{traceback.format_exc()}")
