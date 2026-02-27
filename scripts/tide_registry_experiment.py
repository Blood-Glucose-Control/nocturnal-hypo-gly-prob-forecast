#!/usr/bin/env python3
"""
TiDE experiment using the data registry with patient-level holdout.

This replaces the direct Brown data loader with the DatasetRegistry,
which provides proper patient-level holdout splits (hybrid: patient-based
+ temporal). This means:

  - Training: ~160 patients (excluding 8 holdout patients), with last 5% held for validation
  - Evaluation: 8 fully held-out patients the model has NEVER seen

Previous scripts used Brown's built-in train/val split, which is temporal-only
(same patients in both sets). This script tests true generalization to unseen patients.

Mirrors the Chronos-2 model class pattern:
  DatasetRegistry → flat_df → patient_dict → gap_segments → AutoGluon

USAGE:
    # Best known config (Full HPO trial 3456bc05, RMSE=1.876)
    python scripts/tide_registry_experiment.py --config best

    # Scaled baseline
    python scripts/tide_registry_experiment.py --config scaled

    # Custom holdout config directory
    python scripts/tide_registry_experiment.py --config best --holdout-dir configs/data/holdout_5pct
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datetime import datetime  # noqa: E402

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor  # noqa: E402
from src.data.versioning.dataset_registry import DatasetRegistry  # noqa: E402
from src.data.models import ColumnNames  # noqa: E402
from src.data.preprocessing.gap_handling import segment_all_patients  # noqa: E402
from src.models.chronos2.utils import convert_to_patient_dict  # noqa: E402
from src.evaluation.episode_builders import build_midnight_episodes  # noqa: E402

# Configuration
INTERVAL_MINS = 5
FORECAST_HORIZON = 72  # 6 hours
TARGET_COL = ColumnNames.BG.value
IOB_COL = ColumnNames.IOB.value
PATIENT_COL = "p_num"
TIME_COL = "datetime"


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================


def get_tide_config(config_name: str, prediction_length: int):
    """Get TiDE hyperparameters for specified configuration."""
    shared = {
        "scaling": "mean",
        "known_covariates_names": ["iob"],
        "batch_size": 256,
        "lr": 1e-3,
        "trainer_kwargs": {
            "gradient_clip_val": 1.0,
            "precision": "16-mixed",
        },
    }

    if config_name == "best":
        # Full HPO best trial (3456bc05): RMSE=1.876, discontinuity=0.164
        context_length = 512
        hyperparameters = {
            "TiDE": {
                **shared,
                "context_length": context_length,
                "encoder_hidden_dim": 256,
                "decoder_hidden_dim": 256,
                "temporal_hidden_dim": 256,
                "num_layers_encoder": 2,
                "num_layers_decoder": 2,
                "distr_hidden_dim": 8,
                "dropout": 0.1,
                "lr": 0.000931,
                "num_batches_per_epoch": 300,
            }
        }
        description = "TiDE Best (Full HPO 3456bc05, RMSE=1.876)"

    elif config_name == "scaled":
        # Scaled baseline (512c, 256d)
        context_length = 512
        hyperparameters = {
            "TiDE": {
                **shared,
                "context_length": context_length,
                "encoder_hidden_dim": 256,
                "decoder_hidden_dim": 256,
                "temporal_hidden_dim": 256,
                "num_batches_per_epoch": 200,
            }
        }
        description = "TiDE Scaled Baseline (512c, 256d)"

    else:
        raise ValueError(f"Unknown config: {config_name}. Use 'best' or 'scaled'")

    return context_length, hyperparameters, description


# =============================================================================
# DATA FORMATTING (reuses chronos2 pattern)
# =============================================================================


def format_segments_for_autogluon(segments, target_col, iob_col):
    """Convert gap-handled segments to AutoGluon TimeSeriesDataFrame with IOB."""
    data_list = []

    for seg_id, seg_df in segments.items():
        df = seg_df[[target_col]].copy()
        df = df.rename(columns={target_col: "target"})

        has_iob = iob_col in seg_df.columns and seg_df[iob_col].notna().any()
        df["iob"] = seg_df[iob_col].ffill().fillna(0) if has_iob else 0.0

        df["item_id"] = seg_id
        df["timestamp"] = df.index
        data_list.append(df[["item_id", "timestamp", "target", "iob"]])

    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.set_index(["item_id", "timestamp"])
    return TimeSeriesDataFrame(combined)


def format_episodes_for_autogluon(episodes, forecast_horizon):
    """Convert episode dicts to AutoGluon format for prediction."""
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:04d}"

        # Context data
        df = ep["context_df"].copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[TARGET_COL]
        df["iob"] = df[IOB_COL].ffill().fillna(0) if IOB_COL in df.columns else 0.0
        train_data_list.append(df[["item_id", "timestamp", "target", "iob"]])

        # Known future covariates
        future_covs = ep.get("future_covariates", {})
        future_iob = future_covs.get(IOB_COL, np.zeros(forecast_horizon))
        future_iob = pd.Series(future_iob).ffill().fillna(0).to_numpy()
        future_timestamps = pd.date_range(
            ep["anchor"], periods=len(future_iob), freq=f"{INTERVAL_MINS}min"
        )
        future_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": future_timestamps,
                "iob": future_iob[:forecast_horizon],
            }
        )
        known_cov_list.append(future_df)

    train_combined = pd.concat(train_data_list, ignore_index=True)
    train_combined = train_combined.set_index(["item_id", "timestamp"])
    train_data = TimeSeriesDataFrame(train_combined)

    known_combined = pd.concat(known_cov_list, ignore_index=True)
    known_combined = known_combined.set_index(["item_id", "timestamp"])
    known_covariates = TimeSeriesDataFrame(known_combined)

    return train_data, known_covariates


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_with_discontinuity(predictor, test_data, known_covariates, episodes):
    """Evaluate and measure discontinuity on midnight-anchored episodes."""
    print("\nRunning predictions with known covariates...")
    predictions = predictor.predict(test_data, known_covariates=known_covariates)

    rmse_list, discontinuities, variances = [], [], []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:04d}"
        if item_id not in predictions.index.get_level_values(0):
            continue

        pred = predictions.loc[item_id]["mean"].values
        actual = ep["target_bg"][: len(pred)]

        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        rmse_list.append(rmse)

        last_context = ep["context_df"][TARGET_COL].iloc[-1]
        first_forecast = pred[0]
        discontinuities.append(abs(last_context - first_forecast))
        variances.append(np.var(pred))

    return {
        "num_episodes": len(rmse_list),
        "rmse": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "discontinuity": float(np.mean(discontinuities)),
        "discontinuity_std": float(np.std(discontinuities)),
        "variance": float(np.mean(variances)),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="TiDE with data registry + holdout")
    parser.add_argument(
        "--config",
        type=str,
        choices=["best", "scaled"],
        required=True,
        help="Configuration preset",
    )
    parser.add_argument(
        "--holdout-dir",
        type=str,
        default="configs/data/holdout_5pct",
        help="Holdout config directory (default: 5%% patient holdout)",
    )
    parser.add_argument(
        "--time-limit", type=int, default=7200, help="Training time limit (s)"
    )
    parser.add_argument("--max-eval-episodes", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="brown_2019")
    args = parser.parse_args()

    context_length, hyperparameters, config_desc = get_tide_config(
        args.config, FORECAST_HORIZON
    )

    print("=" * 70)
    print("TiDE EXPERIMENT (DATA REGISTRY + PATIENT HOLDOUT)")
    print("=" * 70)
    print(f"\nConfiguration: {config_desc}")
    print(f"Holdout config: {args.holdout_dir}")
    print(
        f"Context: {context_length} steps ({context_length * INTERVAL_MINS / 60:.1f}h)"
    )
    print(
        f"Forecast: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * INTERVAL_MINS / 60:.1f}h)"
    )

    # =========================================================================
    # LOAD DATA VIA REGISTRY (patient-level holdout)
    # =========================================================================

    print(f"\n{'='*70}")
    print("LOADING DATA VIA REGISTRY")
    print("=" * 70)

    registry = DatasetRegistry(holdout_config_dir=args.holdout_dir)
    train_df, holdout_df = registry.load_dataset_with_split(args.dataset)

    train_patients = train_df[PATIENT_COL].nunique()
    holdout_patients = holdout_df[PATIENT_COL].nunique()
    print(f"Train: {len(train_df):,} rows, {train_patients} patients")
    print(f"Holdout: {len(holdout_df):,} rows, {holdout_patients} patients")

    # Show holdout patient IDs
    holdout_ids = sorted(holdout_df[PATIENT_COL].unique())
    print(f"Holdout patient IDs: {holdout_ids}")

    # Confirm no overlap
    train_ids = set(train_df[PATIENT_COL].unique())
    holdout_ids_set = set(holdout_df[PATIENT_COL].unique())
    patient_overlap = train_ids & holdout_ids_set
    if patient_overlap:
        print(
            f"WARNING: {len(patient_overlap)} patients appear in BOTH sets (temporal holdout portion)"
        )
    else:
        print("Patient-level holdout: ZERO overlap between train and holdout patients")

    # =========================================================================
    # GAP HANDLING + FORMAT FOR AUTOGLUON
    # =========================================================================

    print(f"\n{'='*70}")
    print("GAP HANDLING + SEGMENTATION (training data)")
    print("=" * 70)

    min_segment_length = context_length + FORECAST_HORIZON

    # Convert flat df -> patient dict (same pattern as Chronos-2 model class)
    train_patient_dict = convert_to_patient_dict(train_df, PATIENT_COL, TIME_COL)
    print(f"Training patients: {len(train_patient_dict)}")

    # Gap handling
    train_segments = segment_all_patients(
        train_patient_dict,
        imputation_threshold_mins=45,
        min_segment_length=min_segment_length,
        bg_col=TARGET_COL,
    )

    train_rows = sum(len(df) for df in train_segments.values())
    print(f"Training segments: {len(train_segments)} ({train_rows:,} rows)")

    # Format for AutoGluon
    ts_train = format_segments_for_autogluon(train_segments, TARGET_COL, IOB_COL)
    print(f"AutoGluon training data: {ts_train.shape}, {ts_train.num_items} items")

    # =========================================================================
    # TRAINING
    # =========================================================================

    import os  # noqa: E402

    job_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = str(PROJECT_ROOT / f"models/tide_registry/{args.config}_{job_id}")

    print(f"\n{'='*70}")
    print(f"TRAINING: {config_desc}")
    print("=" * 70)
    print("\nHyperparameters:")
    for key, value in hyperparameters["TiDE"].items():
        if key != "trainer_kwargs":
            print(f"  {key}: {value}")

    predictor = TimeSeriesPredictor(
        prediction_length=FORECAST_HORIZON,
        target="target",
        known_covariates_names=["iob"],
        eval_metric="RMSE",
        path=output_dir,
    )

    predictor.fit(
        train_data=ts_train,
        hyperparameters=hyperparameters,
        time_limit=args.time_limit,
        enable_ensemble=False,
    )

    leaderboard = predictor.leaderboard()
    print(leaderboard)

    # =========================================================================
    # EVALUATE ON HOLDOUT PATIENTS (never seen during training)
    # =========================================================================

    print(f"\n{'='*70}")
    print("EVALUATION ON HOLDOUT PATIENTS (midnight-anchored)")
    print("=" * 70)

    # Convert holdout flat_df -> patient dict
    holdout_patient_dict = convert_to_patient_dict(holdout_df, PATIENT_COL, TIME_COL)
    print(f"Holdout patients: {len(holdout_patient_dict)}")

    # Build midnight episodes from holdout patients
    eval_episodes = []
    total_skip_stats = {"total_anchors": 0, "skipped_bg_nan": 0}

    for pid, pdf in holdout_patient_dict.items():
        episodes, skip_stats = build_midnight_episodes(
            pdf,
            context_length=context_length,
            forecast_length=FORECAST_HORIZON,
            target_col=TARGET_COL,
            covariate_cols=[IOB_COL],
            interval_mins=INTERVAL_MINS,
        )
        eval_episodes.extend(episodes)
        total_skip_stats["total_anchors"] += skip_stats["total_anchors"]
        total_skip_stats["skipped_bg_nan"] += skip_stats["skipped_bg_nan"]

        if len(eval_episodes) >= args.max_eval_episodes:
            break

    eval_episodes = eval_episodes[: args.max_eval_episodes]
    print(f"Evaluation episodes: {len(eval_episodes)}")
    print(f"  Anchors considered: {total_skip_stats['total_anchors']}")
    print(f"  Skipped (BG NaN): {total_skip_stats['skipped_bg_nan']}")

    if len(eval_episodes) == 0:
        print("ERROR: No valid evaluation episodes found!")
        return

    # Format for AutoGluon
    ts_eval, known_cov_eval = format_episodes_for_autogluon(
        eval_episodes, FORECAST_HORIZON
    )

    # Evaluate
    results = evaluate_with_discontinuity(
        predictor, ts_eval, known_cov_eval, eval_episodes
    )

    # =========================================================================
    # RESULTS
    # =========================================================================

    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)

    print(f"\nConfiguration: {config_desc}")
    print(f"Holdout config: {args.holdout_dir}")
    print(f"Evaluation: {results['num_episodes']} midnight-anchored episodes")
    print(f"  (from {holdout_patients} patients the model has NEVER seen)")

    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    print(f"{'RMSE':<30} {results['rmse']:>10.4f} mM")
    print(f"{'RMSE std':<30} {results['rmse_std']:>10.4f}")
    print(f"{'Discontinuity':<30} {results['discontinuity']:>10.4f} mM")
    print(f"{'Discontinuity std':<30} {results['discontinuity_std']:>10.4f}")
    print(f"{'Prediction variance':<30} {results['variance']:>10.4f}")

    # Validation check
    print(f"\n{'='*70}")
    print("VALIDATION")
    print("=" * 70)

    if results["discontinuity"] < 0.2:
        print(f"PASS: Discontinuity {results['discontinuity']:.3f} < 0.2 mM")
    else:
        print(f"FAIL: Discontinuity {results['discontinuity']:.3f} >= 0.2 mM")

    if results["variance"] > 0.1:
        print(f"PASS: Variance {results['variance']:.3f} > 0.1 (not mean reversion)")
    else:
        print(
            f"FAIL: Variance {results['variance']:.3f} <= 0.1 (possible mean reversion)"
        )

    print("\nComparison (previous results on temporal-only holdout):")
    print("  TiDE Full HPO (temporal):     1.876 RMSE, 0.164 discont")
    print("  TiDE Bayesian HPO (temporal): 1.970 RMSE, 0.129 discont")
    print("  Chronos-2 P1 15K (temporal):  1.890 RMSE")
    print(
        f"  TiDE Registry (patient):      {results['rmse']:.3f} RMSE, {results['discontinuity']:.3f} discont"
    )

    # Save results
    results_path = PROJECT_ROOT / f"models/tide_registry/{args.config}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "description": config_desc,
                "holdout_dir": args.holdout_dir,
                "dataset": args.dataset,
                "context_length": context_length,
                "train_patients": train_patients,
                "holdout_patients": holdout_patients,
                "holdout_patient_ids": sorted(
                    str(p) for p in holdout_df[PATIENT_COL].unique()
                ),
                "results": results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
