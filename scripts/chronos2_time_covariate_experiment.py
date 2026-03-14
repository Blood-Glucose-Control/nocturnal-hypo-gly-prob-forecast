#!/usr/bin/env python3
"""
Chronos-2 Time Covariate A/B Experiment.

Fine-tunes Chronos-2 on Brown 2019 in two matched arms:
  Arm A: BG-only (no covariates)
  Arm B: BG + time-of-day features (hour_sin, hour_cos as known future covariates)

Uses DatasetRegistry for holdout splits (patient + temporal holdout).
Training: gap-handled segments with AutoGluon sliding windows.
Evaluation: midnight-anchored episodes on holdout patients.

USAGE:
    python scripts/chronos2_time_covariate_experiment.py
    python scripts/chronos2_time_covariate_experiment.py --steps 10000
    python scripts/chronos2_time_covariate_experiment.py --arm B  # Run only Arm B
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
from src.models.chronos2.utils import build_midnight_episodes, convert_to_patient_dict  # noqa: E402

# Configuration
INTERVAL_MINS = 5
CONTEXT_LENGTH = 512  # ~42.7 hours
FORECAST_HORIZON = 72  # 6 hours
TARGET_COL = ColumnNames.BG.value
TIME_COLS = ["hour_sin", "hour_cos"]
MIN_SEGMENT_LENGTH = CONTEXT_LENGTH + FORECAST_HORIZON  # 584


def format_segments_for_autogluon(segments, target_col, covariate_cols=None):
    """
    Convert gap-handled segments to AutoGluon TimeSeriesDataFrame.

    Args:
        segments: Dict mapping segment_id -> DataFrame with DatetimeIndex.
        target_col: Column name for the target variable.
        covariate_cols: Optional list of covariate column names to include.

    Returns:
        TimeSeriesDataFrame with columns ["target"] + covariate_cols.
    """
    data_list = []

    for seg_id, seg_df in segments.items():
        df = pd.DataFrame({"target": seg_df[target_col]})

        if covariate_cols:
            for col in covariate_cols:
                if col in seg_df.columns and seg_df[col].notna().any():
                    df[col] = seg_df[col].ffill().fillna(0)
                else:
                    df[col] = 0.0

        df["item_id"] = seg_id
        df["timestamp"] = df.index
        keep_cols = ["item_id", "timestamp", "target"]
        if covariate_cols:
            keep_cols += covariate_cols
        data_list.append(df[keep_cols])

    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.set_index(["item_id", "timestamp"])
    return TimeSeriesDataFrame(combined)


def format_episodes_for_eval(episodes, covariate_cols=None, forecast_horizon=72):
    """
    Convert midnight episodes to AutoGluon format for evaluation.

    Returns:
        (test_data, known_covariates) or (test_data, None) if no covariates.
    """
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"

        # Context data
        df = ep["context_df"].copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[TARGET_COL]

        keep_cols = ["item_id", "timestamp", "target"]
        if covariate_cols:
            for col in covariate_cols:
                if col in TIME_COLS:
                    # Recompute time features from timestamps (don't ffill — sin/cos
                    # must match the actual timestamp, not be forward-filled from
                    # the last known value after reindex introduces NaN gaps)
                    mins = df.index.hour * 60 + df.index.minute
                    if col == "hour_sin":
                        df[col] = np.sin(2 * np.pi * mins / 1440)
                    else:
                        df[col] = np.cos(2 * np.pi * mins / 1440)
                elif col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
                else:
                    df[col] = 0.0
                keep_cols.append(col)

        train_data_list.append(df[keep_cols])

        # Future known covariates
        if covariate_cols:
            anchor = ep["anchor"]
            future_timestamps = pd.date_range(
                anchor, periods=forecast_horizon, freq=f"{INTERVAL_MINS}min"
            )

            future_data = {"item_id": item_id, "timestamp": future_timestamps}
            for col in covariate_cols:
                if col in TIME_COLS:
                    # Time features: recompute from timestamps (exact, no approximation)
                    mins = future_timestamps.hour * 60 + future_timestamps.minute
                    if col == "hour_sin":
                        future_data[col] = np.sin(2 * np.pi * mins / 1440)
                    else:
                        future_data[col] = np.cos(2 * np.pi * mins / 1440)
                elif col in ep.get("future_covariates", {}):
                    vals = ep["future_covariates"][col][:forecast_horizon]
                    if len(vals) < forecast_horizon:
                        vals = np.concatenate(
                            [vals, np.zeros(forecast_horizon - len(vals))]
                        )
                    future_data[col] = vals
                else:
                    future_data[col] = np.zeros(forecast_horizon)

            known_cov_list.append(pd.DataFrame(future_data))

    # Combine context data
    train_combined = pd.concat(train_data_list, ignore_index=True)
    train_combined = train_combined.set_index(["item_id", "timestamp"])
    test_data = TimeSeriesDataFrame(train_combined)

    # Combine known covariates
    if covariate_cols and known_cov_list:
        known_combined = pd.concat(known_cov_list, ignore_index=True)
        known_combined = known_combined.set_index(["item_id", "timestamp"])
        known_covariates = TimeSeriesDataFrame(known_combined)
        return test_data, known_covariates

    return test_data, None


def evaluate_arm(predictor, test_data, known_covariates, episodes):
    """Evaluate a predictor on midnight episodes. Returns (rmse, per_episode_errors)."""
    print("  Running predictions...")
    if known_covariates is not None:
        predictions = predictor.predict(test_data, known_covariates=known_covariates)
    else:
        predictions = predictor.predict(test_data)

    rmse_list = []
    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"
        if item_id not in predictions.index.get_level_values(0):
            continue

        pred = predictions.loc[item_id]["mean"].values
        actual = ep["target_bg"]

        min_len = min(len(pred), len(actual))
        pred = pred[:min_len]
        actual = actual[:min_len]

        valid = ~np.isnan(actual) & ~np.isnan(pred)
        if valid.sum() > 0:
            rmse = np.sqrt(np.mean((pred[valid] - actual[valid]) ** 2))
            rmse_list.append(rmse)

    overall_rmse = float(
        np.mean(rmse_list)
    )  # Mean of per-episode RMSEs (consistent with utils.py)
    return overall_rmse, rmse_list


def run_arm(
    arm_name, train_segments, val_segments, eval_patient_dict, covariate_cols, args
):
    """Run one arm of the experiment (train + evaluate)."""
    print(f"\n{'=' * 70}")
    print(
        f"ARM {arm_name}: {'BG + ' + ', '.join(covariate_cols) if covariate_cols else 'BG only'}"
    )
    print(f"{'=' * 70}")

    # Format segments for AutoGluon
    print("\nFormatting segments for AutoGluon...")
    ts_train = format_segments_for_autogluon(train_segments, TARGET_COL, covariate_cols)
    ts_val = format_segments_for_autogluon(val_segments, TARGET_COL, covariate_cols)

    print(f"Training data: {ts_train.shape}, columns={list(ts_train.columns)}")
    print(f"Validation data: {ts_val.shape}")

    # Set up predictor
    covariates_label = "_".join(covariate_cols) if covariate_cols else "bg_only"
    output_dir = str(
        PROJECT_ROOT
        / f"models/chronos2_time_covariate/arm_{covariates_label}_{args.steps}steps"
    )

    predictor_kwargs = {
        "prediction_length": FORECAST_HORIZON,
        "target": "target",
        "eval_metric": "RMSE",
        "path": output_dir,
    }
    if covariate_cols:
        predictor_kwargs["known_covariates_names"] = covariate_cols

    predictor = TimeSeriesPredictor(**predictor_kwargs)

    # Train
    print(f"\nFine-tuning Chronos-2 ({args.steps} steps)...")
    predictor.fit(
        train_data=ts_train,
        hyperparameters={
            "Chronos2": {
                "model_path": "autogluon/chronos-2",
                "fine_tune": True,
                "fine_tune_steps": args.steps,
                "fine_tune_lr": args.lr,
                "context_length": CONTEXT_LENGTH,
            }
        },
        time_limit=args.time_limit,
        enable_ensemble=False,
    )

    # Leaderboard on validation segments
    leaderboard = predictor.leaderboard(ts_val)
    print(leaderboard)
    val_rmse = -leaderboard["score_val"].values[0]
    print(f"Validation RMSE (sliding windows): {val_rmse:.4f}")

    # Evaluate on midnight episodes
    # NOTE: build_midnight_episodes requires at least one covariate to produce
    # episodes (it defaults to ["iob"] if None). For Arm A (BG-only), we still
    # pass covariate_cols=["iob"] to build episodes, but don't pass IOB to the
    # predictor. For Arm B, we pass time features + IOB for episode building
    # (IOB is needed to satisfy the function contract), but only time features
    # go to the predictor via known_covariates_names.
    print("\nBuilding midnight evaluation episodes...")
    episode_covs = list(set((covariate_cols or []) + ["iob"]))
    eval_episodes = []
    for pid, pdf in eval_patient_dict.items():
        eps = build_midnight_episodes(
            pdf,
            target_col=TARGET_COL,
            covariate_cols=episode_covs,
            interval_mins=INTERVAL_MINS,
            context_len=CONTEXT_LENGTH,
            horizon=FORECAST_HORIZON,
        )
        eval_episodes.extend(eps)
        if len(eval_episodes) >= args.max_eval_episodes:
            break

    eval_episodes = eval_episodes[: args.max_eval_episodes]
    print(f"Evaluation episodes: {len(eval_episodes)}")

    # Format for evaluation
    ts_eval, known_cov_eval = format_episodes_for_eval(
        eval_episodes, covariate_cols, FORECAST_HORIZON
    )

    # Load saved predictor and evaluate
    print("\nEvaluating on midnight-anchored episodes...")
    saved_predictor = TimeSeriesPredictor.load(output_dir)
    midnight_rmse, per_ep_errors = evaluate_arm(
        saved_predictor, ts_eval, known_cov_eval, eval_episodes
    )
    print(f"Midnight-anchored RMSE: {midnight_rmse:.4f}")

    return {
        "arm": arm_name,
        "covariates": covariate_cols or [],
        "val_rmse_sliding": val_rmse,
        "midnight_rmse": midnight_rmse,
        "eval_episodes": len(eval_episodes),
        "per_episode_rmse_mean": float(np.mean(per_ep_errors)),
        "per_episode_rmse_std": float(np.std(per_ep_errors)),
        "model_path": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Chronos-2 Time Covariate A/B Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--steps", type=int, default=5000, help="Fine-tuning steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--time-limit", type=int, default=7200, help="Time limit (s)")
    parser.add_argument(
        "--max-eval-episodes", type=int, default=500, help="Max evaluation episodes"
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Which arm to run: A (BG-only), B (BG+time), or both",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CHRONOS-2 TIME COVARIATE A/B EXPERIMENT")
    print("=" * 70)
    print("\nArm A: BG-only (no covariates)")
    print("Arm B: BG + time-of-day (hour_sin, hour_cos as known future covariates)")
    print("\nParameters:")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.lr}")
    print(f"  time_limit: {args.time_limit}s")
    print(f"  max_eval_episodes: {args.max_eval_episodes}")
    print(f"  arm: {args.arm}")

    # =========================================================================
    # LOAD DATA VIA REGISTRY (patient + temporal holdout)
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("LOADING DATA VIA DATASET REGISTRY")
    print("=" * 70)

    registry = DatasetRegistry()
    train_flat, holdout_flat = registry.load_dataset_with_split("brown_2019")

    print(f"Train: {len(train_flat):,} rows, {train_flat['p_num'].nunique()} patients")
    print(
        f"Holdout: {len(holdout_flat):,} rows, {holdout_flat['p_num'].nunique()} patients"
    )

    # Verify time features exist
    for col in TIME_COLS:
        assert (
            col in train_flat.columns
        ), f"{col} missing! Run: rm -rf cache/data/brown_2019/processed/ && python scripts/verify_time_features.py"
    print(f"Time features present: {TIME_COLS}")

    # Convert to patient dicts for gap handling (uses existing utility from utils.py)
    print("\nConverting to patient dicts...")
    train_patients = convert_to_patient_dict(train_flat)
    holdout_patients = convert_to_patient_dict(holdout_flat)
    print(f"Train patient dict: {len(train_patients)} patients")
    print(f"Holdout patient dict: {len(holdout_patients)} patients")

    # =========================================================================
    # GAP HANDLING + SEGMENTATION (shared by both arms)
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("GAP HANDLING + SEGMENTATION")
    print("=" * 70)

    print("\nSegmenting training data...")
    train_segments = segment_all_patients(
        train_patients,
        imputation_threshold_mins=45,
        min_segment_length=MIN_SEGMENT_LENGTH,
    )

    # Use a portion of training segments for validation during fit
    # (holdout patients are reserved for final midnight eval only)
    seg_ids = list(train_segments.keys())
    np.random.seed(42)
    np.random.shuffle(seg_ids)
    split_idx = int(len(seg_ids) * 0.9)
    train_seg_dict = {k: train_segments[k] for k in seg_ids[:split_idx]}
    val_seg_dict = {k: train_segments[k] for k in seg_ids[split_idx:]}

    train_rows = sum(len(df) for df in train_seg_dict.values())
    val_rows = sum(len(df) for df in val_seg_dict.values())
    print(f"Train segments: {len(train_seg_dict)} ({train_rows:,} rows)")
    print(f"Val segments: {len(val_seg_dict)} ({val_rows:,} rows)")

    # =========================================================================
    # RUN EXPERIMENT ARMS
    # =========================================================================

    results = {}

    if args.arm in ("A", "both"):
        results["A"] = run_arm(
            "A",
            train_seg_dict,
            val_seg_dict,
            holdout_patients,
            covariate_cols=[],
            args=args,
        )

    if args.arm in ("B", "both"):
        results["B"] = run_arm(
            "B",
            train_seg_dict,
            val_seg_dict,
            holdout_patients,
            covariate_cols=TIME_COLS,
            args=args,
        )

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Arm':<6} {'Covariates':<25} {'Val RMSE (sliding)':>20} {'Midnight RMSE':>15}"
    )
    print("-" * 70)
    for arm_name, r in results.items():
        cov_str = ", ".join(r["covariates"]) if r["covariates"] else "BG-only"
        print(
            f"{arm_name:<6} {cov_str:<25} {r['val_rmse_sliding']:>20.4f} {r['midnight_rmse']:>15.4f}"
        )

    if "A" in results and "B" in results:
        delta = results["B"]["midnight_rmse"] - results["A"]["midnight_rmse"]
        pct = delta / results["A"]["midnight_rmse"] * 100
        print("-" * 70)
        print(f"{'Delta (B-A)':<31} {'':>20} {delta:>+15.4f} ({pct:+.1f}%)")

    print("\nReference baselines:")
    print("  Chronos-2 zero-shot:         2.555")
    print("  Chronos-2 FT (past IOB):     2.347")
    print("  Chronos-2 FT (BG-only):      2.385")

    # Save results
    results_dir = PROJECT_ROOT / "results" / "chronos2_time_covariate"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = (
        results_dir
        / f"results_{args.steps}steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(results_path, "w") as f:
        json.dump(
            {
                "results": results,
                "config": {
                    "steps": args.steps,
                    "lr": args.lr,
                    "context_length": CONTEXT_LENGTH,
                    "forecast_horizon": FORECAST_HORIZON,
                    "time_limit": args.time_limit,
                    "max_eval_episodes": args.max_eval_episodes,
                },
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
