#!/usr/bin/env python3
"""
Chronos-2 Covariate A/B/C/D Experiment.

Fine-tunes Chronos-2 on Brown 2019 in matched arms:
  Arm A: BG-only (no covariates)
  Arm B: BG + time-of-day features (hour_sin, hour_cos as known future covariates)
  Arm C: BG + IOB (past-only context, NOT known future — avoids leakage)
  Arm D: BG + IOB (past-only) + time-of-day (known future)

Uses DatasetRegistry for holdout splits (patient + temporal holdout).
Training: gap-handled segments with AutoGluon sliding windows.
Evaluation: midnight-anchored episodes on holdout patients.

COVARIATE HANDLING:
  - "known future" covariates (time features): declared in known_covariates_names,
    provided for the forecast horizon at inference. Legitimate because deterministic.
  - "past-only" covariates (IOB): included in training data as extra columns but
    NOT declared as known_covariates_names. The model sees them in context only.
    This avoids data leakage — on AID (Brown 2019), future IOB is entangled with
    future BG through the closed-loop controller.

USAGE:
    python scripts/chronos2_time_covariate_experiment.py --arm CD
    python scripts/chronos2_time_covariate_experiment.py --arm all --steps 10000
    python scripts/chronos2_time_covariate_experiment.py --arm D
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
IOB_COL = "iob"
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


def format_episodes_for_eval(
    episodes, all_covariate_cols=None, known_covariate_cols=None, forecast_horizon=72
):
    """
    Convert midnight episodes to AutoGluon format for evaluation.

    Args:
        episodes: List of episode dicts from build_midnight_episodes.
        all_covariate_cols: ALL covariate columns to include in context
            (both past-only and known-future).
        known_covariate_cols: Subset of covariates that are known for the
            future (e.g. time features). Only these get future values.
            Past-only covariates (e.g. IOB) appear in context only.
        forecast_horizon: Number of future steps.

    Returns:
        (test_data, known_covariates) or (test_data, None) if no known covariates.
    """
    all_covariate_cols = all_covariate_cols or []
    known_covariate_cols = known_covariate_cols or []
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"

        # Context data — includes ALL covariates (past-only + known-future)
        df = ep["context_df"].copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[TARGET_COL]

        keep_cols = ["item_id", "timestamp", "target"]
        for col in all_covariate_cols:
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

        # Future values — ONLY for known-future covariates (not past-only like IOB)
        if known_covariate_cols:
            anchor = ep["anchor"]
            future_timestamps = pd.date_range(
                anchor, periods=forecast_horizon, freq=f"{INTERVAL_MINS}min"
            )

            future_data = {"item_id": item_id, "timestamp": future_timestamps}
            for col in known_covariate_cols:
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
    if known_covariate_cols and known_cov_list:
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

    overall_rmse = float(np.mean(rmse_list))  # Mean of per-episode RMSEs
    return overall_rmse, rmse_list


def run_arm(
    arm_name,
    train_segments,
    val_segments,
    eval_patient_dict,
    past_covariate_cols,
    known_covariate_cols,
    args,
):
    """Run one arm of the experiment (train + evaluate).

    Args:
        past_covariate_cols: Covariates included in training data as context
            columns only (e.g. IOB). NOT declared as known_covariates_names,
            so the model only sees them in the context window, not the future.
        known_covariate_cols: Covariates declared as known future (e.g. time
            features). These are deterministic and provided for the forecast
            horizon at inference time.
    """
    all_covariate_cols = past_covariate_cols + known_covariate_cols

    print(f"\n{'=' * 70}")
    print(
        f"ARM {arm_name}: BG + [{', '.join(all_covariate_cols) if all_covariate_cols else 'none'}]"
    )
    print(f"  Past-only (context): {past_covariate_cols}")
    print(f"  Known-future:        {known_covariate_cols}")
    print(f"{'=' * 70}")

    # Format segments for AutoGluon — ALL covariates go into training data
    print("\nFormatting segments for AutoGluon...")
    ts_train = format_segments_for_autogluon(
        train_segments, TARGET_COL, all_covariate_cols or None
    )
    ts_val = format_segments_for_autogluon(
        val_segments, TARGET_COL, all_covariate_cols or None
    )

    print(f"Training data: {ts_train.shape}, columns={list(ts_train.columns)}")
    print(f"Validation data: {ts_val.shape}")

    # Set up predictor — only KNOWN covariates declared as known_covariates_names
    covariates_label = "_".join(all_covariate_cols) if all_covariate_cols else "bg_only"
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
    if known_covariate_cols:
        predictor_kwargs["known_covariates_names"] = known_covariate_cols

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
    # build_midnight_episodes requires at least one covariate (defaults to ["iob"]
    # if None). We always include IOB for episode building to satisfy its contract.
    print("\nBuilding midnight evaluation episodes...")
    episode_covs = list(set(all_covariate_cols + [IOB_COL]))
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

    # Format for evaluation — all covariates in context, only known in future
    ts_eval, known_cov_eval = format_episodes_for_eval(
        eval_episodes,
        all_covariate_cols=all_covariate_cols or None,
        known_covariate_cols=known_covariate_cols or None,
        forecast_horizon=FORECAST_HORIZON,
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
        "past_covariates": past_covariate_cols,
        "known_covariates": known_covariate_cols,
        "all_covariates": all_covariate_cols,
        "val_rmse_sliding": val_rmse,
        "midnight_rmse": midnight_rmse,
        "eval_episodes": len(eval_episodes),
        "per_episode_rmse_mean": float(np.mean(per_ep_errors)),
        "per_episode_rmse_std": float(np.std(per_ep_errors)),
        "model_path": output_dir,
    }


# Arm definitions: (past_covariate_cols, known_covariate_cols)
ARM_CONFIGS = {
    "A": ([], []),  # BG-only
    "B": ([], TIME_COLS),  # BG + time (known future)
    "C": ([IOB_COL], []),  # BG + IOB (past-only)
    "D": ([IOB_COL], TIME_COLS),  # BG + IOB (past-only) + time (known future)
}


def main():
    parser = argparse.ArgumentParser(
        description="Chronos-2 Covariate A/B/C/D Experiment",
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
        default="CD",
        choices=["A", "B", "C", "D", "AB", "CD", "all"],
        help="Which arm(s) to run: A (BG-only), B (BG+time), "
        "C (BG+IOB past-only), D (BG+IOB+time), AB, CD, or all",
    )
    args = parser.parse_args()

    # Determine which arms to run
    if args.arm == "all":
        arms_to_run = ["A", "B", "C", "D"]
    elif len(args.arm) == 2:
        arms_to_run = list(args.arm)
    else:
        arms_to_run = [args.arm]

    print("=" * 70)
    print("CHRONOS-2 COVARIATE A/B/C/D EXPERIMENT")
    print("=" * 70)
    print("\nArm A: BG-only (no covariates)")
    print("Arm B: BG + time-of-day (hour_sin, hour_cos as known future)")
    print("Arm C: BG + IOB (past-only context, NOT known future)")
    print("Arm D: BG + IOB (past-only) + time-of-day (known future)")
    print(f"\nRunning arms: {arms_to_run}")
    print("\nParameters:")
    print(f"  steps: {args.steps}")
    print(f"  lr: {args.lr}")
    print(f"  time_limit: {args.time_limit}s")
    print(f"  max_eval_episodes: {args.max_eval_episodes}")

    # =========================================================================
    # LOAD DATA VIA REGISTRY (patient + temporal holdout)
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("LOADING DATA VIA DATASET REGISTRY")
    print("=" * 70)

    registry = DatasetRegistry(holdout_config_dir="configs/data/holdout_10pct")
    train_flat, holdout_flat = registry.load_dataset_with_split("brown_2019")

    print(f"Train: {len(train_flat):,} rows, {train_flat['p_num'].nunique()} patients")
    print(
        f"Holdout: {len(holdout_flat):,} rows, {holdout_flat['p_num'].nunique()} patients"
    )

    # Verify required features exist
    for col in TIME_COLS:
        assert (
            col in train_flat.columns
        ), f"{col} missing! Run: rm -rf cache/data/brown_2019/processed/ && python scripts/verify_time_features.py"
    assert IOB_COL in train_flat.columns, f"{IOB_COL} missing from training data!"
    print(f"Time features present: {TIME_COLS}")
    print(f"IOB feature present: {IOB_COL}")

    # Convert to patient dicts for gap handling (uses existing utility from utils.py)
    print("\nConverting to patient dicts...")
    train_patients = convert_to_patient_dict(train_flat)
    holdout_patients = convert_to_patient_dict(holdout_flat)
    print(f"Train patient dict: {len(train_patients)} patients")
    print(f"Holdout patient dict: {len(holdout_patients)} patients")

    # =========================================================================
    # GAP HANDLING + SEGMENTATION (shared by all arms)
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

    for arm_name in arms_to_run:
        past_covs, known_covs = ARM_CONFIGS[arm_name]
        results[arm_name] = run_arm(
            arm_name,
            train_seg_dict,
            val_seg_dict,
            holdout_patients,
            past_covariate_cols=past_covs,
            known_covariate_cols=known_covs,
            args=args,
        )

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Arm':<6} {'All Covariates':<30} {'Known Future':<20} "
        f"{'Val RMSE':>10} {'Midnight RMSE':>15}"
    )
    print("-" * 85)
    for arm_name, r in results.items():
        all_str = ", ".join(r["all_covariates"]) if r["all_covariates"] else "BG-only"
        known_str = (
            ", ".join(r["known_covariates"]) if r["known_covariates"] else "none"
        )
        print(
            f"{arm_name:<6} {all_str:<30} {known_str:<20} "
            f"{r['val_rmse_sliding']:>10.4f} {r['midnight_rmse']:>15.4f}"
        )

    # Print deltas for paired comparisons
    pairs = [
        ("A", "B", "time effect (no IOB)"),
        ("C", "D", "time effect (with IOB)"),
        ("A", "C", "IOB effect (no time)"),
        ("B", "D", "IOB effect (with time)"),
    ]
    print("-" * 85)
    for a, b, label in pairs:
        if a in results and b in results:
            delta = results[b]["midnight_rmse"] - results[a]["midnight_rmse"]
            pct = delta / results[a]["midnight_rmse"] * 100
            print(f"  Delta ({b}-{a}): {delta:+.4f} ({pct:+.1f}%)  [{label}]")

    print(
        "\nReference baselines (different eval methodology — not directly comparable):"
    )
    print("  Chronos-2 zero-shot:         2.555  (get_loader temporal split)")
    print("  Chronos-2 FT (past IOB):     2.347  (get_loader temporal split)")
    print("  Chronos-2 FT (BG-only):      2.385  (get_loader temporal split)")
    print(
        "  Arms A/B (this pipeline):    2.026/2.015  (DatasetRegistry patient holdout)"
    )

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
                    "arms_run": arms_to_run,
                },
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
