#!/usr/bin/env python3
"""
TiDE Validation Experiment: Test AutoGluon's discontinuity prevention.

Trains TiDE from scratch on Brown 2019 with IOB covariates.
Explicitly measures boundary discontinuity to validate AutoGluon's scaling approach.

Two configurations:
1. --config default: TiDE's design (144 context, 4 dims, 1 layer)
2. --config scaled: Chronos-2 parity (512 context, 256 dims, fair comparison)

Expected outcome:
- Smooth predictions (discontinuity < 0.2 mM)
- RMSE: 2.0-2.5 range (between zero-shot 2.555 and fine-tuned 1.890)

USAGE:
    # Default TiDE design
    python scripts/tide_validation_experiment.py --config default

    # Scaled for Chronos-2 parity
    python scripts/tide_validation_experiment.py --config scaled
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
from src.data.diabetes_datasets.data_loader import get_loader  # noqa: E402
from src.data.models import ColumnNames  # noqa: E402
from src.data.preprocessing.gap_handling import segment_all_patients  # noqa: E402

# Configuration
INTERVAL_MINS = 5
FORECAST_HORIZON = 72  # 6 hours prediction
TARGET_COL = ColumnNames.BG.value
IOB_COL = ColumnNames.IOB.value


# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================


def get_tide_config(config_name: str, prediction_length: int):
    """
    Get TiDE hyperparameters for specified configuration.

    Args:
        config_name: "default" or "scaled"
        prediction_length: Forecast horizon length

    Returns:
        Tuple of (context_length, hyperparameters_dict, description)
    """
    if config_name == "default":
        # TiDE's design: defaults from GluonTS
        context_length = max(64, 2 * prediction_length)  # 144 for pred_len=72
        hyperparameters = {
            "TiDE": {
                "context_length": context_length,
                "scaling": "mean",  # Per-window mean absolute scaling
                "num_batches_per_epoch": 100,  # See more data per epoch
                "trainer_kwargs": {
                    "gradient_clip_val": 1.0,
                },
                # All other params use defaults:
                # encoder_hidden_dim=4, decoder_hidden_dim=4, temporal_hidden_dim=4
                # num_layers_encoder=1, num_layers_decoder=1
                # dropout_rate=0.3, layer_norm=False
                # lr=1e-3, batch_size=64, max_epochs=100
            }
        }
        description = "TiDE Default Design (144 context, 4 dims, 1 layer)"

    elif config_name == "scaled":
        # Scaled for Chronos-2 parity
        context_length = 512  # Match Chronos-2 (42.7 hours)
        hyperparameters = {
            "TiDE": {
                "context_length": context_length,
                "encoder_hidden_dim": 256,  # Avoid bottleneck with 512 context
                "decoder_hidden_dim": 256,
                "temporal_hidden_dim": 256,  # Critical for IOB alignment
                "scaling": "mean",  # Per-window mean absolute scaling
                "batch_size": 256,  # H200 can handle large batches
                "num_batches_per_epoch": 200,  # See more data per epoch
                "lr": 1e-3,  # GluonTS default (not 1e-4!)
                "trainer_kwargs": {
                    "gradient_clip_val": 1.0,
                    "precision": "16-mixed",  # Speed up training
                },
            }
        }
        description = "TiDE Scaled for Chronos-2 Parity (512 context, 256 dims)"

    else:
        raise ValueError(f"Unknown config: {config_name}. Use 'default' or 'scaled'")

    return context_length, hyperparameters, description


# =============================================================================
# DATA FORMATTING (same as Chronos-2 script)
# =============================================================================


def format_segments_for_autogluon(segments, target_col, iob_col):
    """
    Convert gap-handled segments to AutoGluon TimeSeriesDataFrame with IOB.

    Identical to chronos2_sliding_gap_experiment.py.
    """
    data_list = []

    for seg_id, seg_df in segments.items():
        cols = [target_col]
        has_iob = iob_col in seg_df.columns and seg_df[iob_col].notna().any()

        df = seg_df[cols].copy()
        df = df.rename(columns={target_col: "target"})

        # Include IOB if available
        if has_iob:
            df["iob"] = seg_df[iob_col].ffill().fillna(0)
        else:
            df["iob"] = 0.0

        df["item_id"] = seg_id
        df["timestamp"] = df.index
        data_list.append(df[["item_id", "timestamp", "target", "iob"]])

    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.set_index(["item_id", "timestamp"])
    return TimeSeriesDataFrame(combined)


def build_midnight_episodes_with_iob(
    patient_df, target_col, iob_col, interval_mins, context_len, horizon
):
    """
    Build midnight-anchored episodes including IOB data.

    Identical to chronos2_sliding_gap_experiment.py.
    """
    df = patient_df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    has_iob = iob_col in df.columns and df[iob_col].notna().any()
    if not has_iob:
        return []

    freq = f"{interval_mins}min"
    grid = pd.date_range(
        df.index.min().floor(freq), df.index.max().floor(freq), freq=freq
    )
    df = df.reindex(grid)

    dt = pd.Timedelta(minutes=interval_mins)
    earliest = df.index.min() + context_len * dt
    latest = df.index.max() - (horizon - 1) * dt

    first_midnight = earliest.normalize()
    if first_midnight < earliest:
        first_midnight += pd.Timedelta(days=1)

    last_midnight = latest.normalize()
    if last_midnight < first_midnight:
        return []

    episodes = []
    for anchor in pd.date_range(first_midnight, last_midnight, freq="D"):
        window_start = anchor - context_len * dt
        window_end = anchor + horizon * dt
        window_index = pd.date_range(
            window_start, window_end, freq=freq, inclusive="left"
        )

        cols_to_get = [target_col, iob_col]
        window_df = df.reindex(window_index)[cols_to_get]

        if window_df[target_col].isna().any():
            continue

        context_df = window_df.iloc[:context_len].copy()
        forecast_df = window_df.iloc[context_len:].copy()

        if context_df[iob_col].isna().mean() > 0.5:
            continue

        target_bg = forecast_df[target_col].to_numpy()
        future_iob = forecast_df[iob_col].to_numpy()

        context_df[iob_col] = context_df[iob_col].ffill().fillna(0)
        future_iob = pd.Series(future_iob).ffill().fillna(0).to_numpy()

        episodes.append(
            {
                "anchor": anchor,
                "context_df": context_df,
                "target_bg": target_bg,
                "future_iob": future_iob,
            }
        )

    return episodes


def format_for_autogluon_with_known_covariates(
    episodes, target_col, iob_col, forecast_horizon=72
):
    """
    Convert episodes to AutoGluon TimeSeriesDataFrame format with IOB covariate.

    Identical to chronos2_sliding_gap_experiment.py.
    """
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"

        # Training data: context with BG and IOB
        df = ep["context_df"].copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[target_col]
        df["iob"] = df[iob_col] if iob_col in df.columns else 0.0
        train_data_list.append(df[["item_id", "timestamp", "target", "iob"]])

        # Known covariates: future IOB trajectory
        future_iob = ep["future_iob"][:forecast_horizon]
        future_timestamps = pd.date_range(
            ep["anchor"], periods=len(future_iob), freq=f"{INTERVAL_MINS}min"
        )
        future_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": future_timestamps,
                "iob": future_iob,
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
# EVALUATION WITH DISCONTINUITY MEASUREMENT
# =============================================================================


def evaluate_with_discontinuity_check(predictor, test_data, known_covariates, episodes):
    """
    Evaluate model with known covariates and measure discontinuity.

    Returns:
        Dictionary with RMSE, discontinuity, and variance statistics.
    """
    print("\nRunning predictions with known covariates...")

    predictions = predictor.predict(test_data, known_covariates=known_covariates)

    rmse_list = []
    discontinuities = []
    variances = []
    mean_bgs = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"
        if item_id not in predictions.index.get_level_values(0):
            continue

        # Get prediction
        pred = predictions.loc[item_id]["mean"].values
        actual = ep["target_bg"][: len(pred)]

        # RMSE
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        rmse_list.append(rmse)

        # Discontinuity: jump at boundary
        last_context = ep["context_df"][TARGET_COL].iloc[-1]
        first_forecast = pred[0]
        discontinuity = abs(last_context - first_forecast)
        discontinuities.append(discontinuity)

        # Variance: check for mean reversion (flat predictions)
        variance = np.var(pred)
        variances.append(variance)

        # Patient BG baseline
        mean_bg = ep["context_df"][TARGET_COL].mean()
        mean_bgs.append(mean_bg)

    # Overall statistics
    results = {
        "avg_rmse": np.mean(rmse_list),
        "avg_discontinuity": np.mean(discontinuities),
        "avg_variance": np.mean(variances),
        "num_episodes": len(rmse_list),
    }

    # By BG range (check for bias)
    df = pd.DataFrame(
        {
            "rmse": rmse_list,
            "discontinuity": discontinuities,
            "variance": variances,
            "mean_bg": mean_bgs,
        }
    )
    df["bg_range"] = pd.cut(
        df["mean_bg"],
        bins=[0, 6, 8, 15],
        labels=["Low (4-6)", "Mid (6-8)", "High (8+)"],
    )

    by_range = df.groupby("bg_range").agg(
        {
            "discontinuity": "mean",
            "rmse": "mean",
            "variance": "mean",
        }
    )
    results["by_bg_range"] = by_range.to_dict()

    return results, predictions


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TiDE validation experiment with two configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["default", "scaled"],
        required=True,
        help="Configuration preset: 'default' (TiDE design) or 'scaled' (Chronos-2 parity)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=7200,
        help="Time limit in seconds (default: 2 hours)",
    )
    parser.add_argument(
        "--max-eval-episodes",
        type=int,
        default=500,
        help="Max evaluation episodes",
    )

    args = parser.parse_args()

    # Get configuration
    context_length, hyperparameters, config_desc = get_tide_config(
        args.config, FORECAST_HORIZON
    )

    print("=" * 70)
    print("TiDE VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration: {config_desc}")
    print(
        f"Context length: {context_length} steps ({context_length * INTERVAL_MINS / 60:.1f} hours)"
    )
    print(
        f"Forecast horizon: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * INTERVAL_MINS / 60:.1f} hours)"
    )
    print("\nKey insight: Testing AutoGluon's per-window scaling approach")
    print("Expected: Smooth predictions without discontinuity at boundary")

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    loader = get_loader(
        data_source_name="brown_2019",
        dataset_type="train",
        use_cached=True,
    )

    train_data = loader.train_data
    val_data = loader.validation_data

    print(f"Training patients: {len(train_data)}")
    print(f"Validation patients: {len(val_data)}")

    # Check IOB availability
    sample_patient = list(train_data.values())[0]
    print(f"\nSample patient columns: {list(sample_patient.columns)}")
    if IOB_COL in sample_patient.columns:
        iob_coverage = sample_patient[IOB_COL].notna().mean() * 100
        print(f"IOB coverage in sample patient: {iob_coverage:.1f}%")
    else:
        print("WARNING: IOB column not found!")
        return

    # =========================================================================
    # GAP HANDLING + SEGMENTATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("GAP HANDLING + SEGMENTATION")
    print("=" * 70)

    MIN_SEGMENT_LENGTH = context_length + FORECAST_HORIZON

    print(f"\nMinimum segment length: {MIN_SEGMENT_LENGTH} steps")
    print("Segmenting training data...")
    train_segments = segment_all_patients(
        train_data,
        imputation_threshold_mins=45,
        min_segment_length=MIN_SEGMENT_LENGTH,
    )

    print("\nSegmenting validation data...")
    val_segments = segment_all_patients(
        val_data,
        imputation_threshold_mins=45,
        min_segment_length=MIN_SEGMENT_LENGTH,
    )

    # Statistics
    train_rows = sum(len(df) for df in train_segments.values())
    val_rows = sum(len(df) for df in val_segments.values())

    print(f"\nTrain: {len(train_segments)} segments ({train_rows:,} rows)")
    print(f"Val: {len(val_segments)} segments ({val_rows:,} rows)")

    # Format for AutoGluon
    print("\nFormatting segments for AutoGluon (with IOB covariate)...")
    ts_train = format_segments_for_autogluon(train_segments, TARGET_COL, IOB_COL)
    ts_val = format_segments_for_autogluon(val_segments, TARGET_COL, IOB_COL)

    print(f"Training data: {ts_train.shape}, columns={list(ts_train.columns)}")
    print(f"Validation data: {ts_val.shape}")

    # =========================================================================
    # TRAINING
    # =========================================================================

    output_dir = str(PROJECT_ROOT / f"models/tide_validation/{args.config}")

    print(f"\n{'='*70}")
    print(f"TRAINING: {config_desc}")
    print(f"{'='*70}")
    print(f"Training data shape: {ts_train.shape}")
    print(f"Series count: {ts_train.num_items}")
    print("\nHyperparameters:")
    for key, value in hyperparameters["TiDE"].items():
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

    leaderboard = predictor.leaderboard(ts_val)
    print(leaderboard)

    train_rmse = -leaderboard["score_val"].values[0]
    print(f"\nValidation RMSE (on segments): {train_rmse:.4f}")
    print(f"Model saved to: {output_dir}")

    # =========================================================================
    # EVALUATION ON MIDNIGHT-ANCHORED EPISODES (with discontinuity check)
    # =========================================================================

    print("\n" + "=" * 70)
    print("EVALUATION ON MIDNIGHT-ANCHORED EPISODES")
    print("=" * 70)

    # Build evaluation episodes with IOB
    print("\nBuilding evaluation episodes with IOB...")
    eval_episodes = []
    for pid, pdf in val_data.items():
        eps = build_midnight_episodes_with_iob(
            pdf, TARGET_COL, IOB_COL, INTERVAL_MINS, context_length, FORECAST_HORIZON
        )
        eval_episodes.extend(eps)
        if len(eval_episodes) >= args.max_eval_episodes:
            break

    eval_episodes = eval_episodes[: args.max_eval_episodes]
    print(f"Using {len(eval_episodes)} evaluation episodes")

    # Format for AutoGluon evaluation
    ts_eval, known_cov_eval = format_for_autogluon_with_known_covariates(
        eval_episodes, TARGET_COL, IOB_COL, forecast_horizon=FORECAST_HORIZON
    )

    # Evaluate with discontinuity measurement
    print("\nEvaluating with discontinuity measurement...")
    results, predictions = evaluate_with_discontinuity_check(
        predictor, ts_eval, known_cov_eval, eval_episodes
    )

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nConfiguration: {config_desc}")
    print(f"Episodes evaluated: {results['num_episodes']}")

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'RMSE':<30} {results['avg_rmse']:>10.4f} mM")
    print(f"{'Boundary discontinuity':<30} {results['avg_discontinuity']:>10.4f} mM")
    print(f"{'Prediction variance':<30} {results['avg_variance']:>10.4f}")

    print("\nBy BG Range:")
    for bg_range, metrics in results["by_bg_range"]["discontinuity"].items():
        disc = metrics
        rmse = results["by_bg_range"]["rmse"][bg_range]
        var = results["by_bg_range"]["variance"][bg_range]
        print(
            f"  {bg_range}: discontinuity={disc:.3f} mM, RMSE={rmse:.3f}, variance={var:.3f}"
        )

    print("\n" + "-" * 70)
    print("Comparison to Chronos-2:")
    print("  Zero-shot Chronos-2:       2.555 RMSE")
    print("  Fine-tuned Chronos-2 (P1): 1.890 RMSE")
    print(f"  TiDE ({args.config}):      {results['avg_rmse']:.3f} RMSE")

    # Validation check
    print("\n" + "=" * 70)
    print("VALIDATION CHECK")
    print("=" * 70)

    success = True
    if results["avg_discontinuity"] < 0.2:
        print("✅ SUCCESS: Discontinuity < 0.2 mM (smooth predictions)")
    else:
        print(
            f"❌ FAIL: Discontinuity = {results['avg_discontinuity']:.3f} mM (> 0.2 mM threshold)"
        )
        success = False

    if results["avg_variance"] > 0.1:
        print("✅ SUCCESS: Variance > 0.1 (not mean reversion)")
    else:
        print(
            f"❌ FAIL: Variance = {results['avg_variance']:.3f} (possible mean reversion)"
        )
        success = False

    if success:
        print("\n🎉 AutoGluon's per-window scaling approach VALIDATED!")
        print("   TTM fix should follow this pattern (Option 2 or 3)")
    else:
        print("\n⚠️  Unexpected results - investigate further")

    # Save results
    results_path = PROJECT_ROOT / f"models/tide_validation/{args.config}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(
            {
                "config": args.config,
                "description": config_desc,
                "context_length": context_length,
                "results": {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in results.items()
                    if k != "by_bg_range"
                },
                "by_bg_range": {
                    bg_range: {
                        metric: {k: float(v) for k, v in values.items()}
                        for metric, values in results["by_bg_range"].items()
                    }
                    for bg_range in results["by_bg_range"]["discontinuity"].keys()
                },
                "evaluated_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
