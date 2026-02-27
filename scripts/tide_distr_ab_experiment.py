#!/usr/bin/env python3
"""
TiDE Distribution A/B Experiment: StudentTOutput vs NormalOutput

Clean A/B test to measure the effect of the distribution head on prediction
interval width (band sharpness) while holding all else constant.

Design:
- A (baseline): StudentTOutput(beta=0.0) — TiDE default, heavy tails
- B (treatment): NormalOutput() — lighter tails, expected tighter bands

Both use identical architecture (512 context, 256 dims, 2 layers) and data
pipeline from the validated tide_validation_experiment.py.

Key metrics:
- RMSE (point accuracy — should be similar)
- 80% CI width (band sharpness — NormalOutput should be tighter)
- Empirical coverage (what % of true values fall within 10-90% CI)
- Boundary discontinuity (should be <0.2 mM for both)

USAGE:
    python scripts/tide_distr_ab_experiment.py --config student_t
    python scripts/tide_distr_ab_experiment.py --config normal
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
from gluonts.torch.distributions import NormalOutput, StudentTOutput  # noqa: E402

from src.data.diabetes_datasets.data_loader import get_loader  # noqa: E402
from src.data.models import ColumnNames  # noqa: E402
from src.data.preprocessing.gap_handling import segment_all_patients  # noqa: E402

# =============================================================================
# CONSTANTS — identical to tide_validation_experiment.py (scaled config)
# =============================================================================

INTERVAL_MINS = 5
FORECAST_HORIZON = 72  # 6 hours
CONTEXT_LENGTH = 512
TARGET_COL = ColumnNames.BG.value
IOB_COL = ColumnNames.IOB.value

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# =============================================================================
# A/B CONFIGS — only distr_output differs
# =============================================================================


def get_config(config_name: str):
    """Return (hyperparameters, label) for A/B config.

    Architecture, training, and scaling parameters are identical between
    configs.  Only distr_output changes.
    """
    # Shared architecture (validated "scaled" config)
    shared = {
        "context_length": CONTEXT_LENGTH,
        "encoder_hidden_dim": 256,
        "decoder_hidden_dim": 256,
        "temporal_hidden_dim": 256,
        "num_layers_encoder": 2,
        "num_layers_decoder": 2,
        "distr_hidden_dim": 32,  # Larger than default 4 for heteroscedastic variance
        "scaling": "mean",
        "batch_size": 256,
        "num_batches_per_epoch": 200,
        "lr": 1e-3,
        "known_covariates_names": ["iob"],
        "trainer_kwargs": {
            "gradient_clip_val": 1.0,
            "precision": "16-mixed",
        },
    }

    if config_name == "student_t":
        shared["distr_output"] = StudentTOutput(beta=0.0)
        label = "TiDE StudentT (default, heavy tails)"
    elif config_name == "normal":
        shared["distr_output"] = NormalOutput()
        label = "TiDE Normal (lighter tails)"
    else:
        raise ValueError(f"Unknown config: {config_name}. Use 'student_t' or 'normal'")

    return {"TiDE": shared}, label


# =============================================================================
# DATA PIPELINE — copied from tide_validation_experiment.py
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


def build_midnight_episodes_with_iob(
    patient_df, target_col, iob_col, interval_mins, context_len, horizon
):
    """Build midnight-anchored episodes including IOB data."""
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
        window_df = df.reindex(window_index)[[target_col, iob_col]]
        if window_df[target_col].isna().any():
            continue

        context_df = window_df.iloc[:context_len].copy()
        forecast_df = window_df.iloc[context_len:].copy()

        if context_df[iob_col].isna().mean() > 0.5:
            continue

        context_df[iob_col] = context_df[iob_col].ffill().fillna(0)
        future_iob = forecast_df[iob_col].ffill().fillna(0).to_numpy()

        episodes.append(
            {
                "anchor": anchor,
                "context_df": context_df,
                "target_bg": forecast_df[target_col].to_numpy(),
                "future_iob": future_iob,
            }
        )

    return episodes


def format_for_autogluon_with_known_covariates(episodes, target_col, iob_col):
    """Convert episodes to AutoGluon format with IOB as known covariate."""
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:04d}"
        df = ep["context_df"].copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[target_col]
        df["iob"] = df[iob_col] if iob_col in df.columns else 0.0
        train_data_list.append(df[["item_id", "timestamp", "target", "iob"]])

        future_iob = ep["future_iob"][:FORECAST_HORIZON]
        future_timestamps = pd.date_range(
            ep["anchor"], periods=len(future_iob), freq=f"{INTERVAL_MINS}min"
        )
        known_cov_list.append(
            pd.DataFrame(
                {
                    "item_id": item_id,
                    "timestamp": future_timestamps,
                    "iob": future_iob,
                }
            )
        )

    train_combined = pd.concat(train_data_list, ignore_index=True)
    train_combined = train_combined.set_index(["item_id", "timestamp"])
    train_data = TimeSeriesDataFrame(train_combined)

    known_combined = pd.concat(known_cov_list, ignore_index=True)
    known_combined = known_combined.set_index(["item_id", "timestamp"])
    known_covariates = TimeSeriesDataFrame(known_combined)

    return train_data, known_covariates


# =============================================================================
# EVALUATION — with quantile extraction for band width + coverage
# =============================================================================


def evaluate_with_uncertainty(predictor, test_data, known_covariates, episodes):
    """Evaluate with RMSE, discontinuity, band width, and coverage."""
    predictions = predictor.predict(test_data, known_covariates=known_covariates)

    rmse_list = []
    discont_list = []
    ci_widths = []  # Average 80% CI width per episode
    boundary_widths = []  # CI width at first forecast step (midnight)
    coverages = []  # Fraction of true values inside 80% CI

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:04d}"
        if item_id not in predictions.index.get_level_values(0):
            continue

        pred_df = predictions.loc[item_id]
        pred_mean = pred_df["mean"].values
        actual = ep["target_bg"][: len(pred_mean)]

        # RMSE
        rmse = np.sqrt(np.mean((pred_mean - actual) ** 2))
        rmse_list.append(rmse)

        # Discontinuity
        last_context = ep["context_df"][TARGET_COL].iloc[-1]
        discont = abs(last_context - pred_mean[0])
        discont_list.append(discont)

        # Quantile-based metrics (80% CI = 10th to 90th percentile)
        has_quantiles = "0.1" in pred_df.columns and "0.9" in pred_df.columns
        if has_quantiles:
            p10 = pred_df["0.1"].values
            p90 = pred_df["0.9"].values

            # Average CI width across the forecast horizon
            width = np.mean(p90 - p10)
            ci_widths.append(width)

            # Boundary CI width (first forecast step)
            boundary_widths.append(p90[0] - p10[0])

            # Empirical coverage: fraction of true values in [p10, p90]
            inside = (actual >= p10[: len(actual)]) & (actual <= p90[: len(actual)])
            coverages.append(np.mean(inside))

    results = {
        "num_episodes": len(rmse_list),
        "rmse": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "discontinuity": float(np.mean(discont_list)),
        "discontinuity_std": float(np.std(discont_list)),
    }

    if ci_widths:
        results.update(
            {
                "ci_width_80": float(np.mean(ci_widths)),
                "ci_width_80_std": float(np.std(ci_widths)),
                "boundary_ci_width": float(np.mean(boundary_widths)),
                "boundary_ci_width_std": float(np.std(boundary_widths)),
                "coverage_80": float(np.mean(coverages)),
                "coverage_80_std": float(np.std(coverages)),
            }
        )

    return results, predictions


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TiDE Distribution A/B Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["student_t", "normal"],
        required=True,
        help="Distribution head: 'student_t' (baseline) or 'normal' (treatment)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=7200,
        help="Training time limit in seconds (default: 2 hours)",
    )
    parser.add_argument(
        "--max-eval-episodes",
        type=int,
        default=500,
        help="Max evaluation episodes",
    )
    args = parser.parse_args()

    hyperparameters, config_label = get_config(args.config)
    output_dir = PROJECT_ROOT / f"models/tide_distr_ab/{args.config}"

    print("=" * 70)
    print("TiDE DISTRIBUTION A/B EXPERIMENT")
    print("=" * 70)
    print(f"Config: {args.config} — {config_label}")
    print(f"distr_output: {hyperparameters['TiDE']['distr_output']}")
    print(f"distr_hidden_dim: {hyperparameters['TiDE']['distr_hidden_dim']}")
    print(f"Output: {output_dir}")
    print(f"Time limit: {args.time_limit}s")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    loader = get_loader("brown_2019", "train", use_cached=True)
    train_patients = loader.train_data
    val_patients = loader.validation_data
    print(f"Train: {len(train_patients)} patients, Val: {len(val_patients)} patients")

    # =========================================================================
    # GAP HANDLING + SEGMENTATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("GAP HANDLING + SEGMENTATION")
    print("=" * 70)

    min_seg_len = CONTEXT_LENGTH + FORECAST_HORIZON
    print(f"Min segment length: {min_seg_len}")

    train_segments = segment_all_patients(
        train_patients,
        imputation_threshold_mins=45,
        min_segment_length=min_seg_len,
    )
    val_segments = segment_all_patients(
        val_patients,
        imputation_threshold_mins=45,
        min_segment_length=min_seg_len,
    )

    train_rows = sum(len(df) for df in train_segments.values())
    val_rows = sum(len(df) for df in val_segments.values())
    print(f"Train: {len(train_segments)} segments ({train_rows:,} rows)")
    print(f"Val: {len(val_segments)} segments ({val_rows:,} rows)")

    ts_train = format_segments_for_autogluon(train_segments, TARGET_COL, IOB_COL)
    print(f"AutoGluon train shape: {ts_train.shape}, series: {ts_train.num_items}")

    # =========================================================================
    # TRAINING
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {config_label}")
    print(f"{'=' * 70}")

    predictor = TimeSeriesPredictor(
        prediction_length=FORECAST_HORIZON,
        target="target",
        known_covariates_names=["iob"],
        eval_metric="WQL",  # Optimize for quantile quality, not just point accuracy
        quantile_levels=QUANTILE_LEVELS,
        path=str(output_dir),
    )

    predictor.fit(
        train_data=ts_train,
        hyperparameters=hyperparameters,
        time_limit=args.time_limit,
        enable_ensemble=False,
    )

    leaderboard = predictor.leaderboard(
        format_segments_for_autogluon(val_segments, TARGET_COL, IOB_COL)
    )
    print("\nLeaderboard:")
    print(leaderboard)

    # =========================================================================
    # EVALUATION ON MIDNIGHT-ANCHORED EPISODES
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION ON MIDNIGHT-ANCHORED EPISODES")
    print("=" * 70)

    eval_episodes = []
    for pid, pdf in val_patients.items():
        eps = build_midnight_episodes_with_iob(
            pdf, TARGET_COL, IOB_COL, INTERVAL_MINS, CONTEXT_LENGTH, FORECAST_HORIZON
        )
        eval_episodes.extend(eps)
        if len(eval_episodes) >= args.max_eval_episodes:
            break
    eval_episodes = eval_episodes[: args.max_eval_episodes]
    print(f"Evaluation episodes: {len(eval_episodes)}")

    ts_eval, known_cov_eval = format_for_autogluon_with_known_covariates(
        eval_episodes, TARGET_COL, IOB_COL
    )

    results, predictions = evaluate_with_uncertainty(
        predictor, ts_eval, known_cov_eval, eval_episodes
    )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nConfig: {args.config} — {config_label}")
    print(f"Episodes: {results['num_episodes']}")
    print()
    print(f"{'Metric':<30s} {'Value':>10s}")
    print("-" * 45)
    print(f"{'RMSE':<30s} {results['rmse']:>10.4f} mM")
    print(f"{'Discontinuity':<30s} {results['discontinuity']:>10.4f} mM")

    if "ci_width_80" in results:
        print(f"{'80% CI Width (avg)':<30s} {results['ci_width_80']:>10.4f} mM")
        print(
            f"{'80% CI Width at boundary':<30s} {results['boundary_ci_width']:>10.4f} mM"
        )
        print(f"{'Empirical coverage (80% CI)':<30s} {results['coverage_80']:>10.2%}")

    # Interpretation
    print("\n" + "-" * 45)
    if "coverage_80" in results:
        cov = results["coverage_80"]
        if cov > 0.85:
            print(
                f"Coverage {cov:.1%} > 85%: model is OVER-CONSERVATIVE (bands too wide)"
            )
        elif cov < 0.75:
            print(
                f"Coverage {cov:.1%} < 75%: model is OVERCONFIDENT (bands too narrow)"
            )
        else:
            print(f"Coverage {cov:.1%}: well-calibrated (target ~80%)")

    # =========================================================================
    # SAVE
    # =========================================================================
    results_out = {
        "config": args.config,
        "label": config_label,
        "distr_output": repr(hyperparameters["TiDE"]["distr_output"]),
        "distr_hidden_dim": hyperparameters["TiDE"]["distr_hidden_dim"],
        "eval_metric": "WQL",
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
