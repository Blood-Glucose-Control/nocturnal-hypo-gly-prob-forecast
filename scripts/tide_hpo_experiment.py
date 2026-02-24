#!/usr/bin/env python3
"""
TiDE Hyperparameter Optimization Experiment

Runs AutoGluon's native Bayesian hyperparameter search to optimize TiDE
for nocturnal hypoglycemia forecasting.

Based on research findings from docs-internal/tide_hyperparameter_research.md
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
from autogluon.common import space  # noqa: E402

from src.data.diabetes_datasets.data_loader import get_loader  # noqa: E402
from src.data.models import ColumnNames  # noqa: E402
from src.data.preprocessing.gap_handling import segment_all_patients  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================

INTERVAL_MINS = 5
PREDICTION_LENGTH = 72  # 6 hours
TARGET_COL = ColumnNames.BG.value
IOB_COL = ColumnNames.IOB.value

# Fixed parameters (from research)
CONTEXT_LENGTH = 512
TEMPORAL_HIDDEN_DIM = 256
NUM_LAYERS_ENCODER = 2
NUM_LAYERS_DECODER = 2
SCALING = "mean"  # CRITICAL for discontinuity prevention

# HPO settings
NUM_TRIALS = 15
TIME_LIMIT_PER_TRIAL = 3600  # 1 hour per trial max
TOTAL_TIME_LIMIT = 18000  # 5 hours total (conservative for overnight)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def segment_and_prepare_data(patient_dict, min_segment_length):
    """Apply gap handling and segment patients."""
    print(f"\nSegmenting {len(patient_dict)} patients...")
    print(f"  Min segment length: {min_segment_length} steps")

    segmented = segment_all_patients(
        patients_data=patient_dict,
        bg_col=TARGET_COL,
        min_segment_length=min_segment_length,
        imputation_threshold_mins=60,
        expected_interval_mins=INTERVAL_MINS,
    )

    # Flatten segments to DataFrame
    all_segments = []
    for segment_id, seg_df in segmented.items():
        df = seg_df.copy()
        df["item_id"] = segment_id
        all_segments.append(df)

    combined = pd.concat(all_segments, ignore_index=False)
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Total rows: {len(combined)}")

    return combined


def format_for_autogluon(df, target_col, iob_col):
    """Format DataFrame for AutoGluon with IOB as known covariate."""
    # Reset index to get timestamp column
    df_reset = df.reset_index()

    # Rename columns for AutoGluon
    df_ag = df_reset.rename(
        columns={
            "datetime": "timestamp",
            target_col: "target",
            iob_col: "iob",
        }
    )

    # Select required columns
    df_ag = df_ag[["item_id", "timestamp", "target", "iob"]]

    # Convert to TimeSeriesDataFrame
    df_ag = df_ag.set_index(["item_id", "timestamp"])
    ts_df = TimeSeriesDataFrame(df_ag)

    return ts_df


def build_midnight_episodes(patient_df, target_col, iob_col, context_len, horizon):
    """Build midnight-anchored evaluation episodes."""
    df = patient_df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    has_iob = iob_col in df.columns and df[iob_col].notna().any()

    freq = f"{INTERVAL_MINS}min"
    grid = pd.date_range(
        df.index.min().floor(freq), df.index.max().floor(freq), freq=freq
    )
    df = df.reindex(grid)

    dt = pd.Timedelta(minutes=INTERVAL_MINS)
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

        cols_to_get = [target_col]
        if has_iob:
            cols_to_get.append(iob_col)

        window_df = df.reindex(window_index)[cols_to_get]

        if window_df[target_col].isna().any():
            continue

        context_df = window_df.iloc[:context_len].copy()
        forecast_df = window_df.iloc[context_len:].copy()

        target_bg = forecast_df[target_col].to_numpy()

        future_iob = None
        if has_iob:
            if context_df[iob_col].isna().mean() > 0.5:
                continue
            context_df[iob_col] = context_df[iob_col].ffill().fillna(0)
            future_iob = forecast_df[iob_col].ffill().fillna(0).to_numpy()

        episodes.append(
            {
                "anchor": anchor,
                "context_df": context_df,
                "target_bg": target_bg,
                "future_iob": future_iob,
            }
        )

    return episodes


def format_episodes_for_autogluon(episodes, target_col, iob_col, context_len):
    """Format episodes for AutoGluon prediction."""
    train_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"

        # Adjust context to match model's context length
        context_df = ep["context_df"].copy()
        if len(context_df) > context_len:
            context_df = context_df.iloc[-context_len:]

        df = context_df.copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[target_col]
        df["iob"] = df[iob_col]

        train_data_list.append(df[["item_id", "timestamp", "target", "iob"]])

        future_timestamps = pd.date_range(
            ep["anchor"], periods=PREDICTION_LENGTH, freq=f"{INTERVAL_MINS}min"
        )
        future_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": future_timestamps,
                "iob": ep["future_iob"],
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


def evaluate_on_midnight_episodes(predictor, val_data, max_episodes=500):
    """Evaluate predictor on midnight-anchored episodes."""
    print("\n" + "=" * 70)
    print("EVALUATION ON MIDNIGHT-ANCHORED EPISODES")
    print("=" * 70)

    # Build episodes
    print("\nBuilding evaluation episodes...")
    all_episodes = []
    for pid, pdf in val_data.items():
        eps = build_midnight_episodes(
            pdf, TARGET_COL, IOB_COL, CONTEXT_LENGTH, PREDICTION_LENGTH
        )
        all_episodes.extend(eps)

    all_episodes = all_episodes[:max_episodes]
    print(f"Using {len(all_episodes)} evaluation episodes")

    # Format for AutoGluon
    ts_eval, known_cov = format_episodes_for_autogluon(
        all_episodes, TARGET_COL, IOB_COL, CONTEXT_LENGTH
    )

    # Make predictions
    print("Running predictions...")
    predictions = predictor.predict(ts_eval, known_covariates=known_cov)

    # Compute metrics
    rmse_list = []
    discont_list = []

    for i, ep in enumerate(all_episodes):
        item_id = f"ep_{i:03d}"

        if item_id not in predictions.index.get_level_values(0):
            continue

        pred = predictions.loc[item_id]["mean"].values
        actual = ep["target_bg"][: len(pred)]

        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        rmse_list.append(rmse)

        # Discontinuity
        last_context = ep["context_df"][TARGET_COL].iloc[-1]
        first_forecast = pred[0]
        discont = abs(last_context - first_forecast)
        discont_list.append(discont)

    avg_rmse = np.mean(rmse_list)
    avg_discont = np.mean(discont_list)
    variance = np.var(np.concatenate([pred for _, ep in enumerate(all_episodes)]))

    print(f"\n{'Metric':<30s} {'Value'}")
    print("-" * 50)
    print(f"{'RMSE':<30s} {avg_rmse:.4f} mM")
    print(f"{'Boundary discontinuity':<30s} {avg_discont:.4f} mM")
    print(f"{'Prediction variance':<30s} {variance:.4f}")

    return {
        "rmse": avg_rmse,
        "discontinuity": avg_discont,
        "variance": variance,
        "num_episodes": len(rmse_list),
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="TiDE HPO Experiment")
    parser.add_argument(
        "--num-trials", type=int, default=NUM_TRIALS, help="Number of HPO trials"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=TOTAL_TIME_LIMIT,
        help="Total time limit in seconds",
    )
    parser.add_argument(
        "--max-eval-episodes",
        type=int,
        default=500,
        help="Max episodes for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/tide_hpo",
        help="Output directory",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TiDE HYPERPARAMETER OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Num trials: {args.num_trials}")
    print(f"Time limit: {args.time_limit}s ({args.time_limit/3600:.1f} hours)")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    loader = get_loader("brown_2019", "train", use_cached=True)

    train_data = loader.train_data
    val_data = loader.validation_data

    print(f"Training patients: {len(train_data)}")
    print(f"Validation patients: {len(val_data)}")

    # ========================================================================
    # GAP HANDLING + SEGMENTATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("GAP HANDLING + SEGMENTATION")
    print("=" * 70)

    min_segment_length = CONTEXT_LENGTH + PREDICTION_LENGTH

    train_segmented = segment_and_prepare_data(train_data, min_segment_length)

    # Format for AutoGluon
    print("\nFormatting for AutoGluon...")
    train_formatted = format_for_autogluon(train_segmented, TARGET_COL, IOB_COL)

    print(f"Training data shape: {train_formatted.shape}")
    print(f"  Unique series: {train_formatted.num_items}")

    # ========================================================================
    # DEFINE SEARCH SPACE
    # ========================================================================
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH SPACE")
    print("=" * 70)

    # CRITICAL CONSTRAINT: TiDE requires encoder_hidden_dim == decoder_hidden_dim
    # Fix to 256 (best manual config) and tune other parameters
    search_space = {
        "TiDE": {
            # FIXED - encoder/decoder must match (architectural constraint)
            "encoder_hidden_dim": 256,
            "decoder_hidden_dim": 256,
            # HIGH IMPACT - Tune these
            "num_batches_per_epoch": space.Categorical(150, 200, 300),
            "lr": space.Real(5e-4, 2e-3, log=True),
            # MEDIUM IMPACT - Tune these
            "dropout": space.Categorical(0.1, 0.2, 0.3),
            "distr_hidden_dim": space.Categorical(8, 16, 32),
            # FIXED - Don't tune (from research)
            "context_length": CONTEXT_LENGTH,
            "temporal_hidden_dim": TEMPORAL_HIDDEN_DIM,
            "num_layers_encoder": NUM_LAYERS_ENCODER,
            "num_layers_decoder": NUM_LAYERS_DECODER,
            "scaling": SCALING,
            "known_covariates_names": ["iob"],
            "trainer_kwargs": {
                "gradient_clip_val": 1.0,
                "precision": "16-mixed",
            },
        }
    }

    print("Search space (with Ray Bayesian optimization):")
    print("  num_batches_per_epoch: [150, 200, 300]")
    print("  lr: Real(5e-4, 2e-3, log=True)")
    print("  dropout: [0.1, 0.2, 0.3]")
    print("  distr_hidden_dim: [8, 16, 32]")
    print("\nFixed parameters:")
    print("  encoder_hidden_dim: 256 (= decoder_hidden_dim)")
    print("  decoder_hidden_dim: 256 (architectural constraint)")
    print(f"  context_length: {CONTEXT_LENGTH}")
    print(f"  temporal_hidden_dim: {TEMPORAL_HIDDEN_DIM}")
    print(f"  scaling: {SCALING}")

    # ========================================================================
    # RUN HPO
    # ========================================================================
    print("\n" + "=" * 70)
    print("RUNNING HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        target="target",
        eval_metric="RMSE",
        path=str(output_dir),
    )

    print(f"\nStarting HPO with {args.num_trials} trials...")
    print("Using Bayesian optimization via HyperOpt (AutoGluon default)")

    predictor.fit(
        train_data=train_formatted,
        hyperparameters=search_space,
        hyperparameter_tune_kwargs={
            "num_trials": args.num_trials,
            "searcher": "auto",  # Bayesian for GluonTS models
            "scheduler": "local",
        },
        time_limit=args.time_limit,
        enable_ensemble=False,
        skip_model_selection=False,
        verbosity=2,
    )

    print("\nHPO complete!")

    # Get leaderboard
    leaderboard = predictor.leaderboard(train_formatted, silent=False)
    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    print(leaderboard)

    # Save leaderboard
    leaderboard_path = output_dir / "hpo_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"\nLeaderboard saved to: {leaderboard_path}")

    # ========================================================================
    # EVALUATE BEST MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("EVALUATING BEST MODEL ON MIDNIGHT EPISODES")
    print("=" * 70)

    eval_results = evaluate_on_midnight_episodes(
        predictor, val_data, max_episodes=args.max_eval_episodes
    )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results = {
        "config": "TiDE HPO with Ray Bayesian Optimization",
        "num_trials": args.num_trials,
        "time_limit": args.time_limit,
        "search_space": {
            "num_batches_per_epoch": [150, 200, 300],
            "lr": [5e-4, 2e-3],
            "dropout": [0.1, 0.2, 0.3],
            "distr_hidden_dim": [8, 16, 32],
        },
        "fixed_params": {
            "encoder_hidden_dim": 256,
            "decoder_hidden_dim": 256,
            "context_length": CONTEXT_LENGTH,
            "temporal_hidden_dim": TEMPORAL_HIDDEN_DIM,
            "scaling": SCALING,
        },
        "evaluation": convert_numpy(eval_results),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_path}")
    print(f"Best model RMSE: {eval_results['rmse']:.4f} mM")
    print(f"Best model discontinuity: {eval_results['discontinuity']:.4f} mM")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
