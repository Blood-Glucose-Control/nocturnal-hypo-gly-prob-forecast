#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Find stable baseline episodes for covariate sensitivity testing.

Searches TFT evaluation results for midnight-anchored episodes with:
- Relatively flat BG in early forecast window (std < 2.0 mmol/L)
- Minimal meal/bolus activity in late context window
- Suitable for synthetic insulin/carb injection testing

Usage:
    python scripts/analysis/find_stable_episodes.py
    python scripts/analysis/find_stable_episodes.py --top-n 10
    python scripts/analysis/find_stable_episodes.py --dataset aleppo_2017
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# TFT evaluation run directory
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "nocturnal_forecasting"
    / "512ctx_96fh"
    / "tft"
    / "2026-04-29_194226_aleppo_2017_finetuned"
)

# Dataset processed directory
DEFAULT_DATA_DIR = PROJECT_ROOT / "cache" / "data" / "aleppo_2017" / "processed"

# Episode selection criteria
FORECAST_STABILITY_HOURS = 2.5  # Hours from forecast start to check (2-3 hours)
FORECAST_STD_THRESHOLD = 0.8  # mmol/L - max std for "flat" BG (stricter)
FORECAST_TREND_THRESHOLD = 0.1  # mmol/L per hour - max trend (stricter)

CONTEXT_QUIET_HOURS = 5.0  # Hours before forecast to check
MEAL_SUM_THRESHOLD = 10.0  # grams - max total carbs in quiet window
BOLUS_MAX_THRESHOLD = 0.5  # units - max insulin per step (allows basal)

# Target BG range at forecast start
TARGET_BG_MIN = 5.0  # mmol/L
TARGET_BG_MAX = 6.0  # mmol/L

INTERVAL_MINS = 5  # Sampling interval
TOP_N_DEFAULT = 5  # Default number of candidates to return

# ══════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ══════════════════════════════════════════════════════════════════════════════


def load_evaluation_data(run_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load forecasts.npz and episodes.parquet from evaluation run.

    Returns:
        actuals: (n_episodes, forecast_length)
        q_forecasts: (n_episodes, n_quantiles, forecast_length)
        episodes_df: DataFrame with episode metadata
    """
    npz_path = run_dir / "forecasts.npz"
    parquet_path = run_dir / "episodes.parquet"

    if not npz_path.exists():
        raise FileNotFoundError(f"forecasts.npz not found: {npz_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"episodes.parquet not found: {parquet_path}")

    data = np.load(npz_path, allow_pickle=False)
    episodes_df = pd.read_parquet(parquet_path)

    actuals = data["actuals"]
    q_forecasts = data["quantile_forecasts"]

    print(f"Loaded {len(episodes_df)} episodes from {run_dir.name}")
    print(f"  Actuals shape: {actuals.shape}")
    print(f"  Quantile forecasts shape: {q_forecasts.shape}")

    return actuals, q_forecasts, episodes_df


def load_patient_context(
    patient_id: str,
    anchor: pd.Timestamp,
    data_dir: Path,
    context_hours: float = 8.0,
    interval_mins: int = INTERVAL_MINS,
) -> pd.DataFrame:
    """Load context window data for a patient-episode from CSV.

    Returns DataFrame with datetime index and columns: bg_mM, dose_units, food_g, iob, cob
    """
    csv_path = data_dir / f"{patient_id}_full.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Patient CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")

    # Extract context window ending just before forecast anchor
    ctx_end = anchor - pd.Timedelta(minutes=interval_mins)
    ctx_start = anchor - pd.Timedelta(hours=context_hours)

    ctx = df.loc[ctx_start:ctx_end].copy()

    # Ensure required columns exist
    required_cols = ["bg_mM", "dose_units", "food_g", "iob", "cob"]
    for col in required_cols:
        if col not in ctx.columns:
            ctx[col] = 0.0

    return ctx[required_cols]


def compute_forecast_stability_score(
    actuals: np.ndarray, steps: int
) -> Dict[str, float]:
    """Compute stability metrics for early forecast window.

    Args:
        actuals: (forecast_length,) actual BG values
        steps: Number of steps to analyze from start

    Returns:
        Dict with std, trend (mmol/L per hour), range
    """
    early_window = actuals[:steps]

    # Remove NaNs
    valid = early_window[~np.isnan(early_window)]
    if len(valid) < 2:
        return {"std": np.inf, "trend": np.inf, "range": np.inf}

    std = float(np.std(valid))
    bg_range = float(np.ptp(valid))  # peak-to-peak (max - min)

    # Linear trend: fit line and get slope
    time_hours = np.arange(len(valid)) * INTERVAL_MINS / 60.0
    if len(valid) >= 2:
        trend_per_hour = float(np.polyfit(time_hours, valid, 1)[0])
    else:
        trend_per_hour = 0.0

    return {
        "std": std,
        "trend": abs(trend_per_hour),
        "range": bg_range,
    }


def compute_context_activity_score(
    ctx: pd.DataFrame,
    quiet_hours: float,
    interval_mins: int = INTERVAL_MINS,
) -> Dict[str, float]:
    """Compute meal/bolus activity in late context window.

    Args:
        ctx: Context DataFrame with dose_units and food_g columns
        quiet_hours: Number of hours before forecast to check

    Returns:
        Dict with meal_sum (g), bolus_max (U), bolus_mean (U)
    """
    quiet_steps = int(quiet_hours * 60 / interval_mins)
    quiet_window = ctx.iloc[-quiet_steps:]

    meal_sum = float(quiet_window["food_g"].sum())
    bolus_max = float(quiet_window["dose_units"].max())
    bolus_mean = float(quiet_window["dose_units"].mean())

    return {
        "meal_sum": meal_sum,
        "bolus_max": bolus_max,
        "bolus_mean": bolus_mean,
    }


def find_stable_episodes(
    run_dir: Path,
    data_dir: Path,
    top_n: int = TOP_N_DEFAULT,
) -> List[Dict]:
    """Find episodes meeting stability criteria.

    Returns:
        List of candidate dicts sorted by stability score (best first)
    """
    # Load evaluation data
    actuals, q_forecasts, episodes_df = load_evaluation_data(run_dir)

    # Compute forecast stability for each episode
    stability_steps = int(FORECAST_STABILITY_HOURS * 60 / INTERVAL_MINS)
    print(
        f"\nAnalyzing forecast stability (first {FORECAST_STABILITY_HOURS}h = {stability_steps} steps)..."
    )

    candidates = []

    for idx, row in episodes_df.iterrows():
        patient_id = row["patient_id"]
        anchor = pd.to_datetime(row["anchor"])
        episode_id = f"{patient_id}::{anchor.strftime('%Y-%m-%d')}"

        # Get actual BG for this episode
        ep_actuals = actuals[idx]

        # Compute forecast stability
        stability = compute_forecast_stability_score(ep_actuals, stability_steps)

        # Skip if forecast not stable enough
        if stability["std"] > FORECAST_STD_THRESHOLD:
            continue
        if stability["trend"] > FORECAST_TREND_THRESHOLD:
            continue

        # Skip if BG at forecast start not in target range (5-6 mmol/L)
        bg_at_forecast_start = ep_actuals[0]
        if not (TARGET_BG_MIN <= bg_at_forecast_start <= TARGET_BG_MAX):
            continue

        # Load patient context to check activity
        try:
            ctx = load_patient_context(patient_id, anchor, data_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {episode_id}: {e}")
            continue

        # Compute context activity
        activity = compute_context_activity_score(ctx, CONTEXT_QUIET_HOURS)

        # Skip if too much activity in context
        if activity["meal_sum"] > MEAL_SUM_THRESHOLD:
            continue
        if activity["bolus_max"] > BOLUS_MAX_THRESHOLD:
            continue

        # Compute median forecast RMSE for this episode
        median_idx = q_forecasts.shape[1] // 2  # Assumes median is middle quantile
        median_forecast = q_forecasts[idx, median_idx, :]
        rmse = float(np.sqrt(np.mean((ep_actuals - median_forecast) ** 2)))

        # Composite stability score (lower is better)
        # Penalize high std, high trend, high activity
        score = (
            stability["std"] * 2.0
            + stability["trend"] * 10.0
            + activity["meal_sum"] * 0.1
            + activity["bolus_max"] * 2.0
            + rmse * 0.5
        )

        candidates.append(
            {
                "episode_idx": int(idx),
                "episode_id": episode_id,
                "patient_id": patient_id,
                "anchor": str(anchor),
                "score": score,
                "forecast_std": stability["std"],
                "forecast_trend": stability["trend"],
                "forecast_range": stability["range"],
                "context_meal_sum": activity["meal_sum"],
                "context_bolus_max": activity["bolus_max"],
                "context_bolus_mean": activity["bolus_mean"],
                "baseline_rmse": rmse,
            }
        )

    # Sort by score (lower is better)
    candidates.sort(key=lambda x: x["score"])

    print(f"\nFound {len(candidates)} episodes meeting criteria:")
    print(f"  Forecast std < {FORECAST_STD_THRESHOLD} mmol/L")
    print(f"  Forecast trend < {FORECAST_TREND_THRESHOLD} mmol/L per hour")
    print(f"  Context meal sum < {MEAL_SUM_THRESHOLD} g")
    print(f"  Context bolus max < {BOLUS_MAX_THRESHOLD} U")

    return candidates[:top_n]


def print_candidates(candidates: List[Dict]) -> None:
    """Print candidate episodes in readable format."""
    print("\n" + "=" * 80)
    print("TOP CANDIDATE EPISODES FOR COVARIATE SENSITIVITY TESTING")
    print("=" * 80)

    for i, c in enumerate(candidates, 1):
        print(f"\n[{i}] {c['episode_id']}")
        print(f"    Patient:       {c['patient_id']}")
        print(f"    Anchor:        {c['anchor']}")
        print(f"    Score:         {c['score']:.2f} (lower is better)")
        print("    Forecast:")
        print(f"      - Std:       {c['forecast_std']:.3f} mmol/L")
        print(f"      - Trend:     {c['forecast_trend']:.3f} mmol/L per hour")
        print(f"      - Range:     {c['forecast_range']:.3f} mmol/L")
        print(f"    Context (last {CONTEXT_QUIET_HOURS}h):")
        print(f"      - Meal sum:  {c['context_meal_sum']:.1f} g")
        print(f"      - Bolus max: {c['context_bolus_max']:.3f} U")
        print(f"      - Bolus avg: {c['context_bolus_mean']:.3f} U")
        print(f"    Baseline RMSE: {c['baseline_rmse']:.3f} mmol/L")

    print("\n" + "=" * 80)
    if candidates:
        print(f"Recommended: Use episode_id = '{candidates[0]['episode_id']}'")
        print(
            f"             (patient {candidates[0]['patient_id']}, anchor {candidates[0]['anchor']})"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find stable episodes for covariate sensitivity testing"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Path to TFT evaluation run directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to dataset processed directory",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_DEFAULT,
        help="Number of top candidates to return",
    )

    args = parser.parse_args()

    # Find candidates
    candidates = find_stable_episodes(
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        top_n=args.top_n,
    )

    # Print results
    print_candidates(candidates)

    # Return exit code
    if not candidates:
        print(
            "ERROR: No episodes found meeting criteria. Consider relaxing thresholds."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
