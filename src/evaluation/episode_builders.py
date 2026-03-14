# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Episode builders for different evaluation protocols.

Each evaluation task (nocturnal hypo prediction, post-meal forecasting, etc.)
needs its own episode construction logic — different context anchoring, different
horizons, different covariate requirements. This module collects those builders.

DESIGN RATIONALE:

Sliding-window evaluation (iter_episodes in holdout_eval.py) measures general
forecast accuracy across all times of day. Task-specific evaluation requires
clinically meaningful anchoring:

  - Nocturnal hypo: context ends at midnight, 6h forecast (overnight window)
  - Post-meal: context ends at meal start, forecast covers postprandial period
  - Exercise-onset: context ends before exercise, forecast covers activity
  - Forecast-horizon-step: assessing accuracy of a forecast at each time step in the horizon
        (e.g., 2 hours with 5-min steps = 24 steps -> RMSE per step [0.1, 0.12, ..., 0.5])

These evaluation modes produce DIFFERENT RMSE numbers and must never be mixed
on the same leaderboard. Each builder returns a list of episode dicts that any
model can consume via the predict(data, **kwargs) interface.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.preprocessing.gap_handling import interpolate_small_gaps

# Default sampling interval for CGM data
SAMPLING_INTERVAL_MINUTES = 5

# TODO: Switch function to be "bedtime epsidodes" instead of midnight-anchored, to better capture real-world sleep
# patterns and avoid issues with patients who have irregular sleep schedules or shift work. Bedtime can be defined
# as the last 3 hours of CGM data before a long gap (e.g., >6 hours) in CGM readings, which likely indicates sleep.
# This would allow us to capture nocturnal hypoglycemia more accurately for a wider range of patients, while still
# providing a consistent evaluation framework.


def build_midnight_episodes(
    patient_df: pd.DataFrame,
    context_length: int,
    forecast_length: int,
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
    interval_mins: int = SAMPLING_INTERVAL_MINUTES,
    max_bg_gap_steps: int = 11,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build midnight-anchored episodes for nocturnal forecasting evaluation.

    Extracts context windows ending at midnight with covariate values
    (e.g., IOB, COB) for both context and forecast horizon.

    Each episode contains:
        - anchor: midnight timestamp
        - context_df: DataFrame with context_length rows (BG + covariates)
        - target_bg: numpy array of forecast_length ground truth BG values
        - future_covariates: Dict[covariate_name -> numpy array] for horizon

    Args:
        patient_df: DataFrame with DatetimeIndex for a single patient.
        context_length: Number of context steps (e.g., 512 for ~42.7 hours).
        forecast_length: Number of forecast steps (e.g., 72 for 6 hours).
        target_col: BG column name.
        covariate_cols: Covariate columns (e.g., ["iob"]). If None, covariates
            are not included but episodes are still built (BG-only).
        interval_mins: Sampling interval in minutes.
        max_bg_gap_steps: Maximum consecutive missing BG steps to fill via
            linear interpolation before considering an episode invalid.
            Default is 11 (fills gaps up to 55 min at 5-min sampling).
            Set to 0 to disable interpolation entirely.

    Returns:
        Tuple of (episodes, skip_stats):
            - episodes: List of episode dicts. Empty if no valid windows exist.
            - skip_stats: Dict with keys:
                - total_anchors: Total midnight anchors considered
                - skipped_bg_nan: Count skipped due to missing BG values
                - interpolated_episodes: Count of episodes where gaps were filled
                - skipped_anchors: List of skipped anchor timestamps

    Note:
        Covariate availability is determined at the patient level — a covariate
        is "available" if it has ANY non-NaN values for this patient. Zero values
        (e.g., COB=0 when patient didn't eat) are valid data, not missing data.
        Episodes are only skipped when BG values are missing (required for eval).
    """
    skip_stats = {
        "total_anchors": 0,
        "skipped_bg_nan": 0,
        "interpolated_episodes": 0,
        "skipped_anchors": [],
    }
    df = patient_df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in patient_df. "
            f"Available columns: {list(df.columns)}"
        )

    # Determine available covariates
    available_covs = []
    if covariate_cols:
        available_covs = [
            c for c in covariate_cols if c in df.columns and df[c].notna().any()
        ]

    # Reindex to regular grid. This may introduce NaN rows where data is missing.
    # BG NaN causes episode skip; covariate NaN is preserved (no forward-fill per
    # design rationale — we want to surface data quality issues, not mask them).
    freq = f"{interval_mins}min"
    grid = pd.date_range(
        df.index.min().floor(freq), df.index.max().floor(freq), freq=freq
    )
    df = df.reindex(grid)

    # Compute valid midnight range
    dt = pd.Timedelta(minutes=interval_mins)
    earliest = df.index.min() + context_length * dt
    latest = df.index.max() - (forecast_length - 1) * dt

    first_midnight = (
        earliest.normalize()
    )  #  truncates the timestamp to midnight (00:00:00) of the same day.
    if first_midnight < earliest:
        first_midnight += pd.Timedelta(days=1)

    last_midnight = (
        latest.normalize()
    )  # truncates the timestamp to midnight (00:00:00) of the same day.
    if last_midnight < first_midnight:
        return [], skip_stats

    # Build episodes
    episodes = []
    cols_to_get = [target_col] + (covariate_cols or [])
    cols_to_get = [c for c in cols_to_get if c in df.columns]

    all_anchors = pd.date_range(first_midnight, last_midnight, freq="D")
    skip_stats["total_anchors"] = len(all_anchors)

    for anchor in all_anchors:
        window_start = anchor - context_length * dt
        window_end = anchor + forecast_length * dt
        window_index = pd.date_range(
            window_start, window_end, freq=freq, inclusive="left"
        )

        window_df = df.reindex(window_index)[cols_to_get]

        # Interpolate short BG gaps using the shared gap handling module.
        # Uses all-or-nothing semantics: a gap is only filled if its entire
        # length fits within max_bg_gap_steps. This avoids V-shaped artifacts
        # from partial filling at large gap boundaries.
        if max_bg_gap_steps > 0 and window_df[target_col].isna().any():
            before_nan = window_df[target_col].isna().sum()
            window_df = interpolate_small_gaps(
                window_df, max_gap_rows=max_bg_gap_steps, bg_col=target_col
            )
            after_nan = window_df[target_col].isna().sum()
            if after_nan < before_nan and after_nan == 0:
                skip_stats["interpolated_episodes"] += 1

        # Skip if BG gaps remain after interpolation (gap too long)
        if window_df[target_col].isna().any():
            skip_stats["skipped_bg_nan"] += 1
            skip_stats["skipped_anchors"].append(anchor)
            continue

        context_df = window_df.iloc[:context_length].copy()
        forecast_df = window_df.iloc[context_length:].copy()

        target_bg = forecast_df[target_col].to_numpy()

        # Extract future covariate arrays (values already computed by Hovorka model)
        future_covariates = {}
        for cov_col in available_covs:
            if cov_col in forecast_df.columns:
                future_covariates[cov_col] = forecast_df[cov_col].to_numpy()
            else:
                future_covariates[cov_col] = np.zeros(len(forecast_df))

        episodes.append(
            {
                "anchor": anchor,
                "context_df": context_df,
                "target_bg": target_bg,
                "future_covariates": future_covariates,
            }
        )

    return episodes, skip_stats
