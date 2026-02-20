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

These evaluation modes produce DIFFERENT RMSE numbers and must never be mixed
on the same leaderboard. Each builder returns a list of episode dicts that any
model can consume via the predict(data, **kwargs) interface.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default sampling interval for CGM data
SAMPLING_INTERVAL_MINUTES = 5


def build_midnight_episodes(
    patient_df: pd.DataFrame,
    context_length: int,
    forecast_length: int,
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
    interval_mins: int = SAMPLING_INTERVAL_MINUTES,
) -> List[Dict[str, Any]]:
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

    Returns:
        List of episode dicts. Empty if no valid midnight windows exist.
    """
    df = patient_df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Determine available covariates
    available_covs = []
    if covariate_cols:
        available_covs = [
            c for c in covariate_cols if c in df.columns and df[c].notna().any()
        ]

    # Reindex to regular grid
    freq = f"{interval_mins}min"
    grid = pd.date_range(
        df.index.min().floor(freq), df.index.max().floor(freq), freq=freq
    )
    df = df.reindex(grid)

    # Compute valid midnight range
    dt = pd.Timedelta(minutes=interval_mins)
    earliest = df.index.min() + context_length * dt
    latest = df.index.max() - (forecast_length - 1) * dt

    first_midnight = earliest.normalize()
    if first_midnight < earliest:
        first_midnight += pd.Timedelta(days=1)

    last_midnight = latest.normalize()
    if last_midnight < first_midnight:
        return []

    # Build episodes
    episodes = []
    cols_to_get = [target_col] + (covariate_cols or [])
    cols_to_get = [c for c in cols_to_get if c in df.columns]

    for anchor in pd.date_range(first_midnight, last_midnight, freq="D"):
        window_start = anchor - context_length * dt
        window_end = anchor + forecast_length * dt
        window_index = pd.date_range(
            window_start, window_end, freq=freq, inclusive="left"
        )

        window_df = df.reindex(window_index)[cols_to_get]

        # Skip if any BG values are missing
        if window_df[target_col].isna().any():
            continue

        context_df = window_df.iloc[:context_length].copy()
        forecast_df = window_df.iloc[context_length:].copy()

        # If covariates requested, check coverage (>50% in context).
        # Covariates with too many NaNs provide unreliable signals even after
        # forward-fill. Most valid CGM windows will have near-complete
        # covariate data from the Hovorka model.
        if available_covs:
            cov_coverage = {c: 1 - context_df[c].isna().mean() for c in available_covs}
            if all(v < 0.5 for v in cov_coverage.values()):
                continue

        target_bg = forecast_df[target_col].to_numpy()

        # Build future covariate arrays (fill NaN via forward-fill)
        future_covariates = {}
        for cov_col in available_covs:
            if cov_col in context_df.columns:
                context_df[cov_col] = context_df[cov_col].ffill().fillna(0)
            if cov_col in forecast_df.columns:
                future_vals = (
                    pd.Series(forecast_df[cov_col].to_numpy())
                    .ffill()
                    .fillna(0)
                    .to_numpy()
                )
            else:
                future_vals = np.zeros(len(forecast_df))
            future_covariates[cov_col] = future_vals

        episodes.append(
            {
                "anchor": anchor,
                "context_df": context_df,
                "target_bg": target_bg,
                "future_covariates": future_covariates,
            }
        )

    return episodes
