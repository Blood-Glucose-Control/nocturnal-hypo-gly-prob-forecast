# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""Chronos-2-specific episode construction and covariate-eval helpers."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "build_midnight_episodes",
    "format_for_autogluon_with_known_covariates",
    "evaluate_with_covariates",
]


def build_midnight_episodes(
    patient_df: pd.DataFrame,
    target_col: str,
    covariate_cols: Optional[List[str]] = None,
    interval_mins: int = 5,
    context_len: int = 512,
    horizon: int = 72,
) -> List[Dict[str, Any]]:
    """Build midnight-anchored episodes with covariates for evaluation.

    Extracts context windows ending at midnight with covariate values
    (e.g., IOB, COB) for both context and forecast horizon. Covariates
    are sourced from preprocessing (Hovorka insulin model, carb model).

    Each episode contains:
        - anchor: midnight timestamp
        - context_df: Historical data with BG and covariates
        - target_bg: Ground truth BG for forecast horizon
        - future_covariates: Dict[covariate_name -> np.ndarray] for horizon
        - covariates_at_midnight: Dict[covariate_name -> float]

    Args:
        patient_df: DataFrame with DatetimeIndex for a single patient.
        target_col: BG column name.
        covariate_cols: Covariate column names (e.g., ["iob"] or ["iob", "cob"]).
            Defaults to ["iob"]. At least one must be present in the data.
        interval_mins: Sampling interval in minutes.
        context_len: Number of context steps.
        horizon: Number of forecast steps.

    Returns:
        List of episode dicts. Empty if no valid episodes or no covariates.
    """
    if covariate_cols is None:
        covariate_cols = ["iob"]

    df = patient_df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Check which covariates are actually present
    available_covs = [
        c for c in covariate_cols if c in df.columns and df[c].notna().any()
    ]
    if not available_covs:
        logger.warning("No covariate data available (need one of %s)", covariate_cols)
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
    cols_to_get = [target_col] + covariate_cols
    # Only select columns that exist in df
    cols_to_get = [c for c in cols_to_get if c in df.columns]

    for anchor in pd.date_range(first_midnight, last_midnight, freq="D"):
        window_start = anchor - context_len * dt
        window_end = anchor + horizon * dt
        window_index = pd.date_range(
            window_start, window_end, freq=freq, inclusive="left"
        )

        window_df = df.reindex(window_index)[cols_to_get]

        # Skip if any BG values are missing
        if window_df[target_col].isna().any():
            continue

        context_df = window_df.iloc[:context_len].copy()
        forecast_df = window_df.iloc[context_len:].copy()

        # Skip if all available covariates have <50% coverage in context.
        # Covariates with too many NaNs provide unreliable signals even after
        # forward-fill. 50% is a reasonable minimum — most valid CGM windows
        # will have near-complete covariate data from the Hovorka model.
        cov_coverage = {c: 1 - context_df[c].isna().mean() for c in available_covs}
        if all(v < 0.5 for v in cov_coverage.values()):
            continue

        target_bg = forecast_df[target_col].to_numpy()

        # Build future covariate arrays and fill NaN
        future_covariates = {}
        covariates_at_midnight = {}
        for cov_col in covariate_cols:
            if cov_col in context_df.columns:
                context_df[cov_col] = context_df[cov_col].ffill().fillna(0)
                covariates_at_midnight[cov_col] = context_df[cov_col].iloc[-1]
            else:
                context_df[cov_col] = 0.0
                covariates_at_midnight[cov_col] = 0.0

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
                "covariates_at_midnight": covariates_at_midnight,
            }
        )

    return episodes


def format_for_autogluon_with_known_covariates(
    episodes: List[Dict[str, Any]],
    target_col: str,
    covariate_cols: Optional[List[str]] = None,
    forecast_horizon: int = 72,
    interval_mins: int = 5,
) -> Tuple[Any, Any]:
    """Convert episodes to AutoGluon format with known future covariates.

    Args:
        episodes: List of episode dicts from build_midnight_episodes().
        target_col: BG column name in episode context_df.
        covariate_cols: Covariate column names (e.g., ["iob"] or ["iob", "cob"]).
        forecast_horizon: Number of forecast steps.
        interval_mins: Sampling interval in minutes.

    Returns:
        Tuple of (test_data, known_covariates) as TimeSeriesDataFrames.
    """
    from autogluon.timeseries import TimeSeriesDataFrame

    if covariate_cols is None:
        covariate_cols = ["iob"]

    context_data_list = []
    known_cov_list = []

    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"

        # Context data: BG + covariates (history up to midnight)
        df = ep["context_df"].copy()
        df["item_id"] = item_id
        df["timestamp"] = df.index
        df["target"] = df[target_col]
        for cov_col in covariate_cols:
            df[cov_col] = df[cov_col] if cov_col in df.columns else 0.0
        out_cols = ["item_id", "timestamp", "target"] + covariate_cols
        context_data_list.append(df[out_cols])

        # Known covariates: future trajectories
        future_timestamps = pd.date_range(
            ep["anchor"],
            periods=forecast_horizon,
            freq=f"{interval_mins}min",
        )
        future_dict = {"item_id": item_id, "timestamp": future_timestamps}
        for cov_col in covariate_cols:
            future_vals = ep["future_covariates"].get(
                cov_col, np.zeros(forecast_horizon)
            )
            future_dict[cov_col] = future_vals[:forecast_horizon]
        known_cov_list.append(pd.DataFrame(future_dict))

    context_combined = pd.concat(context_data_list, ignore_index=True)
    context_combined = context_combined.set_index(["item_id", "timestamp"])
    context_data = TimeSeriesDataFrame(context_combined)

    known_combined = pd.concat(known_cov_list, ignore_index=True)
    known_combined = known_combined.set_index(["item_id", "timestamp"])
    known_covariates = TimeSeriesDataFrame(known_combined)

    return context_data, known_covariates


def evaluate_with_covariates(
    predictor: Any,
    test_data: Any,
    known_covariates: Any,
    episodes: List[Dict[str, Any]],
) -> Tuple[float, Any, List[Dict[str, Any]]]:
    """Evaluate model with known covariates on midnight-anchored episodes.

    Args:
        predictor: AutoGluon TimeSeriesPredictor instance.
        test_data: TimeSeriesDataFrame with context data.
        known_covariates: TimeSeriesDataFrame with future covariates.
        episodes: Original episode dicts (for ground truth).

    Returns:
        Tuple of (average_rmse, predictions_dataframe, per_episode_results).
        per_episode_results is a list of dicts with "pred" (numpy array)
        and "rmse" (float) for each episode, aligned with the episodes list.
    """
    logger.info("Running predictions with known covariates...")

    predictions = predictor.predict(test_data, known_covariates=known_covariates)

    rmse_list = []
    per_episode = []
    for i, ep in enumerate(episodes):
        item_id = f"ep_{i:03d}"
        if item_id not in predictions.index.get_level_values(0):
            per_episode.append({"pred": np.array([]), "rmse": float("nan")})
            continue

        pred = predictions.loc[item_id]["mean"].values
        # Slice ground truth to match prediction length in case AutoGluon
        # returns fewer steps than forecast_horizon (rare edge case)
        actual = ep["target_bg"][: len(pred)]
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        rmse_list.append(rmse)
        per_episode.append({"pred": pred, "rmse": rmse})

    avg_rmse = float(np.mean(rmse_list)) if rmse_list else float("nan")
    logger.info("Evaluation: %.4f avg RMSE over %d episodes", avg_rmse, len(rmse_list))
    return avg_rmse, predictions, per_episode
