# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Probabilistic forecast evaluation metrics.

Primary metrics for this project (aligned with WQL training objective and
clinical hypoglycemia goal):

  - compute_wql: Weighted Quantile Loss — mean pinball loss across all
    quantile levels and forecast timesteps. Aligned with Chronos-2's 21-quantile
    pinball training objective. Lower is better.

  - compute_brier_score: Brier score for P(BG < threshold) — measures
    calibration of hypoglycemia probability forecasts. Lower is better.
    Default threshold = 3.9 mmol/L (clinical hypoglycemia cutoff).

Usage
-----
    from src.evaluation.metrics.probabilistic import compute_wql, compute_brier_score

    # quantile_forecasts: np.ndarray shape (n_quantiles, forecast_length)
    # actuals:            np.ndarray shape (forecast_length,)
    # quantile_levels:    List[float], e.g. [0.1, 0.2, ..., 0.9]

    wql   = compute_wql(quantile_forecasts, actuals, quantile_levels)
    brier = compute_brier_score(quantile_forecasts, actuals, quantile_levels)
"""

from typing import List

import numpy as np

# Clinical hypoglycemia threshold used throughout this project
HYPO_THRESHOLD_MMOL: float = 3.9


def compute_wql(
    quantile_forecasts: np.ndarray,
    actuals: np.ndarray,
    quantile_levels: List[float],
) -> float:
    """Compute Weighted Quantile Loss (mean pinball loss).

    WQL = mean over q in quantile_levels of
              mean over t in [0, forecast_length) of
                  pinball(q, forecast[q, t], actual[t])

    where pinball(q, f, y) = q * max(y - f, 0) + (1 - q) * max(f - y, 0)

    This is the unnormalized version, expressed in the same units as actuals
    (mmol/L for BG). For cross-dataset comparisons use a normalized variant;
    for within-dataset model comparison this is sufficient.

    Args:
        quantile_forecasts: Shape (n_quantiles, forecast_length). Row i is the
            forecast for quantile quantile_levels[i].
        actuals: Shape (forecast_length,). Ground-truth BG values.
        quantile_levels: List of quantile levels, e.g. [0.1, 0.2, ..., 0.9].
            Must have len(quantile_levels) == quantile_forecasts.shape[0].

    Returns:
        float: Mean pinball loss across all quantiles and timesteps (mmol/L).

    Raises:
        ValueError: If array shapes are inconsistent.
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    quantile_levels = list(quantile_levels)

    if quantile_forecasts.ndim != 2:
        raise ValueError(
            f"quantile_forecasts must be 2D (n_quantiles, forecast_length), "
            f"got shape {quantile_forecasts.shape}"
        )
    if len(quantile_levels) != quantile_forecasts.shape[0]:
        raise ValueError(
            f"len(quantile_levels)={len(quantile_levels)} does not match "
            f"quantile_forecasts.shape[0]={quantile_forecasts.shape[0]}"
        )
    if actuals.shape[0] != quantile_forecasts.shape[1]:
        raise ValueError(
            f"actuals length {actuals.shape[0]} does not match "
            f"forecast_length {quantile_forecasts.shape[1]}"
        )

    # q_col: (n_quantiles, 1) for broadcasting against (n_quantiles, forecast_length)
    q_col = np.array(quantile_levels, dtype=np.float64).reshape(-1, 1)

    errors = (
        actuals[np.newaxis, :] - quantile_forecasts
    )  # (n_quantiles, forecast_length)
    pinball = np.where(errors >= 0, q_col * errors, (q_col - 1.0) * errors)

    return float(np.mean(pinball))


def compute_brier_score(
    quantile_forecasts: np.ndarray,
    actuals: np.ndarray,
    quantile_levels: List[float],
    threshold: float = HYPO_THRESHOLD_MMOL,
) -> float:
    """Compute Brier score for P(BG < threshold) over the forecast horizon.

    At each timestep t:
      1. Estimate P(BG_t < threshold) by treating the quantile forecast as a
         discrete CDF and linearly interpolating at the threshold value.
      2. Compute (P_hat - indicator)^2 where indicator = 1 if actual_t < threshold.
    The Brier score is the mean of this quantity over all timesteps.

    Linear extrapolation beyond the outermost quantile values is clamped:
      - Below min quantile value: P = quantile_levels[0]  (e.g. 0.1)
      - Above max quantile value: P = quantile_levels[-1] (e.g. 0.9)
    This is conservative — never claims P=0 or P=1 from a finite quantile set.

    Args:
        quantile_forecasts: Shape (n_quantiles, forecast_length). Must be sorted
            in ascending order along the quantile axis (quantile_levels[0] < ... < [-1]).
        actuals: Shape (forecast_length,). Ground-truth BG values.
        quantile_levels: List of quantile levels in ascending order.
        threshold: BG threshold for the binary event (mmol/L). Default 3.9.

    Returns:
        float: Mean Brier score across all forecast timesteps.

    Raises:
        ValueError: If array shapes are inconsistent or quantile_levels unsorted.
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    quantile_levels = list(quantile_levels)

    if quantile_forecasts.ndim != 2:
        raise ValueError(
            f"quantile_forecasts must be 2D (n_quantiles, forecast_length), "
            f"got shape {quantile_forecasts.shape}"
        )
    if len(quantile_levels) != quantile_forecasts.shape[0]:
        raise ValueError(
            f"len(quantile_levels)={len(quantile_levels)} does not match "
            f"quantile_forecasts.shape[0]={quantile_forecasts.shape[0]}"
        )
    if actuals.shape[0] != quantile_forecasts.shape[1]:
        raise ValueError(
            f"actuals length {actuals.shape[0]} does not match "
            f"forecast_length {quantile_forecasts.shape[1]}"
        )
    q_arr = np.array(quantile_levels, dtype=np.float64)
    if not np.all(np.diff(q_arr) > 0):
        raise ValueError("quantile_levels must be strictly increasing.")

    forecast_length = quantile_forecasts.shape[1]

    # For each timestep, interpolate P(BG < threshold) from the quantile CDF.
    # quantile_forecasts[:, t] are the BG values (x-axis of CDF);
    # quantile_levels are the corresponding probabilities (y-axis).
    # np.interp(threshold, xp, fp) requires xp to be increasing — valid since
    # quantile forecasts are non-decreasing by construction. We clamp to
    # [q_arr[0], q_arr[-1]] to avoid extrapolation to 0/1.
    p_hat = np.empty(forecast_length, dtype=np.float64)
    for t in range(forecast_length):
        x_vals = quantile_forecasts[:, t]
        p_hat[t] = np.interp(
            threshold,
            x_vals,
            q_arr,
            left=q_arr[0],  # clamp: threshold below all quantiles → P = q_min
            right=q_arr[-1],  # clamp: threshold above all quantiles → P = q_max
        )

    indicator = (actuals < threshold).astype(np.float64)
    return float(np.mean((p_hat - indicator) ** 2))


def _find_symmetric_quantile_pair(quantile_levels: List[float], level: float) -> tuple:
    """Find the quantile indices bracketing a symmetric prediction interval.

    For a given coverage level (e.g., 0.8), the symmetric interval uses
    quantiles at (1-level)/2 and (1+level)/2 — i.e., 0.10 and 0.90.
    Returns the indices of the closest available quantiles.

    Args:
        quantile_levels: Sorted list of quantile levels.
        level: Nominal coverage level (e.g., 0.5 or 0.8).

    Returns:
        Tuple (lower_idx, upper_idx) into quantile_levels.

    Raises:
        ValueError: If no suitable pair can be found.
    """
    q_arr = np.array(quantile_levels, dtype=np.float64)
    target_lower = (1.0 - level) / 2.0
    target_upper = (1.0 + level) / 2.0
    lower_idx = int(np.argmin(np.abs(q_arr - target_lower)))
    upper_idx = int(np.argmin(np.abs(q_arr - target_upper)))
    if lower_idx == upper_idx:
        raise ValueError(
            f"Cannot find distinct quantile pair for level={level} "
            f"in quantile_levels={quantile_levels}"
        )
    return lower_idx, upper_idx


def compute_coverage(
    quantile_forecasts: np.ndarray,
    actuals: np.ndarray,
    quantile_levels: List[float],
    level: float = 0.5,
) -> float:
    """Compute prediction interval coverage at a given nominal level.

    For level=0.8, checks what fraction of actual values fall within the
    [q_0.10, q_0.90] prediction interval. Perfect calibration → coverage ≈ level.

    Args:
        quantile_forecasts: Shape (n_quantiles, forecast_length).
        actuals: Shape (forecast_length,).
        quantile_levels: List of quantile levels in ascending order.
        level: Nominal coverage level (e.g., 0.5 or 0.8).

    Returns:
        float: Empirical coverage fraction in [0, 1].
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    lower_idx, upper_idx = _find_symmetric_quantile_pair(quantile_levels, level)
    lower = quantile_forecasts[lower_idx]
    upper = quantile_forecasts[upper_idx]

    covered = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(covered))


def compute_sharpness(
    quantile_forecasts: np.ndarray,
    quantile_levels: List[float],
    level: float = 0.5,
) -> float:
    """Compute mean prediction interval width (sharpness) at a given level.

    Sharpness measures the average width of the prediction interval. Lower is
    better (tighter intervals), conditional on adequate coverage.

    Args:
        quantile_forecasts: Shape (n_quantiles, forecast_length).
        quantile_levels: List of quantile levels in ascending order.
        level: Nominal coverage level (e.g., 0.5 or 0.8).

    Returns:
        float: Mean interval width in the same units as the forecasts (mmol/L).
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)

    lower_idx, upper_idx = _find_symmetric_quantile_pair(quantile_levels, level)
    lower = quantile_forecasts[lower_idx]
    upper = quantile_forecasts[upper_idx]

    return float(np.mean(upper - lower))


def compute_mace(
    quantile_forecasts: np.ndarray,
    actuals: np.ndarray,
    quantile_levels: List[float],
) -> float:
    """Compute Mean Absolute Calibration Error (MACE).

    For each quantile level q, computes the empirical coverage
    c_q = mean(actual <= forecast_q), then returns mean(|c_q - q|) over all
    quantile levels. A perfectly calibrated model has MACE = 0.

    Args:
        quantile_forecasts: Shape (n_quantiles, forecast_length).
        actuals: Shape (forecast_length,).
        quantile_levels: List of quantile levels in ascending order.

    Returns:
        float: MACE in [0, 1]. Lower is better.
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    q_arr = np.array(quantile_levels, dtype=np.float64)

    # Empirical coverage for each quantile: fraction of timesteps where
    # actual <= predicted quantile value.
    # Shape: (n_quantiles,)
    empirical = np.mean(actuals[np.newaxis, :] <= quantile_forecasts, axis=1)

    return float(np.mean(np.abs(empirical - q_arr)))
