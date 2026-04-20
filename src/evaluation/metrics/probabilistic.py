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


def _validate_quantile_inputs(
    quantile_forecasts: np.ndarray,
    quantile_levels: List[float],
    actuals: np.ndarray = None,
) -> np.ndarray:
    """Shared validation for probabilistic metric inputs.

    Checks quantile_forecasts shape, matching length with quantile_levels,
    and monotonicity of quantile_levels. When ``actuals`` is provided, also
    checks it matches the forecast horizon.

    Returns the validated quantile_levels as a sorted np.ndarray.
    """
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
    if actuals is not None and actuals.shape[0] != quantile_forecasts.shape[1]:
        raise ValueError(
            f"actuals length {actuals.shape[0]} does not match "
            f"forecast_length {quantile_forecasts.shape[1]}"
        )
    q_arr = np.array(quantile_levels, dtype=np.float64)
    if not np.all(np.diff(q_arr) > 0):
        raise ValueError("quantile_levels must be strictly increasing.")
    return q_arr


def _interval_bounds(
    quantile_forecasts: np.ndarray,
    q_arr: np.ndarray,
    level: float,
) -> tuple:
    """Return (lower, upper) prediction interval bounds at the given level.

    When the target quantiles (e.g., 0.25 and 0.75 for level=0.5) are not
    present in ``q_arr``, the bounds are linearly interpolated from the
    available quantile forecasts at each timestep. Values outside the
    available range are clamped to the outermost quantile values.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    target_lower = (1.0 - level) / 2.0
    target_upper = (1.0 + level) / 2.0

    forecast_length = quantile_forecasts.shape[1]
    lower = np.empty(forecast_length, dtype=np.float64)
    upper = np.empty(forecast_length, dtype=np.float64)
    for t in range(forecast_length):
        lower[t] = np.interp(target_lower, q_arr, quantile_forecasts[:, t])
        upper[t] = np.interp(target_upper, q_arr, quantile_forecasts[:, t])
    return lower, upper


def compute_coverage(
    quantile_forecasts: np.ndarray,
    actuals: np.ndarray,
    quantile_levels: List[float],
    level: float = 0.5,
) -> float:
    """Compute prediction interval coverage at a given nominal level.

    For level=0.8, measures the fraction of actual values that fall within
    the [q_0.10, q_0.90] prediction interval. Perfect calibration yields
    coverage ≈ level. When the target quantiles are not present, the interval
    bounds are linearly interpolated from the available quantile forecasts.

    Args:
        quantile_forecasts: Shape (n_quantiles, forecast_length).
        actuals: Shape (forecast_length,). Ground-truth values.
        quantile_levels: List of quantile levels in strictly increasing order.
        level: Nominal coverage level in (0, 1). Default 0.5.

    Returns:
        float: Empirical coverage fraction in [0, 1].
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    q_arr = _validate_quantile_inputs(quantile_forecasts, quantile_levels, actuals)

    lower, upper = _interval_bounds(quantile_forecasts, q_arr, level)
    covered = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(covered))


def compute_sharpness(
    quantile_forecasts: np.ndarray,
    quantile_levels: List[float],
    level: float = 0.5,
) -> float:
    """Compute mean prediction interval width (sharpness) at a given level.

    Sharpness measures the average width of the prediction interval. Lower is
    better (tighter intervals), conditional on adequate coverage. When the
    target quantiles are not present, bounds are linearly interpolated.

    Args:
        quantile_forecasts: Shape (n_quantiles, forecast_length).
        quantile_levels: List of quantile levels in strictly increasing order.
        level: Nominal coverage level in (0, 1). Default 0.5.

    Returns:
        float: Mean interval width in the same units as the forecasts (mmol/L).
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    q_arr = _validate_quantile_inputs(quantile_forecasts, quantile_levels)

    lower, upper = _interval_bounds(quantile_forecasts, q_arr, level)
    return float(np.mean(upper - lower))


def compute_coverage_by_step(
    quantile_forecasts_batch: np.ndarray,
    actuals_batch: np.ndarray,
    quantile_levels: List[float],
    level: float = 0.9,
) -> np.ndarray:
    """Compute empirical coverage at each forecast horizon step across episodes.

    Args:
        quantile_forecasts_batch: Shape (n_episodes, n_quantiles, forecast_length).
        actuals_batch: Shape (n_episodes, forecast_length).
        quantile_levels: List of quantile levels in strictly increasing order.
        level: Nominal coverage level in (0, 1).

    Returns:
        np.ndarray: Shape (forecast_length,). Each entry is the fraction of
        episodes where the actual value at that step fell within the
        prediction interval.
    """
    quantile_forecasts_batch = np.asarray(quantile_forecasts_batch, dtype=np.float64)
    actuals_batch = np.asarray(actuals_batch, dtype=np.float64)
    q_arr = np.array(quantile_levels, dtype=np.float64)
    if not np.all(np.diff(q_arr) > 0):
        raise ValueError("quantile_levels must be strictly increasing.")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    target_lower = (1.0 - level) / 2.0
    target_upper = (1.0 + level) / 2.0
    n_eps, _, fh = quantile_forecasts_batch.shape

    lower = np.empty((n_eps, fh), dtype=np.float64)
    upper = np.empty((n_eps, fh), dtype=np.float64)
    for e in range(n_eps):
        for t in range(fh):
            lower[e, t] = np.interp(target_lower, q_arr, quantile_forecasts_batch[e, :, t])
            upper[e, t] = np.interp(target_upper, q_arr, quantile_forecasts_batch[e, :, t])

    covered = (actuals_batch >= lower) & (actuals_batch <= upper)
    return np.mean(covered, axis=0)  # (fh,)


def compute_sharpness_by_step(
    quantile_forecasts_batch: np.ndarray,
    quantile_levels: List[float],
    level: float = 0.9,
) -> np.ndarray:
    """Compute mean prediction interval width at each forecast horizon step.

    Args:
        quantile_forecasts_batch: Shape (n_episodes, n_quantiles, forecast_length).
        quantile_levels: List of quantile levels in strictly increasing order.
        level: Nominal coverage level in (0, 1).

    Returns:
        np.ndarray: Shape (forecast_length,). Each entry is the mean interval
        width (mmol/L) across episodes at that forecast step.
    """
    quantile_forecasts_batch = np.asarray(quantile_forecasts_batch, dtype=np.float64)
    q_arr = np.array(quantile_levels, dtype=np.float64)
    if not np.all(np.diff(q_arr) > 0):
        raise ValueError("quantile_levels must be strictly increasing.")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    target_lower = (1.0 - level) / 2.0
    target_upper = (1.0 + level) / 2.0
    n_eps, _, fh = quantile_forecasts_batch.shape

    lower = np.empty((n_eps, fh), dtype=np.float64)
    upper = np.empty((n_eps, fh), dtype=np.float64)
    for e in range(n_eps):
        for t in range(fh):
            lower[e, t] = np.interp(target_lower, q_arr, quantile_forecasts_batch[e, :, t])
            upper[e, t] = np.interp(target_upper, q_arr, quantile_forecasts_batch[e, :, t])

    return np.mean(upper - lower, axis=0)  # (fh,)


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
        quantile_levels: List of quantile levels in strictly increasing order.

    Returns:
        float: MACE in [0, 1]. Lower is better.
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    q_arr = _validate_quantile_inputs(quantile_forecasts, quantile_levels, actuals)

    # Empirical coverage for each quantile: fraction of timesteps where
    # actual <= predicted quantile value.  Shape: (n_quantiles,)
    empirical = np.mean(actuals[np.newaxis, :] <= quantile_forecasts, axis=1)
    return float(np.mean(np.abs(empirical - q_arr)))
