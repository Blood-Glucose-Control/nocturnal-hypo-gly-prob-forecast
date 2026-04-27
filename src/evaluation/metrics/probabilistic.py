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

import warnings
from typing import List, Optional

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
        quantile_forecasts: Shape (n_quantiles, forecast_length). Quantile values
            are sorted per-timestep before CDF interpolation, so mild crossings
            (e.g. from TimesFM) are handled correctly.
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
    # np.interp requires xp to be non-decreasing: sort both arrays together
    # by BG value so crossing quantiles (e.g. from TimesFM) don't silently
    # produce wrong CDF estimates. We clamp to [q_arr[0], q_arr[-1]] to
    # avoid extrapolation to 0/1.
    p_hat = np.empty(forecast_length, dtype=np.float64)
    for t in range(forecast_length):
        x_vals = quantile_forecasts[:, t]
        sort_idx = np.argsort(x_vals)
        p_hat[t] = np.interp(
            threshold,
            x_vals[sort_idx],
            q_arr[sort_idx],
            left=q_arr[0],  # clamp: threshold below all quantiles → P = q_min
            right=q_arr[-1],  # clamp: threshold above all quantiles → P = q_max
        )

    indicator = (actuals < threshold).astype(np.float64)
    return float(np.mean((p_hat - indicator) ** 2))


def _validate_quantile_inputs(
    quantile_forecasts: np.ndarray,
    quantile_levels: List[float],
    actuals: Optional[np.ndarray] = None,
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

    q_batch = quantile_forecasts[np.newaxis, :, :]
    lower = _interp_quantile_level_batch(q_batch, q_arr, target_lower)[0]
    upper = _interp_quantile_level_batch(q_batch, q_arr, target_upper)[0]
    return lower, upper


def _interp_quantile_level_batch(
    quantile_forecasts_batch: np.ndarray,
    q_arr: np.ndarray,
    target: float,
) -> np.ndarray:
    """Interpolate one target quantile level for a full batch.

    Args:
        quantile_forecasts_batch: Shape (n_episodes, n_quantiles, forecast_length).
        q_arr: Sorted quantile levels with shape (n_quantiles,).
        target: Target quantile to interpolate.

    Returns:
        np.ndarray: Interpolated values with shape (n_episodes, forecast_length).
    """
    if target <= q_arr[0]:
        return quantile_forecasts_batch[:, 0, :]
    if target >= q_arr[-1]:
        return quantile_forecasts_batch[:, -1, :]

    hi = int(np.searchsorted(q_arr, target, side="left"))
    if np.isclose(q_arr[hi], target, atol=1e-12, rtol=0.0):
        return quantile_forecasts_batch[:, hi, :]

    lo = hi - 1
    q_lo, q_hi = q_arr[lo], q_arr[hi]
    weight = (target - q_lo) / (q_hi - q_lo)
    lo_vals = quantile_forecasts_batch[:, lo, :]
    hi_vals = quantile_forecasts_batch[:, hi, :]
    return lo_vals + weight * (hi_vals - lo_vals)


def _validate_batch_quantile_inputs(
    quantile_forecasts_batch: np.ndarray,
    quantile_levels: List[float],
    actuals_batch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Shared validation for batch probabilistic metric inputs.

    Checks batch forecast dimensionality, quantile count consistency, and
    quantile-level monotonicity. When ``actuals_batch`` is provided, also
    checks that it is 2D and matches (n_episodes, forecast_length).

    Returns the validated quantile levels as a sorted np.ndarray.
    """
    if quantile_forecasts_batch.ndim != 3:
        raise ValueError(
            "quantile_forecasts_batch must be 3D "
            "(n_episodes, n_quantiles, forecast_length), "
            f"got shape {quantile_forecasts_batch.shape}"
        )

    n_eps, n_q, fh = quantile_forecasts_batch.shape
    if n_q == 0:
        raise ValueError(
            "quantile_forecasts_batch must include at least one quantile "
            "(n_quantiles > 0)."
        )

    if len(quantile_levels) != n_q:
        raise ValueError(
            f"len(quantile_levels)={len(quantile_levels)} does not match "
            f"quantile_forecasts_batch.shape[1]={n_q}"
        )

    if actuals_batch is not None:
        if actuals_batch.ndim != 2:
            raise ValueError(
                "actuals_batch must be 2D (n_episodes, forecast_length), "
                f"got shape {actuals_batch.shape}"
            )
        if actuals_batch.shape[0] != n_eps:
            raise ValueError(
                f"actuals_batch.shape[0]={actuals_batch.shape[0]} does not "
                f"match quantile_forecasts_batch.shape[0]={n_eps}"
            )
        if actuals_batch.shape[1] != fh:
            raise ValueError(
                f"actuals_batch.shape[1]={actuals_batch.shape[1]} does not "
                f"match quantile_forecasts_batch.shape[2]={fh}"
            )

    q_arr = np.array(quantile_levels, dtype=np.float64)
    if not np.all(np.diff(q_arr) > 0):
        raise ValueError("quantile_levels must be strictly increasing.")

    # Check that quantile forecasts are non-decreasing along the quantile axis
    # (axis=1: shape is n_episodes × n_quantiles × forecast_length).
    # Three thresholds:
    #   ≤ 0.01 mmol/L  → sub-clinical floating-point noise; warn and pass through.
    #   0.01–1.0 mmol/L → real but recoverable quantile crossing (e.g. normal TimesFM
    #                      post-training); sort in-place and warn so metrics remain valid.
    #   > 1.0 mmol/L   → catastrophic inversion consistent with a broken-head run
    #                     (e.g. TimesFM high-LR fine-tuning, sharpness_50 ≈ −742 mmol/L);
    #                     raise ValueError to prevent silently polluted metrics.
    diffs = np.diff(quantile_forecasts_batch, axis=1)  # (n_eps, n_q-1, fh)
    violations = diffs[diffs < 0]
    if violations.size > 0:
        max_violation = float(
            -violations.min()
        )  # most negative diff, as positive magnitude
        n_inverted = int((diffs < 0).sum())
        total = diffs.size
        if max_violation > 1.0:
            raise ValueError(
                f"quantile_forecasts_batch has {n_inverted:,} quantile inversions "
                f"({100.0 * n_inverted / total:.2f}% of adjacent-quantile pairs, "
                f"max violation = {max_violation:.4f} mmol/L). "
                "This magnitude indicates a catastrophic calibration failure "
                "(e.g. TimesFM high-LR fine-tuning). "
                "Discard or retrain before computing metrics."
            )
        elif max_violation > 0.01:
            # Moderate inversions (e.g. normal TimesFM quantile crossing): sort to
            # produce a valid monotone CDF, then warn so the caller is aware.
            quantile_forecasts_batch[:] = np.sort(quantile_forecasts_batch, axis=1)
            warnings.warn(
                f"quantile_forecasts_batch had {n_inverted:,} quantile inversions "
                f"({100.0 * n_inverted / total:.2f}% of adjacent-quantile pairs, "
                f"max violation = {max_violation:.4f} mmol/L). "
                "Quantiles have been sorted in-place to produce a valid CDF. "
                "This is expected for models without a monotone quantile head (e.g. TimesFM).",
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"quantile_forecasts_batch has {n_inverted:,} minor quantile inversions "
                f"(max = {max_violation:.2e} mmol/L, likely floating-point noise). "
                "Results should be unaffected.",
                stacklevel=3,
            )

    return q_arr


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
    q_arr = _validate_batch_quantile_inputs(
        quantile_forecasts_batch, quantile_levels, actuals_batch
    )
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    target_lower = (1.0 - level) / 2.0
    target_upper = (1.0 + level) / 2.0
    lower = _interp_quantile_level_batch(quantile_forecasts_batch, q_arr, target_lower)
    upper = _interp_quantile_level_batch(quantile_forecasts_batch, q_arr, target_upper)

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
    q_arr = _validate_batch_quantile_inputs(quantile_forecasts_batch, quantile_levels)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    target_lower = (1.0 - level) / 2.0
    target_upper = (1.0 + level) / 2.0
    lower = _interp_quantile_level_batch(quantile_forecasts_batch, q_arr, target_lower)
    upper = _interp_quantile_level_batch(quantile_forecasts_batch, q_arr, target_upper)

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


def compute_pit_values(
    quantile_forecasts: np.ndarray,
    actuals: np.ndarray,
    quantile_levels: List[float],
) -> np.ndarray:
    """Compute Probability Integral Transform (PIT) values.

    For each (episode, timestep) pair, evaluates the empirical CDF F̂(y_true)
    by linearly interpolating through the discrete quantile forecast.

    Values below all predicted quantile values are assigned PIT = 0.0; values
    above are assigned PIT = 1.0 (hard clamp). Pile-up at the boundaries
    indicates actuals frequently fall outside the predicted quantile range.

    **Note on the expected PIT distribution**: with a full [0, 1] quantile grid
    a perfectly calibrated model yields PIT ~ Uniform[0, 1].  In practice,
    grids are truncated (e.g. 0.1–0.9), so the clamping above creates point
    masses at 0 and 1 even under perfect calibration; the distribution is
    uniform only within (q_min, q_max).  PIT histograms should therefore be
    interpreted relative to the quantile range used, not against a flat
    Uniform[0, 1] baseline.

    Args:
        quantile_forecasts: Shape (n_episodes, n_quantiles, forecast_length).
            Values along the quantile axis must be non-decreasing.
        actuals: Shape (n_episodes, forecast_length).
        quantile_levels: Strictly increasing list of quantile levels, length
            matching n_quantiles.

    Returns:
        np.ndarray: Flat array of PIT values in [0, 1], shape
            (n_episodes * forecast_length,).
    """
    quantile_forecasts = np.asarray(quantile_forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    q_arr = _validate_batch_quantile_inputs(
        quantile_forecasts, quantile_levels, actuals
    )

    n_episodes, n_quantiles, forecast_length = quantile_forecasts.shape

    # Reshape to (N, n_quantiles) where N = n_episodes * forecast_length.
    # transpose(0, 2, 1) → (n_episodes, forecast_length, n_quantiles)
    xp = quantile_forecasts.transpose(0, 2, 1).reshape(-1, n_quantiles)  # (N, n_q)
    y = actuals.ravel()  # (N,)

    # hi: count of quantile values strictly ≤ y for each sample.
    # Equivalent to searchsorted(xp[i], y[i], side='right') per row.
    # Shape: (N,), values in {0, 1, ..., n_quantiles}.
    hi = (xp <= y[:, np.newaxis]).sum(axis=1)

    pit = np.empty(len(y), dtype=np.float64)
    below = hi == 0
    above = hi == n_quantiles
    inside = ~below & ~above

    pit[below] = 0.0
    pit[above] = 1.0

    if inside.any():
        rows = np.where(inside)[0]
        lo_idx = hi[rows] - 1
        hi_idx = hi[rows]
        xp_lo = xp[rows, lo_idx]
        xp_hi = xp[rows, hi_idx]
        q_lo = q_arr[lo_idx]
        q_hi = q_arr[hi_idx]
        denom = xp_hi - xp_lo
        # When xp_lo == xp_hi (flat quantile region), use midpoint of the bin.
        # safe_denom avoids divide-by-zero in both branches of np.where since
        # numpy evaluates both sides even when the condition is False.
        safe_denom = np.where(denom > 0, denom, 1.0)
        weight = np.where(denom > 0, (y[rows] - xp_lo) / safe_denom, 0.5)
        pit[inside] = q_lo + np.clip(weight, 0.0, 1.0) * (q_hi - q_lo)

    return pit


def compute_reliability_curve(
    quantile_forecasts_batch: np.ndarray,
    actuals_batch: np.ndarray,
    quantile_levels: List[float],
) -> tuple:
    """Compute the reliability (quantile calibration) curve.

    For each nominal quantile level q, the empirical exceedance fraction is:

        empirical_q = mean over all (episode, timestep) pairs of
                          I(actual ≤ forecast_q)

    A perfectly calibrated model satisfies empirical_q == q for every level
    (points lie on the diagonal). A curve above the diagonal indicates
    over-forecasting (the model's predicted quantiles are too high); below
    indicates under-forecasting.

    Args:
        quantile_forecasts_batch: Shape (n_episodes, n_quantiles, forecast_length).
        actuals_batch: Shape (n_episodes, forecast_length).
        quantile_levels: Strictly increasing list of quantile levels, length
            matching n_quantiles.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (nominal, empirical) each of shape
        (n_quantiles,). ``nominal`` is simply ``np.array(quantile_levels)``;
        ``empirical[i]`` is the fraction of observed values at or below the
        i-th quantile forecast.
    """
    quantile_forecasts_batch = np.asarray(quantile_forecasts_batch, dtype=np.float64)
    actuals_batch = np.asarray(actuals_batch, dtype=np.float64)
    q_arr = _validate_batch_quantile_inputs(
        quantile_forecasts_batch, quantile_levels, actuals_batch
    )

    # actuals_batch[:, np.newaxis, :] broadcasts to (n_eps, n_q, fh)
    # Compare each actual to its corresponding quantile forecast at each step.
    empirical = np.mean(
        actuals_batch[:, np.newaxis, :] <= quantile_forecasts_batch,
        axis=(0, 2),  # average over episodes and timesteps
    )  # shape: (n_quantiles,)

    return q_arr, empirical


def compute_ece(
    quantile_forecasts_batch: np.ndarray,
    actuals_batch: np.ndarray,
    quantile_levels: List[float],
) -> float:
    """Compute Expected Calibration Error (ECE) for quantile forecasts.

    ECE is the trapezoidal-integration area between the reliability curve and
    the perfect-calibration diagonal, normalised to [0, 1]:

        ECE = ∫₀¹ |empirical(q) − q| dq  ≈  trapz(|empirical − nominal|, nominal)

    This differs from MACE (which uses equal weights 1/n_q) in that it assigns
    weights proportional to the spacing between consecutive quantile levels,
    making it invariant to the chosen quantile grid density.

    Args:
        quantile_forecasts_batch: Shape (n_episodes, n_quantiles, forecast_length).
        actuals_batch: Shape (n_episodes, forecast_length).
        quantile_levels: Strictly increasing list of quantile levels.

    Returns:
        float: ECE in [0, 0.5]. Lower is better; 0 = perfectly calibrated.
    """
    nominal, empirical = compute_reliability_curve(
        quantile_forecasts_batch, actuals_batch, quantile_levels
    )
    return float(np.trapz(np.abs(empirical - nominal), nominal))
