# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Challenge-aligned evaluation metrics (mg/dL) for the Chronos-2 Optuna PoC.

Reimplements the MetaboNet Glucose Prediction Challenge metrics — RMSE, MARD,
and the Diabetes Technology Society (DTS) error grid — so the search can be
scored locally on an internal validation split without depending on the
challenge toolkit. Predictions and targets are in mg/dL.

Forecast steps map to clinical horizons at 5-minute sampling:
    step 6 -> +30 min, step 12 -> +60 min, step 18 -> +90 min, step 24 -> +120 min.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

# Clinical horizons required by the challenge and their forecast-step indices
# (1-based step count at 5-minute sampling).
HORIZON_MINUTES: tuple[int, ...] = (30, 60, 90, 120)
HORIZON_STEPS: Dict[int, int] = {30: 6, 60: 12, 90: 18, 120: 24}


def rmse(pred: Sequence[float], true: Sequence[float]) -> float:
    """Root mean squared error in mg/dL."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mard(pred: Sequence[float], true: Sequence[float]) -> float:
    """Mean absolute relative difference (%), excluding zero references."""
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    valid = t != 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(p[valid] - t[valid]) / t[valid]) * 100.0)


def dts_error_grid(pred: Sequence[float], true: Sequence[float]) -> Dict[str, float]:
    """Percentage of points in each DTS error-grid zone (A-E).

    Zone boundaries follow the Diabetes Technology Society error grid used by
    the MetaboNet challenge. ``pred`` is the monitor/predicted value and
    ``true`` is the reference glucose, both in mg/dL.
    """
    p = np.asarray(pred, dtype=float)
    ref = np.asarray(true, dtype=float)
    total = ref.size
    zones = ("A", "B", "C", "D", "E")
    if total == 0:
        return {f"dts_{z.lower()}_pct": float("nan") for z in zones}

    # Upper boundaries (prediction above reference).
    b_up = np.where(ref <= 50, 60, (540 / 450) * (ref - 50) + 60)
    c_up = np.where(ref <= 50, 86.5, (513.5 / 297) * (ref - 50) + 86.5)
    d_up = np.where(ref <= 50, 124, (476 / 191) * (ref - 50) + 124)
    e_up = np.where(ref <= 50, 179, (421 / 117) * (ref - 50) + 179)

    # Lower boundaries (prediction below reference).
    b_low = np.where(ref <= 62.5, 0, (430 / 537.5) * (ref - 62.5) + 50)
    c_low = np.where(ref <= 97.5, 0, (257 / 502.5) * (ref - 97.5) + 50)
    d_low = np.where(ref <= 153, 0, (147 / 447) * (ref - 153) + 50)
    e_low = np.where(ref <= 238, 0, (76 / 362) * (ref - 238) + 50)

    zone_a = (p <= b_up) & (p >= b_low)
    zone_b = ((p <= c_up) & (p > b_up)) | ((p < b_low) & (p >= c_low))
    zone_c = ((p <= d_up) & (p > c_up)) | ((p < c_low) & (p >= d_low))
    zone_d = ((p <= e_up) & (p > d_up)) | ((p < d_low) & (p >= e_low))
    zone_e = (p > e_up) | (p < e_low)

    masks = {"A": zone_a, "B": zone_b, "C": zone_c, "D": zone_d, "E": zone_e}
    return {
        f"dts_{name.lower()}_pct": float(np.sum(mask) / total * 100.0)
        for name, mask in masks.items()
    }


def evaluate_horizons(
    pred_matrix: np.ndarray,
    true_matrix: np.ndarray,
) -> Dict[str, float]:
    """Compute per-horizon RMSE/MARD/DTS over a batch of forecasts.

    Args:
        pred_matrix: Array of shape (n_windows, 24) — point forecasts (mg/dL).
        true_matrix: Array of shape (n_windows, 24) — ground-truth CGM (mg/dL).

    Returns:
        Flat metric dict keyed like ``rmse_120``, ``mard_60``,
        ``dts_a_pct_120``. Windows with a NaN target at a given horizon are
        excluded from that horizon's aggregation.
    """
    pred_matrix = np.asarray(pred_matrix, dtype=float)
    true_matrix = np.asarray(true_matrix, dtype=float)
    if pred_matrix.ndim != 2 or pred_matrix.shape[1] < max(HORIZON_STEPS.values()):
        raise ValueError(
            f"Expected forecast matrices with >= {max(HORIZON_STEPS.values())} "
            f"columns; got shape {pred_matrix.shape}."
        )

    results: Dict[str, float] = {}
    for minutes, step in HORIZON_STEPS.items():
        col = step - 1  # 1-based clinical step -> 0-based column
        p = pred_matrix[:, col]
        t = true_matrix[:, col]
        finite = np.isfinite(p) & np.isfinite(t)
        p, t = p[finite], t[finite]
        results[f"rmse_{minutes}"] = rmse(p, t)
        results[f"mard_{minutes}"] = mard(p, t)
        results[f"n_{minutes}"] = float(p.size)
        for zone_key, zone_val in dts_error_grid(p, t).items():
            results[f"{zone_key}_{minutes}"] = zone_val

    # Leaderboard-aligned averages across the 4 horizons: the challenge ranks on
    # the mean of MARD and DTS-A over 30/60/90/120, so expose ``*_avg`` metrics
    # (nanmean, so a missing horizon does not wipe out the average).
    avg_bases = (
        "rmse",
        "mard",
        "dts_a_pct",
        "dts_b_pct",
        "dts_c_pct",
        "dts_d_pct",
        "dts_e_pct",
    )
    for base in avg_bases:
        values = [results.get(f"{base}_{m}", np.nan) for m in HORIZON_MINUTES]
        results[f"{base}_avg"] = float(np.nanmean(values))
    return results
