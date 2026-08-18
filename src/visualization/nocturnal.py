# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Shared visualization helpers for nocturnal forecasting post-run scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

SAMPLING_INTERVAL_MINUTES = 5
DEFAULT_BOXPLOT_QUANTILES = (10.0, 25.0, 50.0, 75.0, 90.0)
DEFAULT_IQR_QUANTILES = (25.0, 75.0)


def resolve_forecast_results_path(path: str | Path) -> Path:
    """Resolve *path* to a concrete results file.

    If *path* is a run directory, prefer ``forecasts.npz`` and fall back to
    ``nocturnal_results.json``.
    """
    candidate = Path(path)
    if candidate.is_dir():
        npz_path = candidate / "forecasts.npz"
        json_path = candidate / "nocturnal_results.json"
        if npz_path.exists():
            return npz_path
        if json_path.exists():
            return json_path
        raise FileNotFoundError(
            f"No results found in {candidate!r}: expected forecasts.npz or "
            "nocturnal_results.json"
        )
    return candidate


def load_prediction_actual_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(predictions, actuals)`` arrays from NPZ or legacy JSON output."""
    resolved_path = resolve_forecast_results_path(path)

    if resolved_path.suffix == ".npz":
        with np.load(resolved_path, allow_pickle=False) as data:
            predictions = np.asarray(data["predictions"], dtype=np.float64)
            actuals = np.asarray(data["actuals"], dtype=np.float64)
    elif resolved_path.suffix == ".json":
        with resolved_path.open() as file_obj:
            payload = json.load(file_obj)
        predictions, actuals = _load_legacy_episode_arrays(payload, resolved_path)
    else:
        raise ValueError(
            f"Unsupported results file format: {resolved_path}. "
            "Expected .npz or .json output."
        )

    if predictions.ndim != 2 or actuals.ndim != 2:
        raise ValueError(
            f"{resolved_path} must contain 2D arrays; got "
            f"predictions shape={predictions.shape}, actuals shape={actuals.shape}"
        )
    if predictions.shape != actuals.shape:
        raise ValueError(
            f"{resolved_path} shape mismatch: predictions shape={predictions.shape} "
            f"!= actuals shape={actuals.shape}"
        )
    return predictions, actuals


def compute_horizon_rmse_quantiles(
    predictions: np.ndarray,
    actuals: np.ndarray,
    sampling_interval_minutes: int = SAMPLING_INTERVAL_MINUTES,
    quantiles: Sequence[float] = DEFAULT_BOXPLOT_QUANTILES,
) -> list[dict[str, float]]:
    """Return per-horizon RMSE summaries and quantiles."""
    quantile_levels = _validate_boxplot_quantiles(quantiles)
    sq_err = (predictions - actuals) ** 2
    horizon_data: list[dict[str, float]] = []
    for idx in range(predictions.shape[1]):
        ep_sq_err = sq_err[:, idx]
        quantile_values = np.sqrt(np.percentile(ep_sq_err, quantile_levels))
        row: dict[str, float] = {
            "horizon_minutes": float((idx + 1) * sampling_interval_minutes),
            "rmse": float(np.sqrt(np.mean(ep_sq_err))),
            "whisker_low": float(quantile_values[0]),
            "box_low": float(quantile_values[1]),
            "median": float(quantile_values[2]),
            "box_high": float(quantile_values[3]),
            "whisker_high": float(quantile_values[4]),
        }
        for level, value in zip(quantile_levels, quantile_values):
            row[_quantile_key(level)] = float(value)
        horizon_data.append(row)
    return horizon_data


def compute_horizon_rmse_stats(
    predictions: np.ndarray,
    actuals: np.ndarray,
    sampling_interval_minutes: int = SAMPLING_INTERVAL_MINUTES,
    iqr_quantiles: Sequence[float] = DEFAULT_IQR_QUANTILES,
) -> dict[str, np.ndarray]:
    """Compute per-horizon mean RMSE, IQR, and cumulative RMSE."""
    iqr_low, iqr_high = _validate_iqr_quantiles(iqr_quantiles)
    n_steps = predictions.shape[1]
    hours = np.array(
        [(i + 1) * sampling_interval_minutes / 60.0 for i in range(n_steps)]
    )
    sq_err = (predictions - actuals) ** 2
    mean_rmse = np.sqrt(np.mean(sq_err, axis=0))
    ep_rmse = np.sqrt(sq_err)
    band_low = np.percentile(ep_rmse, iqr_low, axis=0)
    band_high = np.percentile(ep_rmse, iqr_high, axis=0)
    cum_mse = np.cumsum(np.mean(sq_err, axis=0)) / np.arange(1, n_steps + 1)
    cum_rmse = np.sqrt(cum_mse)
    stats: dict[str, np.ndarray] = {
        "hours": hours,
        "mean": mean_rmse,
        "band_low": band_low,
        "band_high": band_high,
        "cumulative": cum_rmse,
    }
    stats[_quantile_key(iqr_low)] = band_low
    stats[_quantile_key(iqr_high)] = band_high
    return stats


def _load_legacy_episode_arrays(
    payload: dict,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = payload.get("per_episode")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{source_path} missing non-empty 'per_episode' list")

    first_pred = episodes[0].get("pred")
    if not isinstance(first_pred, list) or not first_pred:
        raise ValueError(f"{source_path} has invalid first per_episode.pred payload")
    forecast_length = len(first_pred)

    pred_rows: list[list[float]] = []
    actual_rows: list[list[float]] = []
    for episode in episodes:
        pred = episode.get("pred")
        target = episode.get("target_bg")
        if (
            isinstance(pred, list)
            and isinstance(target, list)
            and len(pred) == forecast_length
            and len(target) == forecast_length
        ):
            pred_rows.append(pred)
            actual_rows.append(target)

    if not pred_rows:
        raise ValueError(
            f"{source_path} did not contain any per_episode rows with matching "
            "pred/target_bg lengths"
        )
    return (
        np.asarray(pred_rows, dtype=np.float64),
        np.asarray(actual_rows, dtype=np.float64),
    )


def _validate_boxplot_quantiles(
    quantiles: Sequence[float],
) -> tuple[float, float, float, float, float]:
    levels = tuple(float(level) for level in quantiles)
    if len(levels) != 5:
        raise ValueError(
            "Expected exactly 5 boxplot quantiles in ascending order "
            "(whisker_low, box_low, median, box_high, whisker_high)."
        )
    if any(level < 0.0 or level > 100.0 for level in levels):
        raise ValueError(f"Quantiles must be between 0 and 100; got {levels}.")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError(f"Quantiles must be strictly increasing; got {levels}.")
    q0, q1, q2, q3, q4 = levels
    return q0, q1, q2, q3, q4


def _validate_iqr_quantiles(quantiles: Sequence[float]) -> tuple[float, float]:
    levels = tuple(float(level) for level in quantiles)
    if len(levels) != 2:
        raise ValueError("Expected exactly two IQR quantiles: (lower, upper).")
    low, high = levels
    if low < 0.0 or high > 100.0:
        raise ValueError(f"IQR quantiles must be between 0 and 100; got {levels}.")
    if low >= high:
        raise ValueError(f"IQR quantiles must satisfy lower < upper; got {levels}.")
    return low, high


def _quantile_key(level: float) -> str:
    if float(level).is_integer():
        return f"q{int(level)}"
    level_str = str(level).rstrip("0").rstrip(".").replace(".", "_")
    return f"q{level_str}"


def resolve_forecast_npz_path(path: str | Path) -> Path:
    """Resolve *path* to ``forecasts.npz``.

    Accepts either:
    - a run directory containing ``forecasts.npz``
    - a direct path to ``forecasts.npz``
    """
    candidate = Path(path)
    if candidate.is_dir():
        npz_path = candidate / "forecasts.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{candidate!r} does not contain forecasts.npz for probabilistic plotting"
            )
        return npz_path
    if candidate.suffix != ".npz":
        raise ValueError(f"Expected a run directory or .npz path, got: {candidate}")
    return candidate


def load_probabilistic_forecast_arrays(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load probabilistic forecast arrays from ``forecasts.npz``.

    Returns:
      ``(actuals, quantile_forecasts, quantile_levels, episode_ids)``
    """
    npz_path = resolve_forecast_npz_path(path)
    with np.load(npz_path, allow_pickle=False) as data:
        actuals = np.asarray(data["actuals"], dtype=np.float64)
        quantile_forecasts = np.asarray(data["quantile_forecasts"], dtype=np.float64)
        quantile_levels = np.asarray(data["quantile_levels"], dtype=np.float64)
        if "episode_ids" in data:
            episode_ids = np.asarray(data["episode_ids"])
        else:
            episode_ids = np.arange(actuals.shape[0], dtype=np.int64)

    if actuals.ndim != 2:
        raise ValueError(f"{npz_path} actuals must be 2D, got shape {actuals.shape}")
    if quantile_forecasts.ndim != 3:
        raise ValueError(
            f"{npz_path} quantile_forecasts must be 3D, got shape {quantile_forecasts.shape}"
        )
    if quantile_levels.ndim != 1:
        raise ValueError(
            f"{npz_path} quantile_levels must be 1D, got shape {quantile_levels.shape}"
        )
    if quantile_forecasts.shape[0] != actuals.shape[0]:
        raise ValueError(
            f"{npz_path} episode count mismatch: actuals={actuals.shape[0]} "
            f"quantile_forecasts={quantile_forecasts.shape[0]}"
        )
    if quantile_forecasts.shape[2] != actuals.shape[1]:
        raise ValueError(
            f"{npz_path} horizon mismatch: actuals={actuals.shape[1]} "
            f"quantile_forecasts={quantile_forecasts.shape[2]}"
        )
    if quantile_forecasts.shape[1] != quantile_levels.shape[0]:
        raise ValueError(
            f"{npz_path} quantile level mismatch: quantile_forecasts has "
            f"{quantile_forecasts.shape[1]} levels but quantile_levels has "
            f"{quantile_levels.shape[0]}"
        )
    if episode_ids.shape[0] != actuals.shape[0]:
        raise ValueError(
            f"{npz_path} episode_ids length mismatch: episode_ids={episode_ids.shape[0]} "
            f"actuals={actuals.shape[0]}"
        )

    return actuals, quantile_forecasts, quantile_levels, episode_ids


def interpolate_quantile_trace(
    quantile_forecast: np.ndarray,
    quantile_levels: np.ndarray,
    target_quantile: float,
) -> np.ndarray:
    """Interpolate one quantile trace from a single-episode quantile forecast.

    Args:
      quantile_forecast: shape ``(n_quantiles, horizon)``
      quantile_levels: shape ``(n_quantiles,)`` ascending in [0, 1]
      target_quantile: desired quantile in [0, 1]
    """
    if quantile_forecast.ndim != 2:
        raise ValueError(
            f"quantile_forecast must be 2D (n_quantiles, horizon), got {quantile_forecast.shape}"
        )
    if quantile_levels.ndim != 1:
        raise ValueError(
            f"quantile_levels must be 1D (n_quantiles,), got {quantile_levels.shape}"
        )
    if quantile_forecast.shape[0] != quantile_levels.shape[0]:
        raise ValueError(
            "quantile_forecast first dimension must match quantile_levels length: "
            f"{quantile_forecast.shape[0]} vs {quantile_levels.shape[0]}"
        )
    if not (0.0 <= target_quantile <= 1.0):
        raise ValueError(f"target_quantile must be in [0, 1], got {target_quantile}")

    if np.any(np.diff(quantile_levels) < 0):
        raise ValueError("quantile_levels must be sorted in non-decreasing order")

    exact_matches = np.where(np.isclose(quantile_levels, target_quantile))[0]
    if exact_matches.size > 0:
        return quantile_forecast[int(exact_matches[0])]

    if target_quantile < quantile_levels[0] or target_quantile > quantile_levels[-1]:
        raise ValueError(
            f"target_quantile={target_quantile} outside available quantile range "
            f"[{quantile_levels[0]}, {quantile_levels[-1]}]"
        )

    idx_hi = int(np.searchsorted(quantile_levels, target_quantile))
    idx_lo = idx_hi - 1
    q_lo = float(quantile_levels[idx_lo])
    q_hi = float(quantile_levels[idx_hi])
    frac = (target_quantile - q_lo) / (q_hi - q_lo)
    return quantile_forecast[idx_lo] + frac * (
        quantile_forecast[idx_hi] - quantile_forecast[idx_lo]
    )
