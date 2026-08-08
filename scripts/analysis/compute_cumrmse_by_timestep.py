# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Compute per-episode cumulative RMSE at multiple forecast horizons for all
models and datasets, then write results/analysis/cumrmse_by_timestep.csv.

For each (model, dataset) pair, the *best* available run with a forecasts.npz
is used (lowest overall RMSE from results_summary.json).

Horizon checkpoints (minutes → steps at 5-min cadence):
  15 min = 3 steps, 30 = 6, 60 = 12, 120 = 24, 240 = 48, 360 = 72, 480 = 96

Per-episode cumRMSE at horizon k:
    cumRMSE_i(k) = sqrt( mean_{t=1..k}( (y_{i,t} - ŷ_{i,t})^2 ) )

The median forecast (q=0.5, index 4 in [0.1…0.9]) is used as ŷ.

Output columns:
    dataset, model,
    15min_mean, 30min_mean, 60min_mean, 120min_mean, 240min_mean, 360min_mean, 480min_mean,
    15min_std,  30min_std,  60min_std,  120min_std,  240min_std,  360min_std,  480min_std

Usage:
    python scripts/analysis/compute_cumrmse_by_timestep.py
    python scripts/analysis/compute_cumrmse_by_timestep.py --out results/analysis/cumrmse_by_timestep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022", "tamborlane_2008"]

# Models to include, in display order.
# Groups: foundation models first, then deep-learning supervised.
MODELS = [
    "chronos2",
    "moirai",
    "timesfm",
    "ttm",
    "toto",
    "moment",
    "sundial",
    "tft",
    "deepar",
    "timegrad",
    "tide",
    "patchtst",
]

# Horizon steps (5-min cadence) → column labels
HORIZONS: list[tuple[int, str]] = [
    (3, "15min"),
    (6, "30min"),
    (12, "60min"),
    (24, "120min"),
    (48, "240min"),
    (72, "360min"),
    (96, "480min"),
]

# Median quantile index in [0.1, 0.2, …, 0.9]
MEDIAN_IDX = 4  # q=0.5

# ---------------------------------------------------------------------------
# TiDE covariate-exclusion: skip runs that used known (future) covariates.
# Runs whose cov_bucket matches any of these are excluded for TiDE.
# ---------------------------------------------------------------------------
TIDE_EXCLUDED_COV_BUCKETS = {"iob", "iob_cob", "cob"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rmse_from_npz(run_dir: Path) -> float | None:
    """
    Compute overall RMSE from scratch using the median forecast in forecasts.npz.
    This is the same computation used for the output values, ensuring run
    selection and reported metrics are always consistent.
    """
    npz_path = run_dir / "forecasts.npz"
    if not npz_path.exists():
        return None
    try:
        npz = np.load(npz_path)
        actuals = npz["actuals"]  # (n_eps, fh)
        q_fc = npz["quantile_forecasts"]  # (n_eps, 9, fh) or (0,0,0)
        if q_fc.ndim == 3 and q_fc.shape[1] > 0:
            median_fc = q_fc[:, MEDIAN_IDX, :]
        else:
            median_fc = npz["predictions"]
        return float(np.sqrt(np.mean((actuals - median_fc) ** 2)))
    except Exception:
        return None


def _cov_bucket(run_dir: Path) -> str:
    """Return cov_bucket from results_summary.json, or '' if missing."""
    summary = run_dir / "results_summary.json"
    if not summary.exists():
        return ""
    try:
        with open(summary) as f:
            d = json.load(f)
        return str(d.get("cov_bucket", ""))
    except Exception:
        return ""


def find_best_run(model: str, dataset: str) -> Path | None:
    """
    Scan experiments/nocturnal_forecasting/512ctx_96fh/<model>/ for runs
    that (a) match *dataset*, (b) have a forecasts.npz, and optionally
    (c) pass the TiDE known-covariate exclusion filter.

    Returns the run directory with the lowest overall_rmse, or None.
    """
    model_dir = (
        REPO_ROOT / "experiments" / "nocturnal_forecasting" / "512ctx_96fh" / model
    )
    if not model_dir.exists():
        return None

    best_rmse = float("inf")
    best_run: Path | None = None

    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if dataset not in run_dir.name:
            continue
        if not (run_dir / "forecasts.npz").exists():
            continue

        # TiDE: skip runs trained with known covariates
        if model == "tide":
            cov = _cov_bucket(run_dir)
            if cov in TIDE_EXCLUDED_COV_BUCKETS:
                print(f"  [tide] Skipping {run_dir.name} (cov={cov})")
                continue

        rmse = _rmse_from_npz(run_dir)
        if rmse is None:
            continue

        if rmse < best_rmse:
            best_rmse = rmse
            best_run = run_dir

    return best_run


def compute_cumrmse(actuals: np.ndarray, median_fc: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    actuals   : (n_episodes, fh)
    median_fc : (n_episodes, fh)

    Returns
    -------
    cumrmse : (n_episodes, fh)  — cumRMSE_i(k) for every k
    """
    sq_err = (actuals - median_fc) ** 2  # (n_eps, fh)
    # cumulative mean of squared errors up to each step
    cum_mean_sq = np.cumsum(sq_err, axis=1) / np.arange(1, sq_err.shape[1] + 1)
    return np.sqrt(cum_mean_sq)  # (n_eps, fh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_csv(out_path: Path) -> None:
    fieldnames = (
        ["dataset", "model", "total_episodes"]
        + [f"{label}_mean" for _, label in HORIZONS]
        + [f"{label}_std" for _, label in HORIZONS]
    )

    rows: list[dict] = []

    for dataset in DATASETS:
        for model in MODELS:
            run_dir = find_best_run(model, dataset)
            if run_dir is None:
                print(f"  WARNING: no valid run found for {model}/{dataset} — skipped")
                continue

            rmse = _rmse_from_npz(run_dir)
            print(f"  {model:12s}  {dataset:15s}  run={run_dir.name}  rmse={rmse:.4f}")

            npz = np.load(run_dir / "forecasts.npz")
            actuals = npz["actuals"]  # (n_eps, fh)
            q_fc = npz[
                "quantile_forecasts"
            ]  # (n_eps, 9, fh) or (0,0,0) for point models

            if q_fc.ndim == 3 and q_fc.shape[1] > 0:
                median_fc = q_fc[:, MEDIAN_IDX, :]  # (n_eps, fh)
            else:
                # Point-only model (e.g. TTM, Moment) — use predictions directly
                median_fc = npz["predictions"]  # (n_eps, fh)

            cumrmse = compute_cumrmse(actuals, median_fc)  # (n_eps, fh)
            n_episodes = cumrmse.shape[0]

            row: dict = {
                "dataset": dataset,
                "model": model,
                "total_episodes": n_episodes,
            }
            for step, label in HORIZONS:
                col = cumrmse[:, step - 1]  # 0-indexed: step k → index k-1
                row[f"{label}_mean"] = round(float(np.mean(col)), 4)
                row[f"{label}_std"] = round(float(np.std(col)), 4)

            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out",
        default="results/analysis/cumrmse_by_timestep_all_models.csv",
        help="Output CSV path (default: results/analysis/cumrmse_by_timestep_all_models.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_csv(Path(args.out))
