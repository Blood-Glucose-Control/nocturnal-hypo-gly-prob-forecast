#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Model-agnostic percentile grid for probabilistic forecast runs.

What this visualization tells you
---------------------------------
- Typical episode behavior at different error percentiles for each dataset.
- How median forecast shape and interval width change from easier to harder
  episodes.

What to look for
----------------
- Median trace misses around turning points in high-percentile (hard) panels.
- Intervals that fail to contain the realized trace near threshold crossings.
- Dataset rows where hard-episode degradation is disproportionately severe.

Required input format
---------------------
1) A run-map CSV (`--run-map-csv`) with at least these columns:
   - `dataset`   : dataset identifier (row label in the grid)
   - `run_path`  : path to a run directory containing `forecasts.npz`, or a direct
                   path to `forecasts.npz`
   Optional:
   - `dataset_label` : display label for the dataset row

2) For each `run_path`, `forecasts.npz` must contain:
   - `actuals`             : float array, shape `(n_episodes, horizon)`
   - `quantile_forecasts`  : float array, shape `(n_episodes, n_quantiles, horizon)`
   - `quantile_levels`     : float array, shape `(n_quantiles,)`, values in `[0, 1]`
   Optional:
   - `episode_ids`         : array, shape `(n_episodes,)`

What the script plots
---------------------
- Rows: datasets from the run-map CSV
- Columns: episode quality percentiles requested by `--episode-percentiles`
  (percentiles of per-episode RMSE for the selected median quantile)
- Each panel:
  - actual BG trace
  - median forecast
  - inner prediction interval (default 25%-75%)
  - outer prediction interval (default 10%-90%)
  - optional threshold line (default 3.9 mmol/L)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from src.visualization.nocturnal import (
    interpolate_quantile_trace,
    load_probabilistic_forecast_arrays,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RUN_MAP_CSV = "experiments/nocturnal_forecasting/run_map.csv"
DEFAULT_OUTPUT = "results/figures/probabilistic_forecast_episode_grid.png"
DEFAULT_EPISODE_PERCENTILES = (10, 25, 50, 75, 90)
DEFAULT_OUTER_INTERVAL = (0.10, 0.90)
DEFAULT_INNER_INTERVAL = (0.25, 0.75)
DEFAULT_FORECAST_STEP_MINUTES = 5

PINK_DARK = "#c5449a"
PINK_LIGHT = "#f2c6e8"
PINK_LIGHTER = "#f9e5f5"


@dataclass(frozen=True)
class RunMapRow:
    dataset: str
    dataset_label: str
    run_path: Path


@dataclass(frozen=True)
class EpisodePanel:
    dataset_label: str
    percentile: int
    episode_id: str
    actual: np.ndarray
    median: np.ndarray
    inner_low: np.ndarray
    inner_high: np.ndarray
    outer_low: np.ndarray
    outer_high: np.ndarray
    rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-map-csv",
        default=DEFAULT_RUN_MAP_CSV,
        help="CSV mapping datasets to run paths (default: %(default)s).",
    )
    parser.add_argument(
        "--episode-percentiles",
        nargs="+",
        type=int,
        default=list(DEFAULT_EPISODE_PERCENTILES),
        help="Episode RMSE percentiles to display as columns (default: 10 25 50 75 90).",
    )
    parser.add_argument(
        "--median-quantile",
        type=float,
        default=0.50,
        help="Quantile level used to score episode RMSE (default: %(default)s).",
    )
    parser.add_argument(
        "--outer-interval",
        nargs=2,
        type=float,
        default=list(DEFAULT_OUTER_INTERVAL),
        metavar=("LOW", "HIGH"),
        help="Outer prediction interval quantiles in [0,1] (default: 0.10 0.90).",
    )
    parser.add_argument(
        "--inner-interval",
        nargs=2,
        type=float,
        default=list(DEFAULT_INNER_INTERVAL),
        metavar=("LOW", "HIGH"),
        help="Inner prediction interval quantiles in [0,1] (default: 0.25 0.75).",
    )
    parser.add_argument(
        "--forecast-step-minutes",
        type=int,
        default=DEFAULT_FORECAST_STEP_MINUTES,
        help="Minutes per forecast step for the x-axis (default: %(default)s).",
    )
    parser.add_argument(
        "--hypo-threshold",
        type=float,
        default=3.9,
        help="Optional horizontal threshold line value (default: %(default)s).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=0.0,
        help="Lower y-axis bound (default: %(default)s).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=22.0,
        help="Upper y-axis bound (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output PNG path (default: %(default)s).",
    )
    parser.add_argument(
        "--output-pdf",
        default=None,
        help="Optional PDF output path. If omitted, only PNG is written.",
    )
    parser.add_argument(
        "--title",
        default="Probabilistic Forecast Episode Grid",
        help="Figure title.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution.")
    return parser.parse_args()


def _parse_run_map(csv_path: Path) -> list[RunMapRow]:
    rows: list[RunMapRow] = []
    with csv_path.open() as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"dataset", "run_path"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{csv_path} must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            run_path = (row.get("run_path") or "").strip()
            if not dataset or not run_path:
                continue
            dataset_label = (row.get("dataset_label") or dataset).strip()
            rows.append(
                RunMapRow(
                    dataset=dataset,
                    dataset_label=dataset_label,
                    run_path=Path(run_path),
                )
            )
    if not rows:
        raise ValueError(f"{csv_path} did not contain any valid dataset/run_path rows")
    return rows


def _validate_unit_interval(low: float, high: float, name: str) -> tuple[float, float]:
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= low < high <= 1, got {(low, high)}")
    return low, high


def _select_episode_index(rmse: np.ndarray, percentile: int) -> int:
    target = float(np.percentile(rmse, percentile))
    return int(np.argmin(np.abs(rmse - target)))


def _select_quantile_batch(
    quantile_forecasts: np.ndarray,
    quantile_levels: np.ndarray,
    target_quantile: float,
) -> np.ndarray:
    matches = np.where(np.isclose(quantile_levels, target_quantile))[0]
    if matches.size > 0:
        return quantile_forecasts[:, int(matches[0]), :]
    return np.stack(
        [
            interpolate_quantile_trace(qf, quantile_levels, target_quantile)
            for qf in quantile_forecasts
        ],
        axis=0,
    )


def _build_panels_for_dataset(
    row: RunMapRow,
    episode_percentiles: list[int],
    median_quantile: float,
    inner_interval: tuple[float, float],
    outer_interval: tuple[float, float],
) -> list[EpisodePanel]:
    actuals, quantile_forecasts, quantile_levels, episode_ids = (
        load_probabilistic_forecast_arrays(row.run_path)
    )

    median_forecasts = _select_quantile_batch(
        quantile_forecasts, quantile_levels, median_quantile
    )
    inner_low = _select_quantile_batch(
        quantile_forecasts, quantile_levels, inner_interval[0]
    )
    inner_high = _select_quantile_batch(
        quantile_forecasts, quantile_levels, inner_interval[1]
    )
    outer_low = _select_quantile_batch(
        quantile_forecasts, quantile_levels, outer_interval[0]
    )
    outer_high = _select_quantile_batch(
        quantile_forecasts, quantile_levels, outer_interval[1]
    )

    rmse = np.sqrt(np.mean((actuals - median_forecasts) ** 2, axis=1))

    panels: list[EpisodePanel] = []
    for percentile in episode_percentiles:
        idx = _select_episode_index(rmse, percentile)
        panels.append(
            EpisodePanel(
                dataset_label=row.dataset_label,
                percentile=percentile,
                episode_id=str(episode_ids[idx]),
                actual=actuals[idx],
                median=median_forecasts[idx],
                inner_low=inner_low[idx],
                inner_high=inner_high[idx],
                outer_low=outer_low[idx],
                outer_high=outer_high[idx],
                rmse=float(rmse[idx]),
            )
        )
    return panels


def _plot_panel(
    ax: plt.Axes,
    panel: EpisodePanel,
    step_minutes: int,
    y_min: float,
    y_max: float,
    threshold: float,
) -> None:
    horizon = len(panel.actual)
    time_hours = np.arange(horizon) * step_minutes / 60.0

    ax.fill_between(
        time_hours,
        panel.outer_low,
        panel.outer_high,
        color=PINK_LIGHTER,
        alpha=0.9,
        label="Outer PI",
    )
    ax.fill_between(
        time_hours,
        panel.inner_low,
        panel.inner_high,
        color=PINK_LIGHT,
        alpha=0.9,
        label="Inner PI",
    )
    ax.plot(time_hours, panel.median, color=PINK_DARK, lw=1.4, label="Forecast median")
    ax.plot(time_hours, panel.actual, color="black", lw=1.3, label="Actual")
    ax.axhline(
        threshold,
        color="#d62728",
        lw=0.8,
        ls="--",
        alpha=0.7,
        label=f"Threshold ({threshold:g})",
    )

    ep_label = panel.episode_id.split("::")[-1]
    min_bg = float(np.min(panel.actual))
    ax.set_title(
        f"{ep_label} · RMSE {panel.rmse:.2f} · min {min_bg:.1f}",
        fontsize=7.5,
        pad=2,
    )
    ax.set_xlim(0, time_hours[-1] if len(time_hours) > 0 else 0)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=6.5)


def build_grid(args: argparse.Namespace) -> None:
    run_map = _parse_run_map(Path(args.run_map_csv))
    if args.forecast_step_minutes <= 0:
        raise ValueError("--forecast-step-minutes must be > 0")
    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be strictly less than --y-max")

    episode_percentiles = [int(p) for p in args.episode_percentiles]
    if any(p < 0 or p > 100 for p in episode_percentiles):
        raise ValueError("--episode-percentiles values must be between 0 and 100")

    inner_interval = _validate_unit_interval(*args.inner_interval, "--inner-interval")
    outer_interval = _validate_unit_interval(*args.outer_interval, "--outer-interval")
    if not (
        outer_interval[0] <= inner_interval[0] < inner_interval[1] <= outer_interval[1]
    ):
        raise ValueError(
            "Expected outer interval to envelop inner interval. "
            f"Got inner={inner_interval} outer={outer_interval}"
        )

    rows = [
        _build_panels_for_dataset(
            row=row,
            episode_percentiles=episode_percentiles,
            median_quantile=args.median_quantile,
            inner_interval=inner_interval,
            outer_interval=outer_interval,
        )
        for row in run_map
    ]

    n_rows = len(rows)
    n_cols = len(episode_percentiles)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3.2 * n_cols + 1.0, 2.8 * n_rows + 1.4),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, panels in enumerate(rows):
        for col_idx, panel in enumerate(panels):
            ax = axes[row_idx, col_idx]
            _plot_panel(
                ax=ax,
                panel=panel,
                step_minutes=args.forecast_step_minutes,
                y_min=args.y_min,
                y_max=args.y_max,
                threshold=args.hypo_threshold,
            )
            if col_idx == 0:
                ax.set_ylabel("BG (mmol/L)", fontsize=7.5)
                ax.annotate(
                    panel.dataset_label,
                    xy=(-0.30, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    rotation=90,
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel("Hours ahead", fontsize=7.5)

    for col_idx, percentile in enumerate(episode_percentiles):
        axes[0, col_idx].annotate(
            f"P{percentile}",
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(args.title, fontsize=11.5, y=1.01)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if args.output_pdf:
        output_pdf_path = Path(args.output_pdf)
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf_path, bbox_inches="tight")
        print(f"Saved: {output_pdf_path}")
    plt.close(fig)


def main() -> None:
    build_grid(parse_args())


if __name__ == "__main__":
    main()
