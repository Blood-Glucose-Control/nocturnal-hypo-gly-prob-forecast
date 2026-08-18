#!/usr/bin/env python3
"""Plot training-step sweep metrics from evaluation ``results_summary.json`` files.

What this visualization tells you
---------------------------------
- Short/medium-run step sensitivity of forecasting quality and calibration.
- Which configured series value (for example context length) is most robust
  across datasets.

What to look for
----------------
- Early gains that saturate (candidate checkpoint for efficient training).
- Coverage rows drifting away from nominal lines (calibration instability).
- Series rank flips across datasets (non-transferable settings).

Required record fields
----------------------
Each discovered ``results_summary.json`` should include:
- ``dataset`` (string)
- ``checkpoint`` (string or null) where ``step_<N>`` can be parsed when present
- ``timestamp`` (string) for deduplication
- ``config`` (object) containing:
  - the series field configured by ``--series-field`` (default: ``context_length``)
  - optional ``covariate_cols`` when using covariate filtering

The script overlays one line per series value (``--series-values``) across
datasets and metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "DejaVu Sans"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METRICS: list[tuple[str, str]] = [
    ("overall_rmse", "RMSE (mmol/L)"),
    ("overall_wql", "WQL"),
    ("overall_coverage_50", "Coverage 50%"),
    ("overall_coverage_80", "Coverage 80%"),
    ("overall_sharpness_50", "Sharpness 50"),
    ("overall_sharpness_80", "Sharpness 80"),
    ("overall_brier", "Brier Score"),
    ("overall_dilate_g001", "DILATE (γ=0.01)"),
]

DATASET_LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
}
DEFAULT_DATASETS = list(DATASET_LABELS)

DEFAULT_SERIES_VALUES = [512, 256, 128, 64]
DEFAULT_SERIES_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
DEFAULT_SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _extract_step(checkpoint: str | None, default_final: int = 10000) -> int:
    """Return training step from a checkpoint path."""
    if checkpoint is None:
        return 0
    match = re.search(r"step_(\d+)", checkpoint)
    if match:
        return int(match.group(1))
    return default_final


def _load_from_dir(
    base: Path,
    datasets: set[str],
    series_field: str,
    series_values: set[int],
    required_covariate: str | None,
    step_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Recursively find results_summary.json files and build record list."""
    records: list[dict[str, Any]] = []

    for path in sorted(base.rglob("results_summary.json")):
        try:
            with path.open() as file_obj:
                payload: dict[str, Any] = json.load(file_obj)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
            continue

        dataset = payload.get("dataset")
        if dataset not in datasets:
            continue

        config = payload.get("config") or {}
        series_value = config.get(series_field)
        if not isinstance(series_value, int) or series_value not in series_values:
            continue

        if required_covariate:
            covariate_cols = config.get("covariate_cols") or []
            if required_covariate not in covariate_cols:
                continue

        step = _extract_step(payload.get("checkpoint"))
        if step_filter is not None and step != step_filter:
            continue

        row: dict[str, Any] = {
            "step": step,
            "series": series_value,
            "dataset": dataset,
            "timestamp": payload.get("timestamp", ""),
        }
        for key, _ in METRICS:
            value = payload.get(key)
            row[key] = float(value) if value is not None else np.nan
        records.append(row)

    return records


def load_all_data(
    sweep_dir: Path,
    ablation_dir: Path,
    datasets: list[str],
    series_field: str,
    series_values: list[int],
    required_covariate: str | None,
) -> pd.DataFrame:
    """Load and merge data from step-sweep and optional step-zero sources."""
    dataset_set = set(datasets)
    series_set = set(series_values)
    records: list[dict[str, Any]] = []

    if sweep_dir.exists():
        sweep_records = _load_from_dir(
            sweep_dir,
            datasets=dataset_set,
            series_field=series_field,
            series_values=series_set,
            required_covariate=required_covariate,
        )
        records.extend(sweep_records)
        print(f"Loaded {len(sweep_records)} records from step-sweep dir.")
    else:
        print(f"[INFO] Step-sweep dir not found: {sweep_dir}", file=sys.stderr)

    if ablation_dir.exists():
        zero_shot_records = _load_from_dir(
            ablation_dir,
            datasets=dataset_set,
            series_field=series_field,
            series_values=series_set,
            required_covariate=required_covariate,
            step_filter=0,
        )
        records.extend(zero_shot_records)
        print(
            f"Loaded {len(zero_shot_records)} zero-shot records from ctx-ablation dir."
        )
    else:
        print(f"[INFO] Ctx-ablation dir not found: {ablation_dir}", file=sys.stderr)

    if not records:
        print("ERROR: No data found in either directory.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    df = (
        df.sort_values("timestamp", ascending=True)
        .drop_duplicates(subset=["step", "series", "dataset"], keep="last")
        .sort_values(["series", "dataset", "step"])
        .reset_index(drop=True)
    )
    print(f"\nFinal dataset: {len(df)} unique (step, series, dataset) records")
    print(df.groupby(["series", "step"]).size().to_frame("n_datasets").to_string())
    return df


def _build_series_meta(
    series_values: list[int],
    series_labels: list[str] | None,
) -> dict[int, dict[str, str]]:
    if series_labels is not None and len(series_labels) != len(series_values):
        raise ValueError("--series-labels must match --series-values length")

    meta: dict[int, dict[str, str]] = {}
    for idx, value in enumerate(series_values):
        label = (
            series_labels[idx]
            if series_labels is not None
            else f"context_length={value}"
        )
        meta[value] = {
            "label": label,
            "color": DEFAULT_SERIES_COLORS[idx % len(DEFAULT_SERIES_COLORS)],
            "marker": DEFAULT_SERIES_MARKERS[idx % len(DEFAULT_SERIES_MARKERS)],
        }
    return meta


def make_plot(
    df: pd.DataFrame,
    output_path: Path,
    datasets: list[str],
    series_meta: dict[int, dict[str, str]],
    title: str,
    legend_title: str,
) -> None:
    """Render the metric grid and save to *output_path*."""
    n_rows = len(METRICS)
    n_cols = len(datasets)

    row_ylims: list[tuple[float, float]] = []
    for metric_key, _ in METRICS:
        vals = df[metric_key].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            row_ylims.append((0.0, 1.0))
            continue
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.08 if hi > lo else abs(hi) * 0.08 or 0.05
        row_ylims.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    xticks = sorted(df["step"].dropna().unique().tolist())

    for col_idx, dataset in enumerate(datasets):
        dataset_df = df[df["dataset"] == dataset]

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax: plt.Axes = axes[row_idx][col_idx]

            for series_value, meta in series_meta.items():
                series_df = dataset_df[
                    dataset_df["series"] == series_value
                ].sort_values("step")
                if series_df.empty:
                    continue

                steps = series_df["step"].to_numpy()
                values = series_df[metric_key].to_numpy()
                mask = ~np.isnan(values)
                if mask.sum() < 2:
                    continue

                ax.plot(
                    steps[mask],
                    values[mask],
                    marker=meta["marker"],
                    color=meta["color"],
                    linewidth=1.8,
                    markersize=6,
                    label=meta["label"],
                )

            if metric_key == "overall_coverage_50":
                ax.axhline(0.50, color="black", linewidth=1.0, linestyle=":", alpha=0.7)
            elif metric_key == "overall_coverage_80":
                ax.axhline(0.80, color="black", linewidth=1.0, linestyle=":", alpha=0.7)

            ax.set_ylim(*row_ylims[row_idx])
            ax.set_xticks(xticks)
            ax.xaxis.set_tick_params(rotation=40, labelsize=8)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}" if x > 0 else "ZS")
            )
            ax.yaxis.set_tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

            if row_idx == n_rows - 1:
                ax.set_xlabel("Training steps", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=9)
            if row_idx == 0:
                ax.set_title(
                    DATASET_LABELS.get(dataset, dataset),
                    fontsize=12,
                    fontweight="bold",
                    pad=6,
                )

    handles, labels = axes[0][-1].get_legend_handles_labels()
    handles.append(
        mlines.Line2D([], [], color="black", linewidth=1.0, linestyle=":", alpha=0.7)
    )
    labels.append("Ideal coverage")
    if handles:
        axes[0][-1].legend(
            handles,
            labels,
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
            title=legend_title,
            title_fontsize=8,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-sweep-dir",
        default=str(PROJECT_ROOT / "experiments" / "nocturnal_forecasting_step_sweep"),
        help="Directory containing step-sweep results_summary.json outputs.",
    )
    parser.add_argument(
        "--ctx-ablation-dir",
        default=str(
            PROJECT_ROOT / "experiments" / "nocturnal_forecasting_ctx_ablation"
        ),
        help="Directory containing optional zero-shot results_summary.json outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to include in the grid (default: aleppo_2017 brown_2019 lynch_2022).",
    )
    parser.add_argument(
        "--series-field",
        default="context_length",
        help="Field name under config used to define overlay series (default: %(default)s).",
    )
    parser.add_argument(
        "--series-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_SERIES_VALUES),
        help="Series values for --series-field to include (default: 512 256 128 64).",
    )
    parser.add_argument(
        "--series-labels",
        nargs="+",
        default=None,
        help="Optional labels matching --series-values order.",
    )
    parser.add_argument(
        "--require-covariate",
        default="iob",
        help="Only include runs whose config.covariate_cols contain this value.",
    )
    parser.add_argument(
        "--no-covariate-filter",
        action="store_true",
        help="Disable covariate filtering and include all matching series values.",
    )
    parser.add_argument(
        "--title",
        default="Training Step Sweep Metrics",
        help="Figure title.",
    )
    parser.add_argument(
        "--legend-title",
        default="Series",
        help="Legend title for overlay series.",
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "notes" / "chronos2" / "figures" / "step_sweep_grid.png"
        ),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_covariate = None if args.no_covariate_filter else args.require_covariate
    series_meta = _build_series_meta(
        series_values=args.series_values,
        series_labels=args.series_labels,
    )

    df = load_all_data(
        sweep_dir=Path(args.step_sweep_dir),
        ablation_dir=Path(args.ctx_ablation_dir),
        datasets=args.datasets,
        series_field=args.series_field,
        series_values=args.series_values,
        required_covariate=required_covariate,
    )
    make_plot(
        df=df,
        output_path=Path(args.output),
        datasets=args.datasets,
        series_meta=series_meta,
        title=args.title,
        legend_title=args.legend_title,
    )


if __name__ == "__main__":
    main()
