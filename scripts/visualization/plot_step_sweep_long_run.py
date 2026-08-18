#!/usr/bin/env python3
"""Plot long-run training-step sweep metrics from ``results_summary.json`` files.

What this visualization tells you
---------------------------------
- How each metric evolves over long fine-tuning horizons.
- Whether one training recipe dominates another consistently or only at
  specific step ranges.

What to look for
----------------
- Coverage lines near 0.50 / 0.80 while RMSE/WQL/brier improve.
- Late-step regressions after an apparent optimum.
- Cross-series tradeoffs (e.g., sharper intervals but worse calibration).

Required record fields
----------------------
Each discovered ``results_summary.json`` should include:
- ``dataset`` (string)
- ``checkpoint`` (string or null) where ``step_<N>`` can be parsed when present
- ``timestamp`` (string) for deduplication
- ``config`` (object), optionally used for zero-shot filtering

Series assignment
-----------------
- Non-zero-shot runs are assigned by matching output directory paths against
  ``--series-stems``.
- Optional zero-shot rows (step 0) can be loaded from ``--ctx-ablation-dir`` and
  expanded across all requested series.
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

DEFAULT_SERIES_STEMS = [
    "16_bg_iob_ia_high_lr_100k",
    "17_bg_iob_ia_high_lr_100k_ensemble",
]
DEFAULT_SERIES_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
DEFAULT_SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

ALL_STEPS = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
STEP_TICK_STRIDE = 10000

DEFAULT_LONG_RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "nocturnal_forecasting_long_run_step_sweep"
    / "512ctx_96fh"
    / "chronos2"
)
DEFAULT_CTX_ABLATION_DIR = (
    PROJECT_ROOT / "experiments" / "nocturnal_forecasting_ctx_ablation"
)


def _extract_step(checkpoint: str | None, default_final: int = 100000) -> int:
    """Return training step from a checkpoint path."""
    if checkpoint is None:
        return 0
    match = re.search(r"step_(\d+)", checkpoint)
    if match:
        return int(match.group(1))
    return default_final


def _build_series_meta(
    series_stems: list[str],
    series_labels: list[str] | None,
) -> dict[str, dict[str, str]]:
    if series_labels is not None and len(series_labels) != len(series_stems):
        raise ValueError("--series-labels must match --series-stems length")

    meta: dict[str, dict[str, str]] = {}
    for idx, stem in enumerate(series_stems):
        label = series_labels[idx] if series_labels is not None else stem
        meta[stem] = {
            "label": label,
            "color": DEFAULT_SERIES_COLORS[idx % len(DEFAULT_SERIES_COLORS)],
            "marker": DEFAULT_SERIES_MARKERS[idx % len(DEFAULT_SERIES_MARKERS)],
        }
    return meta


def _is_matching_zero_shot(
    config: dict[str, Any],
    required_context_length: int | None,
    required_covariate: str | None,
) -> bool:
    if (
        required_context_length is not None
        and config.get("context_length") != required_context_length
    ):
        return False
    if required_covariate:
        covariate_cols = config.get("covariate_cols") or []
        return required_covariate in covariate_cols
    return True


def _load_from_dir(
    base: Path,
    datasets: set[str],
    series_stems: list[str],
    required_zero_shot_context_length: int | None,
    required_zero_shot_covariate: str | None,
    step_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Recursively find results_summary.json files and return record list."""
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

        step = _extract_step(payload.get("checkpoint"))
        if step_filter is not None and step != step_filter:
            continue

        series_key = None
        path_str = str(path)
        for stem in series_stems:
            if stem in path_str:
                series_key = stem
                break

        if series_key is None and step == 0:
            config = payload.get("config") or {}
            if _is_matching_zero_shot(
                config=config,
                required_context_length=required_zero_shot_context_length,
                required_covariate=required_zero_shot_covariate,
            ):
                series_key = "zero_shot"

        if series_key is None:
            continue

        row: dict[str, Any] = {
            "step": step,
            "series": series_key,
            "dataset": dataset,
            "timestamp": payload.get("timestamp", ""),
        }
        for key, _ in METRICS:
            value = payload.get(key)
            row[key] = float(value) if value is not None else np.nan
        records.append(row)

    return records


def load_all_data(
    long_run_dir: Path,
    ctx_ablation_dir: Path,
    datasets: list[str],
    series_stems: list[str],
    required_zero_shot_context_length: int | None,
    required_zero_shot_covariate: str | None,
) -> pd.DataFrame:
    """Load and merge long-run and optional step-zero data."""
    dataset_set = set(datasets)
    records: list[dict[str, Any]] = []

    if long_run_dir.exists():
        long_run_records = _load_from_dir(
            long_run_dir,
            datasets=dataset_set,
            series_stems=series_stems,
            required_zero_shot_context_length=required_zero_shot_context_length,
            required_zero_shot_covariate=required_zero_shot_covariate,
        )
        records.extend(long_run_records)
        print(f"Loaded {len(long_run_records)} records from long-run step-sweep dir.")
    else:
        print(
            f"[INFO] Long-run step-sweep dir not found: {long_run_dir}", file=sys.stderr
        )

    if ctx_ablation_dir.exists():
        zero_shot_records = _load_from_dir(
            ctx_ablation_dir,
            datasets=dataset_set,
            series_stems=series_stems,
            required_zero_shot_context_length=required_zero_shot_context_length,
            required_zero_shot_covariate=required_zero_shot_covariate,
            step_filter=0,
        )
        records.extend(zero_shot_records)
        print(
            f"Loaded {len(zero_shot_records)} zero-shot records from ctx-ablation dir."
        )
    else:
        print(f"[INFO] Ctx-ablation dir not found: {ctx_ablation_dir}", file=sys.stderr)

    if not records:
        print("ERROR: No data found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)

    # Expand zero-shot rows so they appear for every explicit series.
    zero_shot_rows = df[df["series"] == "zero_shot"].copy()
    if not zero_shot_rows.empty:
        expanded = []
        for stem in series_stems:
            rows = zero_shot_rows.copy()
            rows["series"] = stem
            expanded.append(rows)
        df = pd.concat([df[df["series"] != "zero_shot"]] + expanded, ignore_index=True)

    df = (
        df.sort_values("timestamp", ascending=True)
        .drop_duplicates(subset=["step", "series", "dataset"], keep="last")
        .sort_values(["series", "dataset", "step"])
        .reset_index(drop=True)
    )
    print(f"\nFinal dataset: {len(df)} unique (step, series, dataset) records")
    print(df.groupby(["series", "step"]).size().to_frame("n_datasets").to_string())
    return df


def make_plot(
    df: pd.DataFrame,
    output_path: Path,
    datasets: list[str],
    series_meta: dict[str, dict[str, str]],
    step_tick_stride: int,
    title: str,
    legend_title: str,
) -> None:
    """Render the metric grid and save."""
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

    if step_tick_stride <= 0:
        raise ValueError("--step-tick-stride must be > 0")
    xticks = [step for step in ALL_STEPS if step % step_tick_stride == 0]

    for col_idx, dataset in enumerate(datasets):
        dataset_df = df[df["dataset"] == dataset]

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax: plt.Axes = axes[row_idx][col_idx]

            for series_key, meta in series_meta.items():
                series_df = dataset_df[dataset_df["series"] == series_key].sort_values(
                    "step"
                )
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
                    markersize=4,
                    label=meta["label"],
                )

            if metric_key == "overall_coverage_50":
                ax.axhline(0.50, color="black", linewidth=1.0, linestyle=":", alpha=0.7)
            elif metric_key == "overall_coverage_80":
                ax.axhline(0.80, color="black", linewidth=1.0, linestyle=":", alpha=0.7)

            ax.set_ylim(*row_ylims[row_idx])
            ax.set_xticks(xticks)
            ax.xaxis.set_tick_params(rotation=40, labelsize=7)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: "ZS" if x == 0 else f"{int(x) // 1000}k"
                )
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
        "--long-run-dir",
        default=str(DEFAULT_LONG_RUN_DIR),
        help="Directory containing long-run step-sweep results_summary.json outputs.",
    )
    parser.add_argument(
        "--ctx-ablation-dir",
        default=str(DEFAULT_CTX_ABLATION_DIR),
        help="Directory containing optional zero-shot results_summary.json outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to include in the grid (default: aleppo_2017 brown_2019 lynch_2022).",
    )
    parser.add_argument(
        "--series-stems",
        nargs="+",
        default=list(DEFAULT_SERIES_STEMS),
        help="Path substrings used to identify overlay series.",
    )
    parser.add_argument(
        "--series-labels",
        nargs="+",
        default=None,
        help="Optional labels matching --series-stems order.",
    )
    parser.add_argument(
        "--step-tick-stride",
        type=int,
        default=STEP_TICK_STRIDE,
        help="Major x-axis tick spacing in training steps (default: 10000).",
    )
    parser.add_argument(
        "--zero-shot-context-length",
        type=int,
        default=512,
        help="Context length required for zero-shot records from --ctx-ablation-dir.",
    )
    parser.add_argument(
        "--zero-shot-require-covariate",
        default="iob",
        help="Required covariate in config.covariate_cols for zero-shot records.",
    )
    parser.add_argument(
        "--no-zero-shot-covariate-filter",
        action="store_true",
        help="Disable zero-shot covariate filtering.",
    )
    parser.add_argument(
        "--title",
        default="Long-Run Training Step Sweep Metrics",
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
            PROJECT_ROOT / "results" / "figures" / "step_sweep_grid_long_run.png"
        ),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_zero_shot_covariate = (
        None if args.no_zero_shot_covariate_filter else args.zero_shot_require_covariate
    )

    series_meta = _build_series_meta(
        series_stems=args.series_stems,
        series_labels=args.series_labels,
    )

    df = load_all_data(
        long_run_dir=Path(args.long_run_dir),
        ctx_ablation_dir=Path(args.ctx_ablation_dir),
        datasets=args.datasets,
        series_stems=args.series_stems,
        required_zero_shot_context_length=args.zero_shot_context_length,
        required_zero_shot_covariate=required_zero_shot_covariate,
    )
    make_plot(
        df=df,
        output_path=Path(args.output),
        datasets=args.datasets,
        series_meta=series_meta,
        step_tick_stride=args.step_tick_stride,
        title=args.title,
        legend_title=args.legend_title,
    )


if __name__ == "__main__":
    main()
