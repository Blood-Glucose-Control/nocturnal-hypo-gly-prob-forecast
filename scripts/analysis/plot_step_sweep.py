#!/usr/bin/env python3
"""Plot Chronos-2 IOB training step sweep results.

Reads results from two directories:
  - experiments/nocturnal_forecasting_step_sweep/  (steps 2000–10000)
  - experiments/nocturnal_forecasting_ctx_ablation/ (zero-shot, step 0)

Produces an (8 metrics) × (3 datasets) grid with one line per IOB context
config (ctx = 512, 256, 128, 64).  X-axis = training steps (0–10000).

Output:
    notes/chronos2/figures/step_sweep_grid.png

Usage:
    python scripts/analysis/plot_step_sweep.py
    python scripts/analysis/plot_step_sweep.py --output path/to/out.png
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

# (json_key, y-axis label)
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

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022"]
DATASET_LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
}

CTX_CONFIGS = [512, 256, 128, 64]
CTX_COLORS = {512: "#1f77b4", 256: "#ff7f0e", 128: "#2ca02c", 64: "#d62728"}
CTX_MARKERS = {512: "o", 256: "s", 128: "^", 64: "D"}
CTX_LABELS = {
    512: "ctx=512 (cfg04)",
    256: "ctx=256 (cfg10)",
    128: "ctx=128 (cfg11)",
    64: "ctx=64 (cfg12)",
}

ALL_STEPS = [0, 2000, 4000, 6000, 8000, 10000]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _extract_step(checkpoint: str | None) -> int:
    """Return training step from a checkpoint path.

    - None → 0  (zero-shot)
    - path containing ``step_N`` → N
    - path without ``step_N`` (final model.pt) → 10000
    """
    if checkpoint is None:
        return 0
    m = re.search(r"step_(\d+)", checkpoint)
    if m:
        return int(m.group(1))
    return 10000


def _load_from_dir(base: Path, step_filter: int | None = None) -> list[dict[str, Any]]:
    """Recursively find results_summary.json files and build record list.

    Args:
        base: Root directory to scan.
        step_filter: If not None, only keep records whose inferred step equals
            this value.  Use ``step_filter=0`` to collect only zero-shot runs.

    Returns:
        List of record dicts with keys: step, ctx, dataset, timestamp, and one
        key per metric in METRICS.
    """
    records: list[dict[str, Any]] = []

    for p in sorted(base.rglob("results_summary.json")):
        try:
            with open(p) as f:
                d: dict[str, Any] = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not read {p}: {exc}", file=sys.stderr)
            continue

        cfg = d.get("config") or {}
        ctx = cfg.get("context_length")
        if ctx not in CTX_CONFIGS:
            continue

        # Keep only IOB runs (must contain 'iob' in covariate_cols)
        cov_cols: list[str] = cfg.get("covariate_cols") or []
        if "iob" not in cov_cols:
            continue

        dataset = d.get("dataset")
        if dataset not in DATASETS:
            continue

        step = _extract_step(d.get("checkpoint"))
        if step_filter is not None and step != step_filter:
            continue

        timestamp = d.get("timestamp", "")

        row: dict[str, Any] = {
            "step": step,
            "ctx": ctx,
            "dataset": dataset,
            "timestamp": timestamp,
        }
        for key, _ in METRICS:
            val = d.get(key)
            row[key] = float(val) if val is not None else np.nan

        records.append(row)

    return records


def load_all_data() -> pd.DataFrame:
    """Load and merge data from step-sweep and ctx-ablation directories."""
    sweep_dir = PROJECT_ROOT / "experiments" / "nocturnal_forecasting_step_sweep"
    ablation_dir = PROJECT_ROOT / "experiments" / "nocturnal_forecasting_ctx_ablation"

    records: list[dict[str, Any]] = []

    # Steps 2000–10000 from the dedicated step-sweep experiment
    if sweep_dir.exists():
        sweep_records = _load_from_dir(sweep_dir)
        records.extend(sweep_records)
        print(f"Loaded {len(sweep_records)} records from step-sweep dir.")
    else:
        print(f"[INFO] Step-sweep dir not found: {sweep_dir}", file=sys.stderr)

    # Step 0 (zero-shot) from the ctx-ablation experiment
    if ablation_dir.exists():
        zs_records = _load_from_dir(ablation_dir, step_filter=0)
        records.extend(zs_records)
        print(f"Loaded {len(zs_records)} zero-shot records from ctx-ablation dir.")
    else:
        print(f"[INFO] Ctx-ablation dir not found: {ablation_dir}", file=sys.stderr)

    if not records:
        print("ERROR: No data found in either directory.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)

    # Deduplicate: for identical (step, ctx, dataset) keep the most recent run
    df = (
        df.sort_values("timestamp", ascending=True)
        .drop_duplicates(subset=["step", "ctx", "dataset"], keep="last")
        .sort_values(["ctx", "dataset", "step"])
        .reset_index(drop=True)
    )

    print(f"\nFinal dataset: {len(df)} unique (step, ctx, dataset) records")
    print(df.groupby(["ctx", "step"]).size().to_frame("n_datasets").to_string())

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Render the 8 × 3 grid and save to *output_path*."""
    n_rows = len(METRICS)
    n_cols = len(DATASETS)

    # Pre-compute per-row (per-metric) y-axis limits across ALL datasets so
    # each row shares the same scale — makes cross-dataset comparison easy.
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

    for col_idx, dataset in enumerate(DATASETS):
        ds_df = df[df["dataset"] == dataset]

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax: plt.Axes = axes[row_idx][col_idx]

            for ctx in CTX_CONFIGS:
                sub = ds_df[ds_df["ctx"] == ctx].sort_values("step")
                if sub.empty:
                    continue

                steps = sub["step"].to_numpy()
                vals = sub[metric_key].to_numpy()

                # Drop NaN entries so lines don't break
                mask = ~np.isnan(vals)
                if mask.sum() < 2:
                    continue

                ax.plot(
                    steps[mask],
                    vals[mask],
                    marker=CTX_MARKERS[ctx],
                    color=CTX_COLORS[ctx],
                    linewidth=1.8,
                    markersize=6,
                    label=CTX_LABELS[ctx],
                )

            # Ideal coverage reference lines
            if metric_key == "overall_coverage_50":
                ax.axhline(
                    0.50,
                    color="black",
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.7,
                    label="_ideal 50%",
                )
            elif metric_key == "overall_coverage_80":
                ax.axhline(
                    0.80,
                    color="black",
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.7,
                    label="_ideal 80%",
                )

            # Shared y-axis limits for this metric row
            ax.set_ylim(*row_ylims[row_idx])

            ax.set_xticks(ALL_STEPS)
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

            # Column headers (dataset name) on the first row only
            if row_idx == 0:
                ax.set_title(
                    DATASET_LABELS[dataset], fontsize=12, fontweight="bold", pad=6
                )

    # Single legend in the top-right cell
    handles, labels = axes[0][-1].get_legend_handles_labels()
    # Append a proxy for the ideal-coverage reference line
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
            title="Context config",
            title_fontsize=8,
        )

    fig.suptitle(
        "Chronos-2 IOB Configs — Training Step Sweep\n"
        "(episode_context_length = 512, fair fixed-episode set)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "notes" / "chronos2" / "figures" / "step_sweep_grid.png"
        ),
        help="Output PNG path (default: notes/chronos2/figures/step_sweep_grid.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_all_data()
    make_plot(df, Path(args.output))


if __name__ == "__main__":
    main()
