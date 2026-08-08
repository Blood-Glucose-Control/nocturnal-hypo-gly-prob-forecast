#!/usr/bin/env python3
"""Plot Chronos-2 long-run (100k step) IOB checkpoint sweep results.

Reads results from:
  experiments/nocturnal_forecasting_long_run_step_sweep/512ctx_96fh/chronos2/
    16_bg_iob_ia_high_lr_100k_step_5000_aleppo_2017/results_summary.json ...
    17_bg_iob_ia_high_lr_100k_ensemble_step_5000_aleppo_2017/results_summary.json ...

Zero-shot (step 0) results are re-used from:
  experiments/nocturnal_forecasting_ctx_ablation/

Produces an (8 metrics) × (3 datasets) grid with two lines per sub-plot:
  - cfg16: non-ensemble  (LR=5e-5, ctx=512, 100k steps)
  - cfg17: ensemble      (same config, ensemble=True)

X-axis: training steps 0 → 100 000.

Output:
    results/figures/step_sweep_grid_long_run.png

Usage:
    python scripts/visualization/plot_long_run_step_sweep.py
    python scripts/visualization/plot_long_run_step_sweep.py --output path/to/out.png
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

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022"]
DATASET_LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
}

# The two configs being compared.
CFG_META = {
    "16_bg_iob_ia_high_lr_100k": {
        "label": "cfg16 — no ensemble",
        "color": "#1f77b4",
        "marker": "o",
    },
    "17_bg_iob_ia_high_lr_100k_ensemble": {
        "label": "cfg17 — ensemble",
        "color": "#ff7f0e",
        "marker": "s",
    },
}

# Every 10k steps (matching eval script; 5k checkpoints can be added later)
ALL_STEPS = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
STEP_TICK_STRIDE = 10000  # one tick every 10k
LONG_RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "nocturnal_forecasting_long_run_step_sweep"
    / "512ctx_96fh"
    / "chronos2"
)
ZS_DIR = PROJECT_ROOT / "experiments" / "nocturnal_forecasting_ctx_ablation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_step(checkpoint: str | None, default_final: int = 100000) -> int:
    """Return training step from a checkpoint path.

    - None → 0  (zero-shot)
    - path containing ``step_N`` → N
    - path without ``step_N`` → default_final
    """
    if checkpoint is None:
        return 0
    m = re.search(r"step_(\d+)", checkpoint)
    if m:
        return int(m.group(1))
    return default_final


def _load_from_dir(
    base: Path,
    cfg_stem_filter: str | None = None,
    step_filter: int | None = None,
) -> list[dict[str, Any]]:
    """Recursively find results_summary.json files and return record list.

    Args:
        base: Root directory to scan.
        cfg_stem_filter: If given, only accept records whose output directory
            name starts with this stem (e.g. "16_bg_iob_ia_high_lr_100k").
        step_filter: If given, only accept records whose inferred step equals
            this value.

    Returns:
        List of dicts with keys: step, cfg, dataset, timestamp, <metric_keys>.
    """
    records: list[dict[str, Any]] = []

    for p in sorted(base.rglob("results_summary.json")):
        if cfg_stem_filter is not None:
            # The output directory name is expected to start with the cfg stem
            if not p.parent.name.startswith(cfg_stem_filter):
                continue

        try:
            with open(p) as f:
                d: dict[str, Any] = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not read {p}: {exc}", file=sys.stderr)
            continue

        dataset = d.get("dataset")
        if dataset not in DATASETS:
            continue

        step = _extract_step(d.get("checkpoint"))
        if step_filter is not None and step != step_filter:
            continue

        # Infer which config produced this record from the directory name.
        cfg = None
        for stem in CFG_META:
            if stem in str(p):
                cfg = stem
                break
        # For zero-shot records (from ctx-ablation dir) the directory won't
        # contain a cfg stem — determine from covariate_cols + context_length.
        if cfg is None and step == 0:
            cfg_data = d.get("config") or {}
            ctx = cfg_data.get("context_length")
            cov_cols = cfg_data.get("covariate_cols") or []
            # Both cfg16 and cfg17 use ctx=512 + [iob, insulin_availability].
            # Zero-shot is model-independent, so assign to a shared key.
            if ctx == 512 and "iob" in cov_cols:
                cfg = "zero_shot"
            else:
                continue
        elif cfg is None:
            continue

        timestamp = d.get("timestamp", "")
        row: dict[str, Any] = {
            "step": step,
            "cfg": cfg,
            "dataset": dataset,
            "timestamp": timestamp,
        }
        for key, _ in METRICS:
            val = d.get(key)
            row[key] = float(val) if val is not None else np.nan

        records.append(row)

    return records


def load_all_data() -> pd.DataFrame:
    """Load and merge data from long-run step-sweep and zero-shot dirs."""
    records: list[dict[str, Any]] = []

    if LONG_RUN_DIR.exists():
        lr_records = _load_from_dir(LONG_RUN_DIR)
        records.extend(lr_records)
        print(f"Loaded {len(lr_records)} records from long-run step-sweep dir.")
    else:
        print(
            f"[INFO] Long-run step-sweep dir not found: {LONG_RUN_DIR}", file=sys.stderr
        )

    # Step 0 (zero-shot) from ctx-ablation experiment — ctx=512 rows only
    if ZS_DIR.exists():
        zs_records = _load_from_dir(ZS_DIR, step_filter=0)
        # Keep only ctx=512 zero-shot
        zs_records_512 = []
        for r in zs_records:
            if r["cfg"] == "zero_shot":
                zs_records_512.append(r)
        records.extend(zs_records_512)
        print(f"Loaded {len(zs_records_512)} zero-shot records from ctx-ablation dir.")
    else:
        print(f"[INFO] Ctx-ablation dir not found: {ZS_DIR}", file=sys.stderr)

    if not records:
        print("ERROR: No data found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)

    # Expand zero-shot rows so they appear for both cfg16 and cfg17 lines.
    # Zero-shot performance is model-independent (pre-trained weights only).
    zs_rows = df[df["cfg"] == "zero_shot"].copy()
    expanded = []
    for stem in CFG_META:
        rows = zs_rows.copy()
        rows["cfg"] = stem
        expanded.append(rows)
    df = pd.concat([df[df["cfg"] != "zero_shot"]] + expanded, ignore_index=True)

    # Deduplicate: for identical (step, cfg, dataset) keep most recent run
    df = (
        df.sort_values("timestamp", ascending=True)
        .drop_duplicates(subset=["step", "cfg", "dataset"], keep="last")
        .sort_values(["cfg", "dataset", "step"])
        .reset_index(drop=True)
    )

    print(f"\nFinal dataset: {len(df)} unique (step, cfg, dataset) records")
    print(df.groupby(["cfg", "step"]).size().to_frame("n_datasets").to_string())

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Render the 8 × 3 grid and save."""
    n_rows = len(METRICS)
    n_cols = len(DATASETS)

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

    # x-axis ticks: 0, 10k, 20k, …, 100k
    xticks = [s for s in ALL_STEPS if s % STEP_TICK_STRIDE == 0]

    for col_idx, dataset in enumerate(DATASETS):
        ds_df = df[df["dataset"] == dataset]

        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax: plt.Axes = axes[row_idx][col_idx]

            for stem, meta in CFG_META.items():
                sub = ds_df[ds_df["cfg"] == stem].sort_values("step")
                if sub.empty:
                    continue

                steps = sub["step"].to_numpy()
                vals = sub[metric_key].to_numpy()
                mask = ~np.isnan(vals)
                if mask.sum() < 2:
                    continue

                ax.plot(
                    steps[mask],
                    vals[mask],
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
                    DATASET_LABELS[dataset],
                    fontsize=12,
                    fontweight="bold",
                    pad=6,
                )

    # Legend in top-right cell
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
            title="Config",
            title_fontsize=8,
        )

    fig.suptitle(
        "Chronos-2 IOB Long Run — Training Step Sweep (0 → 100k)\n"
        "(LR=5e-5, ctx=512, iob+insulin_availability — fixed episode set)",
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
            PROJECT_ROOT / "results" / "figures" / "step_sweep_grid_long_run.png"
        ),
        help="Output PNG path (default: results/figures/step_sweep_grid_long_run.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_all_data()
    make_plot(df, Path(args.output))


if __name__ == "__main__":
    main()
