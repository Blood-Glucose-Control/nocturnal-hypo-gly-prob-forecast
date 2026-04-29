#!/usr/bin/env python3
"""Context-length ablation plot for AutoGluon baseline models.

Reads results from two directories:
  - experiments/nocturnal_forecasting_ctx_ablation/  (64/128/256/512 ctx)
  - experiments/nocturnal_forecasting/768ctx_96fh/   (768 ctx; different config family)

Produces an (8 metrics) × (4 datasets) grid matching the step-sweep layout:
  aleppo_2017 | brown_2019 | lynch_2022 | tamborlane_2008

Rows = metrics, columns = datasets.
Lines = one per model config (DeepAR BG, PatchTST BG, TFT BG, TFT IOB).
X-axis = context length (64 → 512 solid; 512 → 768 dotted with ★ marker).

Y-axis scale is shared across the entire row so cross-dataset comparisons
are not distorted by differing scales.

The 768-ctx points come from a dedicated long-context config and are connected
from the 512-ctx endpoint with a dotted segment to visually distinguish them
from the winner-config ablation family.

Output:
    results/figures/ctx_ablation_grid.png

Usage:
    python scripts/visualization/plot_ctx_ablation.py
    python scripts/visualization/plot_ctx_ablation.py --output path/to/out.png
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

MODEL_CONFIGS = ["deepar", "patchtst", "tft_bg", "tft_iob"]
MODEL_LABELS = {
    "deepar": "DeepAR (BG)",
    "patchtst": "PatchTST (BG)",
    "tft_bg": "TFT (BG)",
    "tft_iob": "TFT (IOB)",
}
MODEL_COLORS = {
    "deepar": "#1f77b4",
    "patchtst": "#ff7f0e",
    "tft_bg": "#2ca02c",
    "tft_iob": "#d62728",
}
MODEL_MARKERS = {
    "deepar": "o",
    "patchtst": "s",
    "tft_bg": "^",
    "tft_iob": "D",
}

DATASETS_ALL = ["aleppo_2017", "brown_2019", "lynch_2022", "tamborlane_2008"]
DATASETS_IOB = ["aleppo_2017", "brown_2019", "lynch_2022"]
MODEL_DATASETS = {
    "deepar": DATASETS_ALL,
    "patchtst": DATASETS_ALL,
    "tft_bg": DATASETS_ALL,
    "tft_iob": DATASETS_IOB,
}

DATASET_LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
    "tamborlane_2008": "Tamborlane 2008",
}

# Context lengths in the winner-config ablation family
CTX_ABLATION = [64, 128, 256, 512]

# Winner config checkpoint stems (prefix match) per model config key
WINNER_STEMS = {
    "deepar": "10_low_lr",
    "patchtst": "09_high_lr",
    "tft_bg": "01_bg_wide",
    "tft_iob": "11_iob_high_lr",
}

# 768-ctx config stems per model config key (multiple configs → best WQL kept)
STEMS_768CTX: dict[str, set[str]] = {
    "deepar": {"02_long_ctx", "11_big"},
    "patchtst": {"04_long_ctx"},
    "tft_bg": {"02_bg_long_ctx"},
    "tft_iob": {"08_iob_long_ctx"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_checkpoint_stem(checkpoint: str | None) -> str:
    """Extract the config stem from a checkpoint path."""
    if not checkpoint:
        return ""
    # Pattern: artifacts/<model>/<date>_RID<rid>_<stem>  (stem may contain /)
    m = re.search(r"artifacts/[^/]+/[^_]+_\d{4}_RID\d+_[^_]+_\d+_(.+)$", checkpoint)
    if m:
        return m.group(1)
    return Path(checkpoint).name


def _model_config_key(model: str, cov_cols: list | None) -> str | None:
    """Map (autogluon model name, covariate_cols) → MODEL_CONFIGS key."""
    cov = set(cov_cols or [])
    if model == "deepar":
        return "deepar"
    if model == "patchtst":
        return "patchtst"
    if model == "tft":
        return "tft_iob" if "iob" in cov else "tft_bg"
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ablation_data() -> pd.DataFrame:
    """Load 64/128/256/512-ctx results from the ctx-ablation experiment dir.

    Filters to only the winner-config stems per model.
    """
    ablation_dir = PROJECT_ROOT / "experiments" / "nocturnal_forecasting_ctx_ablation"
    records: list[dict[str, Any]] = []

    for p in sorted(ablation_dir.rglob("results_summary.json")):
        try:
            d: dict[str, Any] = json.load(open(p))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {p}: {exc}", file=sys.stderr)
            continue

        model = d.get("model")
        cfg = d.get("config") or {}
        ctx = cfg.get("context_length")
        cov_cols = cfg.get("covariate_cols")
        dataset = d.get("dataset")
        checkpoint = d.get("checkpoint", "")
        timestamp = d.get("timestamp", "")

        if ctx not in CTX_ABLATION:
            continue

        cfg_key = _model_config_key(model, cov_cols)
        if cfg_key is None:
            continue

        stem = _get_checkpoint_stem(checkpoint)
        winner_stem = WINNER_STEMS.get(cfg_key, "")
        if not stem.startswith(winner_stem):
            continue  # skip non-winner configs (e.g. other chronos2 runs)

        valid_datasets = MODEL_DATASETS.get(cfg_key, [])
        if dataset not in valid_datasets:
            continue

        row: dict[str, Any] = {
            "model_config": cfg_key,
            "ctx": ctx,
            "dataset": dataset,
            "timestamp": timestamp,
            "from_768cfg": False,
        }
        for key, _ in METRICS:
            val = d.get(key)
            row[key] = float(val) if val is not None else np.nan

        records.append(row)

    df = pd.DataFrame(records) if records else pd.DataFrame()
    print(f"Loaded {len(df)} ablation records (64/128/256/512 ctx).")
    return df


def load_768ctx_data() -> pd.DataFrame:
    """Load 768-ctx results from experiments/nocturnal_forecasting/768ctx_96fh/.

    Selects the best-WQL config per (model_config, dataset).
    Overrides context_length to 768 — the results_summary.json stored 512 due
    to a pre-fix output-dir naming bug; the actual models were trained at 768.
    """
    dir_768 = PROJECT_ROOT / "experiments" / "nocturnal_forecasting" / "768ctx_96fh"
    if not dir_768.exists():
        print(f"[INFO] 768ctx dir not found: {dir_768}", file=sys.stderr)
        return pd.DataFrame()

    candidates: list[dict[str, Any]] = []

    for p in sorted(dir_768.rglob("results_summary.json")):
        try:
            d: dict[str, Any] = json.load(open(p))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {p}: {exc}", file=sys.stderr)
            continue

        model = d.get("model")
        cfg = d.get("config") or {}
        cov_cols = cfg.get("covariate_cols")
        dataset = d.get("dataset")
        checkpoint = d.get("checkpoint", "")
        timestamp = d.get("timestamp", "")
        wql = d.get("overall_wql")

        if wql is None:
            continue

        cfg_key = _model_config_key(model, cov_cols)
        if cfg_key is None:
            continue

        stem = _get_checkpoint_stem(checkpoint)
        valid_stems = STEMS_768CTX.get(cfg_key, set())
        if not any(s in stem for s in valid_stems):
            continue

        valid_datasets = MODEL_DATASETS.get(cfg_key, [])
        if dataset not in valid_datasets:
            continue

        row: dict[str, Any] = {
            "model_config": cfg_key,
            "ctx": 768,
            "dataset": dataset,
            "timestamp": timestamp,
            "from_768cfg": True,
            "_wql_sort": float(wql),
        }
        for key, _ in METRICS:
            val = d.get(key)
            row[key] = float(val) if val is not None else np.nan

        candidates.append(row)

    if not candidates:
        print("[INFO] No 768-ctx records found.")
        return pd.DataFrame()

    df = pd.DataFrame(candidates)
    # Keep best WQL per (model_config, dataset)
    df = (
        df.sort_values("_wql_sort")
        .drop_duplicates(subset=["model_config", "dataset"], keep="first")
        .drop(columns=["_wql_sort"])
        .reset_index(drop=True)
    )
    print(f"Loaded {len(df)} 768-ctx records (best config per model/dataset).")
    return df


def load_all_data() -> pd.DataFrame:
    abl_df = load_ablation_data()

    if abl_df.empty:
        print("ERROR: No data found.", file=sys.stderr)
        sys.exit(1)

    df = abl_df.copy()

    # Exclude the diverged DeepAR 256ctx run until it is retrained
    before = len(df)
    df = df[~((df["model_config"] == "deepar") & (df["ctx"] == 256))]
    if len(df) < before:
        print(
            f"[INFO] Excluded DeepAR 256ctx ({before - len(df)} rows — diverged run)."
        )

    # Deduplicate: keep latest timestamp for identical (model_config, ctx, dataset)
    df = (
        df.sort_values("timestamp", ascending=True)
        .drop_duplicates(subset=["model_config", "ctx", "dataset"], keep="last")
        .sort_values(["model_config", "dataset", "ctx"])
        .reset_index(drop=True)
    )

    print(f"\nFinal dataset: {len(df)} unique (model_config, ctx, dataset) records")
    print(df.groupby(["model_config", "ctx"]).size().to_frame("n_datasets").to_string())
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Render the 8-metric × 4-dataset grid and save.

    Columns = datasets, rows = metrics, lines = model configs.
    Y-axis scale is shared across every column in the same row.
    """
    n_rows = len(METRICS)
    n_cols = len(DATASETS_ALL)

    # Context ticks: whatever levels are present in the data
    available_ctx = sorted(df["ctx"].unique())

    # Pre-compute shared y-limits per metric row across ALL datasets and models
    row_ylims: list[tuple[float, float]] = []
    for metric_key, _ in METRICS:
        vals = df[metric_key].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            row_ylims.append((0.0, 1.0))
            continue
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.10 if hi > lo else abs(hi) * 0.10 or 0.05
        row_ylims.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for col_idx, dataset in enumerate(DATASETS_ALL):
        for row_idx, (metric_key, metric_label) in enumerate(METRICS):
            ax: plt.Axes = axes[row_idx][col_idx]

            for cfg_key in MODEL_CONFIGS:
                # TFT-IOB has no tamborlane_2008 data — skip silently
                if dataset not in MODEL_DATASETS[cfg_key]:
                    continue

                sub = df[
                    (df["model_config"] == cfg_key) & (df["dataset"] == dataset)
                ].sort_values("ctx")
                if sub.empty:
                    continue

                color = MODEL_COLORS[cfg_key]
                marker = MODEL_MARKERS[cfg_key]

                winner_rows = sub[~sub["from_768cfg"]]
                ctx768_rows = sub[sub["from_768cfg"]]

                # --- Solid line: 64–512 winner-config ablation ---
                if not winner_rows.empty:
                    x_vals = winner_rows["ctx"].to_numpy()
                    y_vals = winner_rows[metric_key].to_numpy()
                    mask = ~np.isnan(y_vals)
                    if mask.sum() >= 1:
                        ax.plot(
                            x_vals[mask],
                            y_vals[mask],
                            marker=marker,
                            color=color,
                            linewidth=1.8,
                            markersize=6,
                            linestyle="-",
                            label=MODEL_LABELS[cfg_key],
                        )

                # --- Dotted bridge: 512 → 768 (different config family) ---
                if not ctx768_rows.empty:
                    y768_val = ctx768_rows[metric_key].values
                    if len(y768_val) and not np.isnan(y768_val[0]):
                        y768 = y768_val[0]
                        pt512 = winner_rows[winner_rows["ctx"] == 512]
                        if not pt512.empty:
                            y512 = pt512[metric_key].values[0]
                            if not np.isnan(y512):
                                ax.plot(
                                    [512, 768],
                                    [y512, y768],
                                    color=color,
                                    linewidth=1.2,
                                    linestyle=":",
                                    alpha=0.8,
                                )
                        ax.plot(
                            768,
                            y768,
                            marker="*",
                            color=color,
                            markersize=9,
                            linestyle="",
                            zorder=5,
                        )

            # Ideal coverage reference lines
            if metric_key == "overall_coverage_50":
                ax.axhline(
                    0.50, color="black", linewidth=1.0, linestyle="--", alpha=0.45
                )
            elif metric_key == "overall_coverage_80":
                ax.axhline(
                    0.80, color="black", linewidth=1.0, linestyle="--", alpha=0.45
                )

            # Shared y-limits for this metric row
            ax.set_ylim(*row_ylims[row_idx])

            ax.set_xticks(available_ctx)
            ax.set_xticklabels([str(x) for x in available_ctx], rotation=40, fontsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")

            if row_idx == n_rows - 1:
                ax.set_xlabel("Context length (steps)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=9)

            # Column header (dataset name) on first row only
            if row_idx == 0:
                ax.set_title(
                    DATASET_LABELS[dataset],
                    fontsize=12,
                    fontweight="bold",
                    pad=6,
                )

    # --- Legend in the top-right cell (last column, first row) ---
    # Collect handles/labels from all axes to capture every model line
    handles: list = []
    labels: list = []
    seen: set = set()
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            high, low = axes[row_idx][col_idx].get_legend_handles_labels()
            for hi, li in zip(high, low):
                if li not in seen:
                    handles.append(hi)
                    labels.append(li)
                    seen.add(li)

    # Style guide entry
    handles.append(
        mlines.Line2D([], [], color="black", linewidth=1.0, linestyle="--", alpha=0.45)
    )
    labels.append("Ideal coverage")

    axes[0][-1].legend(
        handles,
        labels,
        loc="upper right",
        fontsize=7.5,
        framealpha=0.88,
        title="Model",
        title_fontsize=8,
    )

    fig.suptitle(
        "AutoGluon Baseline Models — Context-Length Ablation\n"
        "64–512 ctx: winner-config re-trained  (DeepAR 256ctx excluded — diverged, pending retrain)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "figures" / "ctx_ablation_grid.png"),
        help="Output PNG path (default: results/figures/ctx_ablation_grid.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_all_data()
    make_plot(df, Path(args.output))


if __name__ == "__main__":
    main()
