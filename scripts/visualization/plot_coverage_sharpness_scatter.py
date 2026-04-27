#!/usr/bin/env python3
"""
Scatter: coverage calibration error vs. sharpness.

Two side-by-side panels – one for the 50 % prediction interval (PI) and one
for the 80 % PI:

    X-axis : nominal_level − empirical_coverage
               = 0  → perfectly calibrated
               > 0  → under-coverage  (intervals too tight)
               < 0  → over-coverage   (intervals too wide)

    Y-axis : mean PI width  (sharpness, mmol L⁻¹)
               lower = sharper / tighter intervals

    Colour : model type
    Marker : dataset

"Best" forecasters cluster near the bottom-centre of each panel (well-
calibrated *and* sharp).  Rows with invalid or non-positive sharpness (e.g.
degenerate quantile inversions) are dropped automatically.

Usage
-----
    python scripts/visualization/plot_coverage_sharpness_scatter.py
    python scripts/visualization/plot_coverage_sharpness_scatter.py \\
        --summary experiments/nocturnal_forecasting/summary.csv \\
        --out results/coverage_sharpness_scatter.png
    python scripts/visualization/plot_coverage_sharpness_scatter.py \\
        --filter-outliers          # trims extreme sharpness outliers (3×IQR)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------

MODEL_COLORS: dict[str, str] = {
    "chronos2": "#e377c2",  # pink
    "moirai": "#1f77b4",  # blue
    "toto": "#ff7f0e",  # orange
    "tide": "#2ca02c",  # green
    "timesfm": "#d62728",  # red
    "timegrad": "#9467bd",  # purple
    "sundial": "#8c564b",  # brown
}
_DEFAULT_COLOR = "#7f7f7f"

DATASET_MARKERS: dict[str, str] = {
    "aleppo_2017": "o",  # circle
    "brown_2019": "s",  # square
    "lynch_2022": "^",  # triangle-up
    "tamborlane_2008": "D",  # diamond
}
_DEFAULT_MARKER = "P"

DATASET_LABELS: dict[str, str] = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
    "tamborlane_2008": "Tamborlane 2008",
}

_LEVELS = [
    {"level": 50, "cov_col": "coverage_50", "shp_col": "sharpness_50", "nominal": 0.50},
    {"level": 80, "cov_col": "coverage_80", "shp_col": "sharpness_80", "nominal": 0.80},
]

DEFAULT_SUMMARY = "experiments/nocturnal_forecasting/summary.csv"
DEFAULT_BEST_CSV = "experiments/nocturnal_forecasting/best_by_model_dataset.csv"
DEFAULT_OUT = "results/coverage_sharpness_scatter.png"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--summary", default=DEFAULT_SUMMARY, help="Path to summary.csv  [%(default)s]"
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path  [%(default)s]")
    p.add_argument(
        "--filter-outliers",
        action="store_true",
        help="Drop rows where sharpness > Q3 + 3×IQR (per panel)",
    )
    p.add_argument(
        "--best-csv",
        default=DEFAULT_BEST_CSV,
        help="Path to best_by_model_dataset.csv  [%(default)s]",
    )
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="Show every run, not just the best per model×dataset",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_valid(df: pd.DataFrame, cov_col: str, shp_col: str) -> pd.DataFrame:
    """Keep rows with finite, positive coverage and sharpness values."""
    mask = (
        df[cov_col].notna()
        & df[shp_col].notna()
        & np.isfinite(df[cov_col].astype(float))
        & np.isfinite(df[shp_col].astype(float))
        & (df[shp_col].astype(float) > 0)
    )
    return df[mask].copy()


def iqr_trim(df: pd.DataFrame, shp_col: str, k: float = 3.0) -> pd.DataFrame:
    """Remove sharpness values beyond Q3 + k×IQR (extreme outliers only)."""
    q1, q3 = df[shp_col].quantile([0.25, 0.75])
    return df[df[shp_col] <= q3 + k * (q3 - q1)]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    cov_col: str,
    shp_col: str,
    nominal: float,
    level: int,
    filter_outliers: bool,
) -> None:
    sub = load_valid(df, cov_col, shp_col)
    if filter_outliers:
        sub = iqr_trim(sub, shp_col)

    n_dropped = len(load_valid(df, cov_col, shp_col)) - len(sub)
    if n_dropped:
        print(f"  [{level}% PI] {n_dropped} extreme sharpness outlier(s) dropped.")

    for (model, dataset), grp in sub.groupby(["model", "dataset"]):
        color = MODEL_COLORS.get(str(model), _DEFAULT_COLOR)
        marker = DATASET_MARKERS.get(str(dataset), _DEFAULT_MARKER)
        x = nominal - grp[cov_col].astype(float)
        y = grp[shp_col].astype(float)
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=65,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Perfect-calibration reference
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.55, zorder=2)
    ax.text(
        0,
        ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else sub[shp_col].max(),
        " perfect\n calibration",
        ha="left",
        va="top",
        fontsize=7.5,
        color="black",
        alpha=0.55,
    )

    # Ideal-zone annotation (low-centre)
    ax.annotate(
        "★ ideal\n  zone",
        xy=(0, sub[shp_col].quantile(0.10)),
        xytext=(0, sub[shp_col].quantile(0.10)),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#444",
        style="italic",
    )

    ax.set_title(f"{level}% prediction interval", fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "Coverage error    (nominal − empirical)\n"
        "← over-coverage · 0 = perfect · under-coverage →",
        fontsize=9.5,
    )
    ax.set_ylabel(
        "PI width / sharpness (mmol L⁻¹)\nlower = tighter intervals", fontsize=9.5
    )
    ax.grid(True, linestyle=":", alpha=0.35)


def _add_legends(fig: plt.Figure) -> None:
    model_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=8,
            label=model.capitalize(),
        )
        for model, color in MODEL_COLORS.items()
    ]
    dataset_handles = [
        mlines.Line2D(
            [],
            [],
            color="#333",
            marker=marker,
            linestyle="None",
            markersize=8,
            label=DATASET_LABELS.get(ds, ds),
        )
        for ds, marker in DATASET_MARKERS.items()
    ]

    leg1 = fig.legend(
        handles=model_handles,
        title="Model  (colour)",
        loc="lower left",
        bbox_to_anchor=(0.01, -0.02),
        ncol=len(MODEL_COLORS),
        fontsize=8.5,
        title_fontsize=9.5,
        framealpha=0.85,
    )
    fig.legend(
        handles=dataset_handles,
        title="Dataset  (marker)",
        loc="lower right",
        bbox_to_anchor=(0.99, -0.02),
        ncol=len(DATASET_MARKERS),
        fontsize=8.5,
        title_fontsize=9.5,
        framealpha=0.85,
    )
    fig.add_artist(leg1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    print(f"Loaded {len(df)} rows from {args.summary}")
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["model", "dataset", "mode", "checkpoint"], keep="last"
    )
    print(f"After dedup (unique checkpoint × dataset): {len(df)} rows.")

    if not args.all_runs:
        best = pd.read_csv(args.best_csv)
        df = df[df["run_id"].isin(best["run_id"])].copy()
        print(f"After filtering to best run per model×dataset: {len(df)} rows.")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Coverage calibration error vs. interval sharpness\n"
        "Best forecasters: bottom-centre  (well-calibrated and sharp)",
        fontsize=13.5,
        y=1.02,
    )

    for ax, cfg in zip(axs, _LEVELS):
        _scatter_panel(
            ax,
            df,
            cov_col=cfg["cov_col"],
            shp_col=cfg["shp_col"],
            nominal=cfg["nominal"],
            level=cfg["level"],
            filter_outliers=args.filter_outliers,
        )

    _add_legends(fig)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
