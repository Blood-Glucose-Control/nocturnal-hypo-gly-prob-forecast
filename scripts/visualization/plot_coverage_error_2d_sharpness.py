#!/usr/bin/env python3
"""
2-D coverage calibration error scatter with sharpness encoded as a third
dimension.

X-axis : 0.50 − coverage_50  (50% PI coverage error)
Y-axis : 0.80 − coverage_80  (80% PI coverage error)
3rd dim: sharpness (three alternative encodings — one panel each)

Panel A — Marker size proportional to mean normalised sharpness.
           Larger = wider / less-sharp intervals.

Panel B — Marker alpha encodes sharpness (faint = sharp, opaque = wide).
           Colour = model, marker shape = dataset (same as Panel A).
           Desaturated points are sharper; fully opaque points are wider.

Panel C — Bubble + colour-map.  Colour now encodes sharpness (viridis),
           with model identity lost to keep the colourmap legible.  Best
           used when you want to read off sharpness precisely.

All panels: marker *shape* = dataset, cross-hair at (0, 0) = ideal.

Sharpness used is the mean of rank-normalised sharpness_50 and sharpness_80
so both PI levels contribute equally regardless of their different absolute
scales.

Rows with non-positive or non-finite sharpness are dropped.

Usage
-----
    python scripts/visualization/plot_coverage_error_2d_sharpness.py
    python scripts/visualization/plot_coverage_error_2d_sharpness.py \\
        --summary experiments/nocturnal_forecasting/summary.csv \\
        --out results/coverage_error_2d_sharpness.png \\
        --filter-outliers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------

MODEL_COLORS: dict[str, str] = {
    "chronos2": "#e377c2",  # pink
    "moirai": "#1f77b4",
    "toto": "#ff7f0e",
    "tide": "#2ca02c",
    "timesfm": "#d62728",
    "timegrad": "#9467bd",
    "sundial": "#8c564b",
}
_DEFAULT_COLOR = "#7f7f7f"

DATASET_MARKERS: dict[str, str] = {
    "aleppo_2017": "o",
    "brown_2019": "s",
    "lynch_2022": "^",
    "tamborlane_2008": "D",
}
_DEFAULT_MARKER = "P"

DATASET_LABELS: dict[str, str] = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
    "tamborlane_2008": "Tamborlane 2008",
}

DEFAULT_SUMMARY = "experiments/nocturnal_forecasting/summary.csv"
DEFAULT_OUT = "results/coverage_error_2d_sharpness.png"

# Marker size range for Panel A (min/max pixels²)
_MIN_S, _MAX_S = 40, 350


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument(
        "--filter-outliers",
        action="store_true",
        help="Drop sharpness values beyond Q3 + 3×IQR before plotting",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame, filter_outliers: bool) -> pd.DataFrame:
    mask = (
        df["coverage_50"].notna()
        & df["coverage_80"].notna()
        & df["sharpness_50"].notna()
        & df["sharpness_80"].notna()
        & np.isfinite(df["coverage_50"].astype(float))
        & np.isfinite(df["coverage_80"].astype(float))
        & (df["sharpness_50"].astype(float) > 0)
        & (df["sharpness_80"].astype(float) > 0)
    )
    d = df[mask].copy()

    if filter_outliers:
        for col in ("sharpness_50", "sharpness_80"):
            q1, q3 = d[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            d = d[d[col] <= q3 + 3 * iqr]
        print(f"After outlier filter: {len(d)} rows.")

    d["err_50"] = 0.50 - d["coverage_50"].astype(float)
    d["err_80"] = 0.80 - d["coverage_80"].astype(float)

    # Rank-normalise each sharpness column to [0, 1] independently, then average.
    # Rank normalisation: rank / (n-1) so 0 = sharpest, 1 = widest.
    n = len(d)
    rank50 = (rankdata(d["sharpness_50"].values, method="average") - 1) / max(n - 1, 1)
    rank80 = (rankdata(d["sharpness_80"].values, method="average") - 1) / max(n - 1, 1)
    d["sharpness_norm"] = (rank50 + rank80) / 2.0  # 0 = sharpest, 1 = widest

    # Also keep raw mean for colourmap panel
    d["sharpness_raw_mean"] = (
        d["sharpness_50"].astype(float) + d["sharpness_80"].astype(float)
    ) / 2.0

    return d


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def _crosshair(ax: plt.Axes, lim: float) -> None:
    ax.axvline(0, color="black", linestyle="--", linewidth=0.85, alpha=0.4, zorder=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.85, alpha=0.4, zorder=2)
    ax.scatter(
        [0],
        [0],
        marker="*",
        s=180,
        color="gold",
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, linestyle=":", alpha=0.35)


def _axis_labels(ax: plt.Axes) -> None:
    ax.set_xlabel(
        "50% PI coverage error\n← over-cov · 0=perfect · under-cov →",
        fontsize=9,
    )
    ax.set_ylabel(
        "80% PI coverage error\n← over-cov · 0=perfect · under-cov →",
        fontsize=9,
    )


def _dataset_legend(ax: plt.Axes, loc: str = "lower right") -> None:
    handles = [
        mlines.Line2D(
            [],
            [],
            color="#555",
            marker=mk,
            linestyle="None",
            markersize=8,
            label=DATASET_LABELS.get(ds, ds),
        )
        for ds, mk in DATASET_MARKERS.items()
    ]
    ax.legend(
        handles=handles,
        title="Dataset (marker)",
        loc=loc,
        fontsize=7.5,
        title_fontsize=8.5,
        framealpha=0.85,
    )


def _model_legend(ax: plt.Axes, loc: str = "upper left") -> None:
    handles = [
        mlines.Line2D(
            [],
            [],
            color=c,
            marker="o",
            linestyle="None",
            markersize=8,
            label=m.capitalize(),
        )
        for m, c in MODEL_COLORS.items()
    ]
    leg = ax.legend(
        handles=handles,
        title="Model (colour)",
        loc=loc,
        fontsize=7.5,
        title_fontsize=8.5,
        framealpha=0.85,
    )
    ax.add_artist(leg)


# ---------------------------------------------------------------------------
# Panel A: marker SIZE encodes sharpness
# ---------------------------------------------------------------------------


def panel_a(ax: plt.Axes, df: pd.DataFrame, lim: float) -> None:
    # Map normalised sharpness [0,1] → marker size [_MIN_S, _MAX_S]
    sizes = _MIN_S + df["sharpness_norm"] * (_MAX_S - _MIN_S)

    for (model, dataset), grp in df.groupby(["model", "dataset"]):
        color = MODEL_COLORS.get(str(model), _DEFAULT_COLOR)
        marker = DATASET_MARKERS.get(str(dataset), _DEFAULT_MARKER)
        idx = grp.index
        ax.scatter(
            grp["err_50"],
            grp["err_80"],
            s=sizes.loc[idx],
            color=color,
            marker=marker,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    _crosshair(ax, lim)
    _axis_labels(ax)
    ax.set_title(
        "A · Marker size = sharpness\n(larger = wider interval)",
        fontsize=10,
        fontweight="bold",
    )

    _model_legend(ax, loc="upper left")
    _dataset_legend(ax, loc="lower right")

    # Size legend (sharpness guide)
    guide_sizes = [_MIN_S, (_MIN_S + _MAX_S) / 2, _MAX_S]
    guide_labels = ["sharp (narrow)", "medium", "wide (dull)"]
    size_handles = [
        mlines.Line2D(
            [],
            [],
            color="#555",
            marker="o",
            linestyle="None",
            markersize=np.sqrt(s),
            label=lbl,
        )
        for s, lbl in zip(guide_sizes, guide_labels)
    ]
    ax.legend(
        handles=size_handles,
        title="Sharpness (size)",
        loc="upper right",
        fontsize=7.5,
        title_fontsize=8.5,
        framealpha=0.85,
    )
    # Re-add dataset legend since it was replaced
    _dataset_legend(ax, loc="lower right")
    # Re-add model legend too
    _model_legend(ax, loc="upper left")


# ---------------------------------------------------------------------------
# Panel B: marker ALPHA encodes sharpness, shape = dataset, colour = model
# ---------------------------------------------------------------------------


def panel_b(ax: plt.Axes, df: pd.DataFrame, lim: float) -> None:
    """Alpha range: 0.15 (sharpest) to 0.95 (widest)."""
    alpha_min, alpha_max = 0.15, 0.95

    for (model, dataset), grp in df.groupby(["model", "dataset"]):
        color = MODEL_COLORS.get(str(model), _DEFAULT_COLOR)
        marker = DATASET_MARKERS.get(str(dataset), _DEFAULT_MARKER)
        for _, row in grp.iterrows():
            alpha = alpha_min + row["sharpness_norm"] * (alpha_max - alpha_min)
            ax.scatter(
                row["err_50"],
                row["err_80"],
                s=65,
                color=color,
                marker=marker,
                alpha=float(alpha),
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

    _crosshair(ax, lim)
    _axis_labels(ax)
    ax.set_title(
        "B · Marker alpha = sharpness\n(faint = sharp, opaque = wide)",
        fontsize=10,
        fontweight="bold",
    )

    _model_legend(ax, loc="upper left")
    # Alpha guide
    guide_alphas = [alpha_min, (alpha_min + alpha_max) / 2, alpha_max]
    guide_labels = ["sharp", "medium", "wide"]
    alpha_handles = [
        mlines.Line2D(
            [],
            [],
            color="#333",
            marker="o",
            linestyle="None",
            markersize=8,
            alpha=a,
            label=lbl,
        )
        for a, lbl in zip(guide_alphas, guide_labels)
    ]
    leg = ax.legend(
        handles=alpha_handles,
        title="Sharpness (alpha)",
        loc="upper right",
        fontsize=7.5,
        title_fontsize=8.5,
        framealpha=0.85,
    )
    ax.add_artist(leg)
    _dataset_legend(ax, loc="lower right")


# ---------------------------------------------------------------------------
# Panel C: colourmap encodes sharpness (model colour sacrificed)
# ---------------------------------------------------------------------------


def panel_c(ax: plt.Axes, df: pd.DataFrame, lim: float, fig: plt.Figure) -> None:
    cmap = plt.get_cmap(
        "viridis_r"
    )  # reversed: low (sharp) = yellow, high (wide) = purple
    norm = mcolors.Normalize(
        vmin=df["sharpness_raw_mean"].min(),
        vmax=df["sharpness_raw_mean"].max(),
    )

    for (model, dataset), grp in df.groupby(["model", "dataset"]):
        marker = DATASET_MARKERS.get(str(dataset), _DEFAULT_MARKER)
        colors = [cmap(norm(v)) for v in grp["sharpness_raw_mean"]]
        ax.scatter(
            grp["err_50"],
            grp["err_80"],
            s=75,
            c=colors,
            marker=marker,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )

    _crosshair(ax, lim)
    _axis_labels(ax)
    ax.set_title(
        "C · Colourmap = mean sharpness (mmol L⁻¹)\n(yellow = sharp, purple = wide)",
        fontsize=10,
        fontweight="bold",
    )
    _dataset_legend(ax, loc="lower right")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("mean PI width (mmol L⁻¹)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.summary)
    print(f"Loaded {len(raw)} rows from {args.summary}")
    raw = raw.sort_values("timestamp").drop_duplicates(
        subset=["model", "dataset", "mode", "checkpoint"], keep="last"
    )
    print(f"After dedup (unique checkpoint × dataset): {len(raw)} rows.")

    df = prepare(raw, args.filter_outliers)
    print(
        f"Using {len(df)} rows with finite coverage & positive sharpness at both levels."
    )

    lim = max(df["err_50"].abs().max(), df["err_80"].abs().max()) * 1.08

    fig, axs = plt.subplots(1, 3, figsize=(19, 6.5))
    fig.suptitle(
        "Coverage calibration error (50% vs 80%) with sharpness as 3rd dimension\n"
        "★ = ideal origin  ·  colour = model (panels A & B)  ·  marker = dataset",
        fontsize=12,
        y=1.01,
    )

    panel_a(axs[0], df, lim)
    panel_b(axs[1], df, lim)
    panel_c(axs[2], df, lim, fig)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
