#!/usr/bin/env python3
"""Context-length ablation — publication figure (NeurIPS single-column format).

6 metrics × 3 datasets grid.  Each row shares a y-axis scale; each column
shares an x-axis (context length, categorical spacing).  One line per model.

Usage:
    python scripts/visualization/plot_ctx_ablation_pub.py
    python scripts/visualization/plot_ctx_ablation_pub.py --output ctx_ablation_v2

Output:
    results/figures/ctx_ablation_pub.pdf   (vector — for submission)
    results/figures/ctx_ablation_pub.png   (raster — for preview)
"""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────────────
# All visual and data parameters live here.  Edit freely; do not touch the
# plotting logic below the "END CONFIG" marker.
# ────────────────────────────────────────────────────────────────────────────

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -- Paths -------------------------------------------------------------------
DATA_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "nocturnal_forecasting_ctx_ablation"
    / "visualization_data.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_STEM = "ctx_ablation_pub"  # filename without extension
OUTPUT_FORMATS: list[str] = ["pdf", "png"]
OUTPUT_DPI = 300  # for PNG raster; PDF is lossless

# -- Page geometry -----------------------------------------------------------
# NeurIPS single-column textwidth = 5.5 in.
# Keep height well under ~9 in to leave room for section header + caption.
FIGURE_WIDTH_IN = 5.5
FIGURE_HEIGHT_IN = 7.5

# -- Typography --------------------------------------------------------------
# Set USE_LATEX = True only if a full LaTeX distribution is installed; it
# enables proper math fonts but is slower and can fail silently.
USE_LATEX = False
FONT_FAMILY = "serif"  # "serif" for paper; "sans-serif" for slides
FONT_SIZE_BASE = 7  # default / body text
FONT_SIZE_COL_TITLE = 7  # dataset column headers
FONT_SIZE_AXIS_LABEL = 7  # y-axis metric label and x-axis label
FONT_SIZE_TICK = 6  # tick labels
FONT_SIZE_LEGEND = 7  # legend entries

# -- Subplot spacing ---------------------------------------------------------
HSPACE = 0.20  # vertical gap between rows (subplot-height fraction)
WSPACE = 0.12  # horizontal gap between columns
LEGEND_SPACE_BOTTOM = 0.07  # bottom figure margin reserved for the legend

# -- Lines & markers ---------------------------------------------------------
LINE_WIDTH = 1.2  # pt
MARKER_SIZE = 4.0  # pt
MARKER_EDGE_WIDTH = 0.6  # pt
ALPHA = 1.0  # line opacity (0–1)

# -- X-axis ------------------------------------------------------------------
# Values appear with equal pixel spacing (categorical, not log).
CONTEXT_LENGTHS: list[int] = [64, 128, 256, 512]

# -- Y-axis ------------------------------------------------------------------
# Fractional headroom added above and below the data range of each shared row.
Y_PADDING_FRAC = 0.10

# -- Reference lines ---------------------------------------------------------
# Horizontal reference lines drawn on specific metric rows.
# {csv_column_name: y_value} — set to {} to disable all.
REFERENCE_LINES: dict[str, float] = {
    "coverage_50": 0.5,
    "coverage_80": 0.8,
}
REFERENCE_LINE_COLOR = "#999999"  # grey
REFERENCE_LINE_LINEWIDTH = 0.8
REFERENCE_LINE_LINESTYLE = "--"
REFERENCE_LINE_ZORDER = 1  # draw behind data lines

# -- Metrics (rows, top to bottom) -------------------------------------------
# (csv_column_name,  y-axis label displayed on the leftmost subplot)
METRICS: list[tuple[str, str]] = [
    ("rmse", "RMSE (mmol/L)"),
    ("wql", "WQL"),
    ("coverage_50", r"$\mathrm{Cov}_{0.5}$"),
    ("sharpness_50", r"$\mathrm{Sharp}_{0.5}$"),
    ("coverage_80", r"$\mathrm{Cov}_{0.8}$"),
    ("sharpness_80", r"$\mathrm{Sharp}_{0.8}$"),
]

# -- Datasets (columns, left to right) ---------------------------------------
# (csv value in the 'dataset' column,  display label shown at column top)
DATASETS: list[tuple[str, str]] = [
    ("aleppo_2017", "Replace-BG"),
    ("brown_2019", "DCLP3"),
    ("lynch_2022", "IOBP2"),
]

# -- Model styles ------------------------------------------------------------
# (csv value in the 'model' column, legend label, hex colour, marker, linestyle)
MODEL_STYLES: list[tuple[str, str, str, str, str]] = [
    ("chronos2", "Chronos-2", "#1f77b4", "o", "-"),
    # ("deepar",   "DeepAR",    "#d62728", "s", "-"),
    ("patchtst", "PatchTST", "#2ca02c", "^", "-"),
    ("tft", "TFT", "#ff7f0e", "D", "-"),
]

# -- Legend ------------------------------------------------------------------
LEGEND_NCOL = 4  # entries per row in legend
LEGEND_LOC = "lower center"
LEGEND_BBOX_TO_ANCHOR = (0.5, 0.0)  # figure-fraction coordinates
LEGEND_FRAMEON = False

# ── END CONFIG ──────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_rcparams() -> None:
    """Push all config-driven values into matplotlib's global rcParams."""
    if USE_LATEX:
        matplotlib.rcParams.update(
            {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}
        )
    matplotlib.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_COL_TITLE,
            "axes.labelsize": FONT_SIZE_AXIS_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    ctx_set = set(CONTEXT_LENGTHS)
    dataset_set = {d for d, _ in DATASETS}
    df = df[df["context_length"].isin(ctx_set) & df["dataset"].isin(dataset_set)].copy()
    # Safety dedup: one row per (model, dataset, context_length)
    df = df.drop_duplicates(subset=["model", "dataset", "context_length"], keep="first")
    return df


def _legend_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=label,
        )
        for _, label, color, marker, ls in MODEL_STYLES
    ]


def _set_ylim(ax: plt.Axes, values: list[float]) -> None:
    """Set y-limits with symmetric padding around the data range."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    pad = max((hi - lo) * Y_PADDING_FRAC, abs(hi) * 0.01 + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_stem: str | None = None) -> None:
    _apply_rcparams()
    df = _load_data()

    n_rows = len(METRICS)
    n_cols = len(DATASETS)
    x_pos = list(range(len(CONTEXT_LENGTHS)))  # [0, 1, 2, 3] — equal spacing

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharey="row",
        sharex=True,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
    )

    # Collect all y-values per row to compute shared y-limits after plotting.
    # sharey='row' links the axes limits, so setting them on axes[row, 0]
    # propagates across the row.
    row_yvals: list[list[float]] = [[] for _ in range(n_rows)]

    for row_i, (metric_col, metric_label) in enumerate(METRICS):
        for col_j, (dataset_col, dataset_label) in enumerate(DATASETS):
            ax: plt.Axes = axes[row_i, col_j]
            sub = df[df["dataset"] == dataset_col]

            for model_col, _label, color, marker, ls in MODEL_STYLES:
                msub = (
                    sub[sub["model"] == model_col]
                    .loc[sub["context_length"].isin(CONTEXT_LENGTHS)]
                    .sort_values("context_length")
                )
                if msub.empty or metric_col not in msub.columns:
                    continue

                xs = [x_pos[CONTEXT_LENGTHS.index(c)] for c in msub["context_length"]]
                ys = msub[metric_col].tolist()

                ax.plot(
                    xs,
                    ys,
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    alpha=ALPHA,
                    clip_on=True,
                )
                row_yvals[row_i].extend(v for v in ys if np.isfinite(v))

            # Reference line (e.g. nominal coverage target)
            if metric_col in REFERENCE_LINES:
                ref_y = REFERENCE_LINES[metric_col]
                ax.axhline(
                    ref_y,
                    color=REFERENCE_LINE_COLOR,
                    linewidth=REFERENCE_LINE_LINEWIDTH,
                    linestyle=REFERENCE_LINE_LINESTYLE,
                    zorder=REFERENCE_LINE_ZORDER,
                )
                row_yvals[row_i].append(ref_y)

            # Column title — top row only
            if row_i == 0:
                ax.set_title(dataset_label, fontsize=FONT_SIZE_COL_TITLE, pad=4)

            # Y-axis label — leftmost column only
            ax.set_ylabel(
                metric_label if col_j == 0 else "",
                fontsize=FONT_SIZE_AXIS_LABEL,
                labelpad=3,
            )

    # Apply shared y-limits per row (setting on col 0 propagates via sharey)
    # Also apply 2-decimal-place formatter to every y-axis for consistent spacing.
    for row_i in range(n_rows):
        _set_ylim(axes[row_i, 0], row_yvals[row_i])
        for col_j in range(n_cols):
            axes[row_i, col_j].yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.2f")
            )

    # X-axis ticks & labels — set on each column's bottom axes
    # (sharex links all rows within a column; labels appear on the bottom row only)
    for col_j in range(n_cols):
        axes[-1, col_j].set_xticks(x_pos)
        axes[-1, col_j].set_xticklabels([str(c) for c in CONTEXT_LENGTHS])
        axes[-1, col_j].set_xlabel(
            "Context length (L)", fontsize=FONT_SIZE_AXIS_LABEL, labelpad=3
        )

    # Legend below the full grid
    fig.legend(
        handles=_legend_handles(),
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        ncol=LEGEND_NCOL,
        frameon=LEGEND_FRAMEON,
        fontsize=FONT_SIZE_LEGEND,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=LEGEND_SPACE_BOTTOM, hspace=HSPACE, wspace=WSPACE)

    # Save
    stem = output_stem or OUTPUT_STEM
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        out = OUTPUT_DIR / f"{stem}.{fmt}"
        fig.savefig(out, format=fmt, dpi=OUTPUT_DPI, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate context-length ablation publication figure."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename stem (no extension). Overrides OUTPUT_STEM.",
    )
    args = parser.parse_args()
    main(output_stem=args.output)
