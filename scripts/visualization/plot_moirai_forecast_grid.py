# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
4×5 grid of Moirai probabilistic forecast episodes.

Rows    : aleppo_2017, brown_2019, lynch_2022, tamborlane_2008
Columns : 10th, 25th, 50th, 75th, 90th percentile of per-episode median RMSE
           (P10 = best quality / lowest RMSE → P90 = worst quality / highest RMSE)

Episode quality is judged by the RMSE of the median quantile forecast (q=0.5)
against the actual BG trace.  The best Moirai checkpoint for each dataset
is used.

Each panel shows:
  - Actual BG trace (black)
  - Median forecast (q=0.5, dark blue)
  - 80 % PI band  (q=0.1 – q=0.9, lightest blue filled)
  - 50 % PI band  (q=0.25 – q=0.75 via linear interp, medium blue filled)
  - 3.9 mmol/L hypoglycaemia threshold (dashed red)

Usage:
    python scripts/visualization/plot_moirai_forecast_grid.py
    python scripts/visualization/plot_moirai_forecast_grid.py --out results/my_grid.png
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# LaTeX rendering  (set USE_LATEX = True/False to override; None = auto-detect)
# ---------------------------------------------------------------------------

USE_LATEX: bool | None = None  # None = use LaTeX if the `latex` binary is found

_latex_available = shutil.which("latex") is not None
if USE_LATEX is None:
    USE_LATEX = _latex_available

if USE_LATEX:
    if not _latex_available:
        raise RuntimeError("USE_LATEX=True but `latex` binary not found on PATH.")
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

HYPO_THRESHOLD = 3.9  # mmol/L – hypoglycaemia threshold line (dashed red)
HYPER_THRESHOLD = 10.0  # mmol/L – hyperglycaemia threshold line (dashed orange)
FORECAST_HORIZON = 96
INTERVAL_MINS = 5

# y-axis range for all panels
YLIM_MIN = 0
YLIM_MAX = 22

# Best Moirai run dirs (relative to REPO_ROOT)
BEST_RUNS: dict[str, str] = {
    "aleppo_2017": "experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-19_2045_aleppo_2017_finetuned_rerun01",
    "brown_2019": "experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-19_0653_brown_2019_finetuned_rerun01",
    "lynch_2022": "experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-23_0620_lynch_2022_finetuned_rerun01",
    "tamborlane_2008": "experiments/nocturnal_forecasting/512ctx_96fh/moirai/2026-04-23_0541_tamborlane_2008_finetuned",
}

DATASET_LABELS: dict[str, str] = {
    "aleppo_2017": "Replace-BG",
    "brown_2019": "DCLP3",
    "lynch_2022": "IOBP2",
    "tamborlane_2008": "Tamborlane",
}

PERCENTILE_COLS: list[tuple[int, str, str]] = [
    (10, r"$\hat{Q}$P10", "Percentile 10"),
    (25, r"$\hat{Q}$P25", "Percentile 25"),
    (50, r"$\hat{Q}$P50", "Percentile 50"),
    (75, r"$\hat{Q}$P75", "Percentile 75"),
    (90, r"$\hat{Q}$P90", "Percentile 90"),
]

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022", "tamborlane_2008"]

BLUE_DARK = "#1565c0"
BLUE_MED = "#5e97f6"
BLUE_LIGHT = "#90caf9"
BLUE_LIGHTER = "#e3f2fd"

DEFAULT_OUT_PNG = "results/moirai_forecast_grid.png"
DEFAULT_OUT_PDF = "results/moirai_forecast_grid.pdf"

# ── Figure size ──────────────────────────────────────────────────────────────
# NeurIPS 2025: textwidth = 5.5 in (single-column, letter paper, 1.5in margins)
FIG_WIDTH_IN = 5.5  # figure width (inches)
FIG_HEIGHT_IN = 7.5  # figure height (inches)
FIG_DPI = 300  # output resolution

# ── Subplot layout (fractions of figure; tweak to fix spacing) ────────────────
SUBPLOT_LEFT = 0.01  # left edge of leftmost column (increase to give row labels room)
SUBPLOT_RIGHT = 0.99  # right edge of rightmost column
SUBPLOT_TOP = 0.92  # top edge of top row
SUBPLOT_BOTTOM = 0.09  # bottom edge of bottom row (leaves room for legend)
SUBPLOT_HSPACE = 0.20  # vertical gap between rows (fraction of row height)
SUBPLOT_WSPACE = 0.10  # horizontal gap between columns (fraction of col width)

# ── Font sizes (pt) ──────────────────────────────────────────────────────────
SUPTITLE_FONTSIZE = 9.0  # main figure title
COL_HEADER_FONTSIZE = 8.0  # P10/P25/… column headers
ROW_LABEL_FONTSIZE = 8.0  # dataset name row labels
PANEL_TITLE_FONTSIZE = 5.5  # per-panel episode/RMSE subtitle
AXIS_LABEL_FONTSIZE = 6.5  # x/y axis labels
TICK_FONTSIZE = 5.5  # tick mark labels
LEGEND_FONTSIZE = 6.5  # bottom legend

# Column header offset above the top row (points)
COL_HEADER_YOFFSET = 10

# Row label x position (axes fraction; negative = left of left axis spine)
ROW_LABEL_X = -0.42

# Legend anchor y (negative places it below the bottom of the axes grid)
LEGEND_Y = 0.00
LEGEND_NCOL = 6  # number of columns in the bottom legend

# ── Line widths & opacity ─────────────────────────────────────────────────────
ACTUAL_LW = 1.3  # actual BG trace
MEDIAN_LW = 1.4  # median forecast
THRESHOLD_LW = 0.8  # clinical threshold lines
ALPHA_80PI = 0.9  # opacity of 80 % PI band
ALPHA_50PI = 0.9  # opacity of 50 % PI band


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def interpolate_quantile(
    q_forecasts: np.ndarray, q_levels: np.ndarray, q_target: float
) -> np.ndarray:
    """
    Linearly interpolate a quantile level that is not in q_levels.

    Parameters
    ----------
    q_forecasts : shape (n_quantiles, fh)
    q_levels    : shape (n_quantiles,), assumed sorted ascending
    q_target    : scalar, must be within [q_levels[0], q_levels[-1]]

    Returns
    -------
    interpolated forecast : shape (fh,)
    """
    if q_target in q_levels:
        idx = int(np.searchsorted(q_levels, q_target))
        return q_forecasts[idx]

    idx_hi = int(np.searchsorted(q_levels, q_target))
    idx_lo = idx_hi - 1
    q_lo, q_hi = q_levels[idx_lo], q_levels[idx_hi]
    frac = (q_target - q_lo) / (q_hi - q_lo)
    return q_forecasts[idx_lo] + frac * (q_forecasts[idx_hi] - q_forecasts[idx_lo])


def load_percentile_episodes(dataset: str) -> list[dict]:
    """
    Load the best Moirai forecasts.npz for *dataset*, compute per-episode
    RMSE of the median forecast, and return one episode dict per percentile
    defined in PERCENTILE_COLS.
    """
    run_dir = REPO_ROOT / BEST_RUNS[dataset]
    npz = np.load(run_dir / "forecasts.npz")

    actuals = npz["actuals"]  # (n_eps, fh)
    q_forecasts = npz["quantile_forecasts"]  # (n_eps, 9, fh)
    q_levels = npz["quantile_levels"]  # (9,)
    episode_ids = npz["episode_ids"]  # (n_eps,)

    # Median forecast = q=0.5, index 4 in [0.1, 0.2, ..., 0.9]
    q05 = q_forecasts[:, 4, :]  # (n_eps, fh)
    rmse = np.sqrt(np.mean((actuals - q05) ** 2, axis=1))  # (n_eps,)

    episodes = []
    for pct, short_label, long_label in PERCENTILE_COLS:
        target_rmse = float(np.percentile(rmse, pct))
        idx = int(np.argmin(np.abs(rmse - target_rmse)))
        episodes.append(
            {
                "actuals": actuals[idx],  # (fh,)
                "q_forecasts": q_forecasts[idx],  # (9, fh)
                "q_levels": q_levels,
                "episode_id": str(episode_ids[idx]),
                "rmse": float(rmse[idx]),
                "pct": pct,
                "short_label": short_label,
                "long_label": long_label,
            }
        )
    return episodes


def plot_panel(
    ax: plt.Axes,
    ep: dict,
    show_ylabel: bool,
    show_xlabel: bool,
    show_ytick_labels: bool = True,
    show_xtick_labels: bool = True,
) -> None:
    """Draw one forecast panel onto ax."""
    fh = FORECAST_HORIZON
    t = np.arange(fh) * INTERVAL_MINS / 60.0  # hours

    actuals = ep["actuals"]
    q_fc = ep["q_forecasts"]
    q_levels = ep["q_levels"]

    q01 = q_fc[0]  # 0.1
    q09 = q_fc[8]  # 0.9
    q05 = q_fc[4]  # 0.5 (median)
    q025 = interpolate_quantile(q_fc, q_levels, 0.25)
    q075 = interpolate_quantile(q_fc, q_levels, 0.75)

    # 80 % PI
    ax.fill_between(t, q01, q09, color=BLUE_LIGHTER, alpha=ALPHA_80PI, label="80 % PI")
    # 50 % PI
    ax.fill_between(t, q025, q075, color=BLUE_LIGHT, alpha=ALPHA_50PI, label="50 % PI")
    # Median forecast
    ax.plot(t, q05, color=BLUE_DARK, lw=MEDIAN_LW, label="Forecast (median)", zorder=3)
    # Actual trace
    ax.plot(t, actuals, color="black", lw=ACTUAL_LW, label="Actual BG", zorder=4)
    # Clinical thresholds
    ax.axhline(
        HYPO_THRESHOLD,
        color="#d62728",
        lw=THRESHOLD_LW,
        ls="--",
        alpha=0.7,
        label="3.9 mmol/L",
    )
    ax.axhline(
        HYPER_THRESHOLD,
        color="#ff7f0e",
        lw=THRESHOLD_LW,
        ls="--",
        alpha=0.7,
        label="10 mmol/L",
    )

    min_bg = float(actuals.min())
    # ep_label = ep["episode_id"].split("::")[-1]
    ax.set_title(
        f"RMSE:{ep['rmse']:.2f} · min(BG):{min_bg:.1f}",
        fontsize=PANEL_TITLE_FONTSIZE,
        pad=2,
    )
    ax.set_xlim(0, (fh - 1) * INTERVAL_MINS / 60.0)
    ax.set_ylim(YLIM_MIN, YLIM_MAX)
    ax.tick_params(
        labelsize=TICK_FONTSIZE,
        labelbottom=show_xtick_labels,
        labelleft=show_ytick_labels,
    )

    if show_ylabel:
        ax.set_ylabel("BG (mmol/L)", fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("Hours ahead", fontsize=AXIS_LABEL_FONTSIZE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_grid(out_png: str, out_pdf: str) -> None:
    fig, axes = plt.subplots(
        nrows=4,
        ncols=5,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
    )
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        bottom=SUBPLOT_BOTTOM,
        hspace=SUBPLOT_HSPACE,
        wspace=SUBPLOT_WSPACE,
    )

    for row, dataset in enumerate(DATASETS):
        print(f"  Loading {dataset}…")
        episodes = load_percentile_episodes(dataset)

        for col, ep in enumerate(episodes):
            ax = axes[row, col]
            plot_panel(
                ax,
                ep,
                show_ylabel=(col == 0),
                show_xlabel=(row == 3),
                show_ytick_labels=(col == 0),
                show_xtick_labels=(row == 3),
            )

        axes[row, 0].annotate(
            DATASET_LABELS[dataset],
            xy=(ROW_LABEL_X, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=ROW_LABEL_FONTSIZE,
            fontweight="bold",
            rotation=90,
        )

    for col, (_, _, long_label) in enumerate(PERCENTILE_COLS):
        axes[0, col].annotate(
            long_label,
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, COL_HEADER_YOFFSET),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=COL_HEADER_FONTSIZE,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=LEGEND_NCOL,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        bbox_to_anchor=(0.5, LEGEND_Y),
    )

    fig.suptitle(
        "Moirai Probabilistic Forecasts — episode quality by RMSE percentile",
        fontsize=SUPTITLE_FONTSIZE,
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default=DEFAULT_OUT_PNG, help="Output PNG path")
    p.add_argument("--out-pdf", default=DEFAULT_OUT_PDF, help="Output PDF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_grid(out_png=args.out, out_pdf=args.out_pdf)
