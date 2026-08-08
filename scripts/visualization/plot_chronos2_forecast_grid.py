# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
4×5 grid of Chronos-2 probabilistic forecast episodes.

Rows    : aleppo_2017, brown_2019, lynch_2022, tamborlane_2008
Columns : 10th, 25th, 50th, 75th, 90th percentile of per-episode median RMSE
           (P10 = best quality / lowest RMSE → P90 = worst quality / highest RMSE)

Episode quality is judged by the RMSE of the median quantile forecast (q=0.5)
against the actual BG trace.  The best Chronos-2 checkpoint for each dataset
is used.

Each panel shows:
  - Actual BG trace (black)
  - Median forecast (q=0.5, dark pink)
  - 80 % PI band  (q=0.1 – q=0.9, lightest pink filled)
  - 50 % PI band  (q=0.25 – q=0.75 via linear interp, medium pink filled)
  - 3.9 mmol/L hypoglycaemia threshold (dashed red)

Usage:
    python scripts/visualization/plot_chronos2_forecast_grid.py
    python scripts/visualization/plot_chronos2_forecast_grid.py --out results/my_grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

HYPO_THRESHOLD = 3.9  # mmol/L
FORECAST_HORIZON = 96
INTERVAL_MINS = 5

# Best Chronos-2 run dirs (relative to REPO_ROOT)
BEST_RUNS: dict[str, str] = {
    "aleppo_2017": "experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-25_183908_aleppo_2017_finetuned",
    "brown_2019": "experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-24_192415_brown_2019_finetuned",
    "lynch_2022": "experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-24_191927_lynch_2022_finetuned",
    "tamborlane_2008": "experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-04-25_183555_tamborlane_2008_finetuned",
}

DATASET_LABELS: dict[str, str] = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
    "tamborlane_2008": "Tamborlane 2008",
}

# Pre-sampled interesting episodes (idx → episode from forecasts.npz)
# Columns: 10th / 25th / 50th / 75th / 90th percentile of per-episode median RMSE
# (P10 = lowest RMSE = best quality; P90 = highest RMSE = worst quality)
PERCENTILE_COLS: list[tuple[int, str, str]] = [
    (10, "P10", "P10  (best)"),
    (25, "P25", "P25"),
    (50, "P50", "P50  (median)"),
    (75, "P75", "P75"),
    (90, "P90", "P90  (worst)"),
]

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022", "tamborlane_2008"]

PINK_DARK = "#c5449a"
PINK_MED = "#e377c2"
PINK_LIGHT = "#f2c6e8"
PINK_LIGHTER = "#f9e5f5"

DEFAULT_OUT_PNG = "results/chronos2_forecast_grid.png"
DEFAULT_OUT_PDF = "results/chronos2_forecast_grid.pdf"


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
    Load the best Chronos-2 forecasts.npz for *dataset*, compute per-episode
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


def plot_panel(ax: plt.Axes, ep: dict, show_ylabel: bool, show_xlabel: bool) -> None:
    """Draw one forecast panel onto ax."""
    fh = FORECAST_HORIZON
    t = np.arange(fh) * INTERVAL_MINS / 60.0  # hours

    actuals = ep["actuals"]
    q_fc = ep["q_forecasts"]
    q_levels = ep["q_levels"]

    # Quantile traces
    q01 = q_fc[0]  # 0.1
    q09 = q_fc[8]  # 0.9
    q05 = q_fc[4]  # 0.5 (median)
    q025 = interpolate_quantile(q_fc, q_levels, 0.25)
    q075 = interpolate_quantile(q_fc, q_levels, 0.75)

    # 80 % PI
    ax.fill_between(t, q01, q09, color=PINK_LIGHTER, alpha=0.9, label="80 % PI")
    # 50 % PI
    ax.fill_between(t, q025, q075, color=PINK_LIGHT, alpha=0.9, label="50 % PI")
    # Median forecast
    ax.plot(t, q05, color=PINK_DARK, lw=1.4, label="Forecast (median)", zorder=3)
    # Actual trace
    ax.plot(t, actuals, color="black", lw=1.3, label="Actual BG", zorder=4)
    # Hypoglycaemia threshold
    ax.axhline(
        HYPO_THRESHOLD, color="#d62728", lw=0.8, ls="--", alpha=0.7, label="3.9 mmol/L"
    )

    min_bg = float(actuals.min())
    ep_label = ep["episode_id"].split("::")[-1]  # e.g. ep005
    ax.set_title(
        f"{ep_label} · RMSE {ep['rmse']:.2f} · min {min_bg:.1f}",
        fontsize=7.5,
        pad=2,
    )
    ax.set_xlim(0, (fh - 1) * INTERVAL_MINS / 60.0)
    ax.set_ylim(0, 22)
    ax.tick_params(labelsize=6.5)

    if show_ylabel:
        ax.set_ylabel("BG (mmol/L)", fontsize=7.5)
    if show_xlabel:
        ax.set_xlabel("Hours ahead", fontsize=7.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_grid(out_png: str, out_pdf: str) -> None:
    fig, axes = plt.subplots(
        nrows=4,
        ncols=5,
        figsize=(17, 12),
        constrained_layout=True,
    )

    for row, dataset in enumerate(DATASETS):
        print(f"  Loading {dataset}…")
        episodes = load_percentile_episodes(dataset)  # list of 5 episode dicts

        for col, ep in enumerate(episodes):
            ax = axes[row, col]
            plot_panel(
                ax,
                ep,
                show_ylabel=(col == 0),
                show_xlabel=(row == 3),
            )

        # Row label on the left margin
        axes[row, 0].annotate(
            DATASET_LABELS[dataset],
            xy=(-0.28, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            rotation=90,
        )

    # Column headers
    for col, (_, _, long_label) in enumerate(PERCENTILE_COLS):
        axes[0, col].annotate(
            long_label,
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # Shared legend (bottom of figure)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.025),
    )

    fig.suptitle(
        "Chronos-2 Probabilistic Forecasts — episode quality by median RMSE percentile (8-hour horizon)",
        fontsize=11.5,
        y=1.01,
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
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
