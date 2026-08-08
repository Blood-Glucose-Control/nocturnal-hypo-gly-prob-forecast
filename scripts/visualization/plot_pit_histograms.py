#!/usr/bin/env python3
"""Probability Integral Transform (PIT) histogram plots.

Generates two tiers of calibration figures for the paper:

  Main-body figure  — one subplot per probabilistic model, PIT aggregated
                       across all available datasets. Saved as
                       ``<outdir>/pit_main_body.{png,pdf}``.

  Appendix figures  — one figure per dataset, one subplot per model.
                       Saved as ``<outdir>/pit_appendix_{dataset}.{png,pdf}``.

Under a perfectly calibrated forecast, the PIT histogram should be flat
(uniform). Deviations indicate:
  U-shaped  → under-dispersion (intervals too narrow)
  Hump      → over-dispersion  (intervals too wide)
  Left skew → positive bias    (forecasts consistently too high)
  Right skew→ negative bias    (forecasts consistently too low)

Usage
-----
    python scripts/visualization/plot_pit_histograms.py
    python scripts/visualization/plot_pit_histograms.py \\
        --best-csv experiments/nocturnal_forecasting/best_by_model_dataset.csv \\
        --outdir   results/pit_histograms
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics.probabilistic import compute_pit_values

# ---------------------------------------------------------------------------
# Aesthetics — kept consistent with other visualisation scripts
# ---------------------------------------------------------------------------

MODEL_COLORS: Dict[str, str] = {
    "chronos2": "#e377c2",
    "moirai": "#1f77b4",
    "toto": "#ff7f0e",
    "tide": "#2ca02c",
    "timesfm": "#d62728",
    "timegrad": "#9467bd",
    "sundial": "#8c564b",
}
_DEFAULT_COLOR = "#7f7f7f"

DATASET_LABELS: Dict[str, str] = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
    "tamborlane_2008": "Tamborlane 2008",
}

# Display-friendly model names
MODEL_LABELS: Dict[str, str] = {
    "chronos2": "Chronos-2",
    "moirai": "Moirai",
    "toto": "Toto",
    "tide": "TiDE",
    "timesfm": "TimesFM",
    "timegrad": "TimeGrad",
    "sundial": "Sundial",
}

N_BINS = 10
UNIFORM_DENSITY = 1.0  # PIT ~ U[0,1] → density = 1 everywhere

DEFAULT_BEST_CSV = "experiments/nocturnal_forecasting/best_by_model_dataset.csv"
DEFAULT_OUTDIR = "results/pit_histograms"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--best-csv",
        default=DEFAULT_BEST_CSV,
        help="Path to best_by_model_dataset.csv (default: %(default)s)",
    )
    p.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Directory for output figures (default: %(default)s)",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF output (only save PNG)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution (default: %(default)s)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_pit_for_run(run_path: str) -> Optional[np.ndarray]:
    """Load forecasts.npz from *run_path* and return flat PIT values.

    Returns None if the run is non-probabilistic (empty quantile_forecasts).
    """
    npz_path = Path(run_path) / "forecasts.npz"
    if not npz_path.exists():
        warnings.warn(f"forecasts.npz not found: {npz_path}", stacklevel=2)
        return None

    data = np.load(npz_path, allow_pickle=False)
    q_forecasts = data["quantile_forecasts"]  # (n_eps, n_q, fh) or (0, 0, 0)
    q_levels = data["quantile_levels"]  # (n_q,)
    actuals = data["actuals"]  # (n_eps, fh)

    if q_forecasts.ndim != 3 or q_forecasts.shape[1] == 0:
        return None  # point-only run

    return compute_pit_values(q_forecasts, actuals, list(q_levels))


def load_all_pit_data(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return nested dict pit_data[model][dataset] = flat PIT array."""
    pit_data: Dict[str, Dict[str, np.ndarray]] = {}

    for _, row in df.iterrows():
        model = str(row["model"])
        dataset = str(row["dataset"])
        run_path = str(row["run_path"])

        pit = _load_pit_for_run(run_path)
        if pit is None:
            print(f"  Skipping {model}/{dataset} (non-probabilistic or missing)")
            continue

        print(f"  Loaded {model}/{dataset}: {len(pit):,} PIT values")
        pit_data.setdefault(model, {})[dataset] = pit

    return pit_data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _draw_pit_subplot(
    ax: plt.Axes,
    pit: np.ndarray,
    color: str,
    title: str,
    n_samples: int,
    show_x_label: bool = True,
    show_y_label: bool = True,
) -> None:
    """Draw one PIT histogram panel."""
    counts, edges = np.histogram(pit, bins=N_BINS, range=(0.0, 1.0), density=False)
    density = counts / (len(pit) * (edges[1] - edges[0]))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax.bar(
        bin_centers,
        density,
        width=width * 0.9,
        color=color,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Uniform reference line
    ax.axhline(
        UNIFORM_DENSITY,
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        zorder=4,
        label="Uniform (ideal)",
    )

    # KS test against U[0,1]
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    significance = "**" if ks_p < 0.01 else ("*" if ks_p < 0.05 else "")
    ax.text(
        0.97,
        0.96,
        f"KS={ks_stat:.3f}{significance}\nn={n_samples:,}",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="right",
        va="top",
        color="#222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7, ec="none"),
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)

    if show_x_label:
        ax.set_xlabel("PIT value", fontsize=8.5)
    if show_y_label:
        ax.set_ylabel("Density", fontsize=8.5)

    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)


def _add_uniform_legend(ax: plt.Axes) -> None:
    h = mlines.Line2D(
        [], [], color="#333333", linestyle="--", linewidth=1.1, label="Uniform (ideal)"
    )
    ax.legend(handles=[h], fontsize=8, loc="upper left", framealpha=0.8)


# ---------------------------------------------------------------------------
# Main-body figure: one subplot per model, aggregated over all datasets
# ---------------------------------------------------------------------------


def plot_main_body(
    pit_data: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """One subplot per probabilistic model, PIT pooled across all datasets."""
    models = sorted(pit_data.keys())
    n_models = len(models)
    if n_models == 0:
        print("No probabilistic models to plot.")
        return

    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.5))
    axes_flat = np.array(axes).ravel()

    fig.suptitle(
        "Probability Integral Transform (PIT) histograms\n"
        "Pooled across all datasets — dashed line: ideal uniform distribution",
        fontsize=12,
        fontweight="bold",
    )

    for i, model in enumerate(models):
        ax = axes_flat[i]
        all_pit = np.concatenate(list(pit_data[model].values()))
        datasets_used = ", ".join(
            DATASET_LABELS.get(d, d) for d in sorted(pit_data[model].keys())
        )
        color = MODEL_COLORS.get(model, _DEFAULT_COLOR)
        label = MODEL_LABELS.get(model, model.capitalize())

        _draw_pit_subplot(
            ax,
            all_pit,
            color=color,
            title=label,
            n_samples=len(all_pit),
            show_x_label=(i >= n_models - ncols),
            show_y_label=(i % ncols == 0),
        )
        ax.text(
            0.03,
            0.03,
            datasets_used,
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="bottom",
            color="#555",
            style="italic",
        )

    # Add legend in first subplot
    _add_uniform_legend(axes_flat[0])

    # Hide unused axes
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, outdir / "pit_main_body", dpi=dpi, pdf=save_pdf)


# ---------------------------------------------------------------------------
# Appendix figures: one per dataset, subplots per model
# ---------------------------------------------------------------------------


def plot_appendix_per_dataset(
    pit_data: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """One figure per dataset with all probabilistic models as subplots."""
    # Collect all datasets present in the data
    all_datasets: List[str] = sorted(
        {ds for model_dict in pit_data.values() for ds in model_dict}
    )

    for dataset in all_datasets:
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        models_with_data = sorted(m for m, d in pit_data.items() if dataset in d)
        if not models_with_data:
            continue

        n_models = len(models_with_data)
        ncols = 3
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.5))
        axes_flat = np.array(axes).ravel()

        fig.suptitle(
            f"PIT histograms — {dataset_label}\n"
            "Dashed line: ideal uniform distribution",
            fontsize=12,
            fontweight="bold",
        )

        for i, model in enumerate(models_with_data):
            ax = axes_flat[i]
            pit = pit_data[model][dataset]
            color = MODEL_COLORS.get(model, _DEFAULT_COLOR)
            label = MODEL_LABELS.get(model, model.capitalize())

            _draw_pit_subplot(
                ax,
                pit,
                color=color,
                title=label,
                n_samples=len(pit),
                show_x_label=(i >= n_models - ncols),
                show_y_label=(i % ncols == 0),
            )

        _add_uniform_legend(axes_flat[0])

        for j in range(n_models, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        _save(fig, outdir / f"pit_appendix_{dataset}", dpi=dpi, pdf=save_pdf)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, stem: Path, dpi: int, pdf: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {png_path}")
    if pdf:
        pdf_path = stem.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved → {pdf_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    print(f"Reading {args.best_csv} …")
    df = pd.read_csv(args.best_csv)
    print(f"  {len(df)} rows, models: {sorted(df['model'].unique())}")

    print("Loading forecasts.npz and computing PIT values …")
    pit_data = load_all_pit_data(df)

    n_prob = len(pit_data)
    print(f"\n{n_prob} probabilistic model(s) found: {sorted(pit_data.keys())}")
    if n_prob == 0:
        print("Nothing to plot.")
        return

    print("\nGenerating main-body figure …")
    plot_main_body(pit_data, outdir, dpi=args.dpi, save_pdf=not args.no_pdf)

    print("\nGenerating appendix figures (one per dataset) …")
    plot_appendix_per_dataset(pit_data, outdir, dpi=args.dpi, save_pdf=not args.no_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()
