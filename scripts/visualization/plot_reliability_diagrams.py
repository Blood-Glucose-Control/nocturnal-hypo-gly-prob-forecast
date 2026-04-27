#!/usr/bin/env python3
"""Reliability (quantile calibration) diagrams with ECE annotations.

For each model × dataset pair, plots the reliability curve:
  x-axis : nominal quantile level q
  y-axis : empirical P(actual ≤ forecast_q) across all episodes and timesteps

A perfectly calibrated model lies on the diagonal. Deviation indicates:
  Above diagonal → over-forecasting (predicted quantiles too high / over-dispersed)
  Below diagonal → under-forecasting (predicted quantiles too low / under-dispersed)

Like the PIT plots, two output tiers are produced:

  Main-body figure  — one subplot per model, reliability curve aggregated
                       across all available datasets. Includes ECE annotation.
                       Saved as ``<outdir>/reliability_main_body.{png,pdf}``.

  Appendix figures  — one figure per dataset, one subplot per model.
                       Saved as ``<outdir>/reliability_appendix_{dataset}.{png,pdf}``.

Usage
-----
    python scripts/visualization/plot_reliability_diagrams.py
    python scripts/visualization/plot_reliability_diagrams.py \\
        --best-csv experiments/nocturnal_forecasting/best_by_model_dataset.csv \\
        --outdir   results/reliability_diagrams
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics.probabilistic import compute_reliability_curve, compute_ece

# ---------------------------------------------------------------------------
# Aesthetics — consistent with other visualisation scripts
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

MODEL_LABELS: Dict[str, str] = {
    "chronos2": "Chronos-2",
    "moirai": "Moirai",
    "toto": "Toto",
    "tide": "TiDE",
    "timesfm": "TimesFM",
    "timegrad": "TimeGrad",
    "sundial": "Sundial",
}

DEFAULT_BEST_CSV = "experiments/nocturnal_forecasting/best_by_model_dataset.csv"
DEFAULT_OUTDIR = "results/reliability_diagrams"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--best-csv", default=DEFAULT_BEST_CSV)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--no-pdf", action="store_true", help="Skip PDF output")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Type alias: (nominal, empirical) curve pair
_Curve = Tuple[np.ndarray, np.ndarray]


def _load_curve_for_run(run_path: str) -> Optional[Tuple[_Curve, float, int]]:
    """Load forecasts.npz and return (reliability_curve, ece, n_samples).

    Returns None for non-probabilistic runs or missing files.
    """
    npz_path = Path(run_path) / "forecasts.npz"
    if not npz_path.exists():
        warnings.warn(f"forecasts.npz not found: {npz_path}", stacklevel=2)
        return None

    data = np.load(npz_path, allow_pickle=False)
    q_forecasts = data["quantile_forecasts"]  # (n_eps, n_q, fh) or (0,0,0)
    q_levels = data["quantile_levels"]  # (n_q,)
    actuals = data["actuals"]  # (n_eps, fh)

    if q_forecasts.ndim != 3 or q_forecasts.shape[1] == 0:
        return None  # point-only run

    levels = list(q_levels)
    nominal, empirical = compute_reliability_curve(q_forecasts, actuals, levels)
    ece = compute_ece(q_forecasts, actuals, levels)
    n_samples = q_forecasts.shape[0] * q_forecasts.shape[2]
    return (nominal, empirical), ece, n_samples


def load_all_curves(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Tuple[_Curve, float, int]]]:
    """Return nested dict: curves[model][dataset] = ((nominal, empirical), ece, n)."""
    curves: Dict[str, Dict[str, Tuple[_Curve, float, int]]] = {}

    for _, row in df.iterrows():
        model = str(row["model"])
        dataset = str(row["dataset"])
        result = _load_curve_for_run(str(row["run_path"]))
        if result is None:
            print(f"  Skipping {model}/{dataset} (non-probabilistic or missing)")
            continue
        (nominal, empirical), ece, n = result
        print(f"  Loaded {model}/{dataset}: ECE={ece:.4f}, n={n:,}")
        curves.setdefault(model, {})[dataset] = ((nominal, empirical), ece, n)

    return curves


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _draw_reliability_subplot(
    ax: plt.Axes,
    curves: List[Tuple[np.ndarray, np.ndarray]],
    eces: List[float],
    labels: List[str],
    colors: List[str],
    title: str,
    show_x_label: bool = True,
    show_y_label: bool = True,
    show_legend: bool = False,
) -> None:
    """Draw one reliability diagram panel, potentially multiple curves."""
    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        zorder=2,
        label="Perfect calibration",
    )

    for (nominal, empirical), ece, label, color in zip(curves, eces, labels, colors):
        ax.plot(
            nominal,
            empirical,
            color=color,
            linewidth=2.0,
            alpha=0.85,
            zorder=3,
            label=f"{label}  ECE={ece:.3f}",
        )
        ax.fill_between(nominal, nominal, empirical, color=color, alpha=0.08, zorder=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, fontweight="bold")

    if show_x_label:
        ax.set_xlabel("Nominal quantile level", fontsize=8.5)
    if show_y_label:
        ax.set_ylabel("Empirical coverage", fontsize=8.5)

    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.35, zorder=0)

    if show_legend:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)


# ---------------------------------------------------------------------------
# Main-body figure: one subplot per model, curve pooled across all datasets
# ---------------------------------------------------------------------------


def plot_main_body(
    curves: Dict[str, Dict[str, Tuple[_Curve, float, int]]],
    outdir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """Each model gets one subplot; the curve is computed over all its datasets."""
    models = sorted(curves.keys())
    n_models = len(models)
    if n_models == 0:
        print("No probabilistic models to plot.")
        return

    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.8))
    axes_flat = np.array(axes).ravel()

    fig.suptitle(
        "Reliability diagrams — quantile calibration\n"
        "Pooled across all datasets — dashed: perfect calibration",
        fontsize=12,
        fontweight="bold",
    )

    for i, model in enumerate(models):
        ax = axes_flat[i]
        model_data = curves[model]

        # Pool all episodes/timesteps across datasets to get a single curve
        for (nominal, _), _ece, _n in model_data.values():
            # We need to reload from disk here; instead, just merge curves
            # by taking a weighted mean (weighted by n_samples).
            pass  # handled below via weighted average of empirical arrays

        # Weighted average of per-dataset empirical curves
        total_n = sum(n for (_, __), ___, n in model_data.values())
        nominal_ref = None
        weighted_emp = None
        for (nominal, empirical), _ece, n in model_data.values():
            if nominal_ref is None:
                nominal_ref = nominal
                weighted_emp = empirical * n
            else:
                assert np.allclose(nominal, nominal_ref), (
                    f"Quantile grids differ across datasets for model '{model}'; "
                    "cannot average empirical curves without interpolation."
                )
                weighted_emp += empirical * n
        assert nominal_ref is not None and weighted_emp is not None
        agg_empirical = weighted_emp / total_n

        # ECE from the aggregated curve
        agg_ece = float(np.trapz(np.abs(agg_empirical - nominal_ref), nominal_ref))

        datasets_used = ", ".join(
            DATASET_LABELS.get(d, d) for d in sorted(model_data.keys())
        )
        color = MODEL_COLORS.get(model, _DEFAULT_COLOR)
        label = MODEL_LABELS.get(model, model.capitalize())

        _draw_reliability_subplot(
            ax,
            curves=[(nominal_ref, agg_empirical)],
            eces=[agg_ece],
            labels=[label],
            colors=[color],
            title=label,
            show_x_label=(i >= n_models - ncols),
            show_y_label=(i % ncols == 0),
            show_legend=True,
        )
        ax.text(
            0.97,
            0.03,
            datasets_used,
            transform=ax.transAxes,
            fontsize=6.5,
            ha="right",
            va="bottom",
            color="#555",
            style="italic",
        )

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    _save(fig, outdir / "reliability_main_body", dpi=dpi, pdf=save_pdf)


# ---------------------------------------------------------------------------
# Appendix figures: one per dataset, all models overlaid on one panel +
#                   individual per-model subplots for clarity
# ---------------------------------------------------------------------------


def plot_appendix_per_dataset(
    curves: Dict[str, Dict[str, Tuple[_Curve, float, int]]],
    outdir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """One figure per dataset.  Layout: overlay panel + per-model subplots."""
    all_datasets: List[str] = sorted(
        {ds for model_dict in curves.values() for ds in model_dict}
    )

    for dataset in all_datasets:
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        models_with_data = sorted(m for m, d in curves.items() if dataset in d)
        if not models_with_data:
            continue

        n_models = len(models_with_data)
        # Layout: 1 overlay panel + n_models individual panels
        n_panels = 1 + n_models
        ncols = min(4, n_panels)
        nrows = (n_panels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.8))
        axes_flat = np.array(axes).ravel()

        fig.suptitle(
            f"Reliability diagrams — {dataset_label}\n" "Dashed: perfect calibration",
            fontsize=12,
            fontweight="bold",
        )

        # Panel 0: all models overlaid
        ax_overlay = axes_flat[0]
        overlay_curves = []
        overlay_eces = []
        overlay_labels = []
        overlay_colors = []
        for model in models_with_data:
            (nominal, empirical), ece, _ = curves[model][dataset]
            overlay_curves.append((nominal, empirical))
            overlay_eces.append(ece)
            overlay_labels.append(MODEL_LABELS.get(model, model.capitalize()))
            overlay_colors.append(MODEL_COLORS.get(model, _DEFAULT_COLOR))

        _draw_reliability_subplot(
            ax_overlay,
            curves=overlay_curves,
            eces=overlay_eces,
            labels=overlay_labels,
            colors=overlay_colors,
            title="All models",
            show_x_label=True,
            show_y_label=True,
            show_legend=True,
        )

        # Panels 1..n_models: individual model panels
        for j, model in enumerate(models_with_data):
            ax = axes_flat[1 + j]
            (nominal, empirical), ece, n = curves[model][dataset]
            color = MODEL_COLORS.get(model, _DEFAULT_COLOR)
            label = MODEL_LABELS.get(model, model.capitalize())

            _draw_reliability_subplot(
                ax,
                curves=[(nominal, empirical)],
                eces=[ece],
                labels=[label],
                colors=[color],
                title=label,
                show_x_label=True,
                show_y_label=(j % ncols == 0),
                show_legend=True,
            )

        for k in range(n_panels, len(axes_flat)):
            axes_flat[k].set_visible(False)

        plt.tight_layout()
        _save(fig, outdir / f"reliability_appendix_{dataset}", dpi=dpi, pdf=save_pdf)


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

    print("Loading forecasts.npz and computing reliability curves …")
    curves = load_all_curves(df)

    n_prob = len(curves)
    print(f"\n{n_prob} probabilistic model(s): {sorted(curves.keys())}")
    if n_prob == 0:
        print("Nothing to plot.")
        return

    print("\nGenerating main-body figure …")
    plot_main_body(curves, outdir, dpi=args.dpi, save_pdf=not args.no_pdf)

    print("\nGenerating appendix figures (one per dataset) …")
    plot_appendix_per_dataset(curves, outdir, dpi=args.dpi, save_pdf=not args.no_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()
