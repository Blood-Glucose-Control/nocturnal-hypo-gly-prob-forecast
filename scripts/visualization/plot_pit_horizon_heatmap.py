#!/usr/bin/env python3
"""Per-horizon PIT density heatmap.

For each probabilistic model, produces a heatmap that shows how calibration
quality evolves across the 8-hour (96-step, 5-min cadence) forecast horizon:

  X-axis : forecast horizon in minutes (0 → 480 min)
  Y-axis : PIT value bin (0 → 1)
  Colour : PIT density — colourmap centred at uniform density (1.0)
             red   = over-dense  (systematic bias)
             blue  = under-dense (probability mass missing here)
             white = perfectly uniform (well-calibrated)

A dashed horizontal line at PIT = 0.5 marks the median reference.

Under a perfectly calibrated model the heatmap is uniformly coloured
throughout.  Common failure modes visible here:

  Vertical colour drift      → calibration worsens at longer horizons
  Red band near PIT ≈ 0 or 1 → systematic directional bias
  U-shape in colour (red at
  extremes, blue in centre)  → over-dispersion (intervals too wide)
  Inverted-U                 → under-dispersion (intervals too narrow)

Two binning parameters let you trade resolution for smoothness:

  --n-bins     (default 10)  Number of PIT-value buckets on the Y-axis.
  --step-bins  (default 96)  Number of time-step groups on the X-axis.
               Set to 12 (40-min blocks), 24 (20-min), or 48 (10-min)
               to pool adjacent steps and reduce per-column noise.

Usage
-----
    # Auto-discover best runs from summary CSV
    python scripts/visualization/plot_pit_horizon_heatmap.py

    # Custom CSV / output directory
    python scripts/visualization/plot_pit_horizon_heatmap.py \\
        --best-csv experiments/nocturnal_forecasting/best_by_model_dataset.csv \\
        --outdir   results/pit_horizon_heatmaps

    # Smooth noise by grouping into 24 time-step bins (20-min resolution)
    python scripts/visualization/plot_pit_horizon_heatmap.py --step-bins 24

    # Test against a single run directory
    python scripts/visualization/plot_pit_horizon_heatmap.py \\
        --eval-dirs experiments/nocturnal_forecasting/512ctx_96fh/toto/<run>/ \\
        --labels    toto \\
        --outdir    /tmp/heatmap_test/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd


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

MODEL_LABELS: Dict[str, str] = {
    "chronos2": "Chronos-2",
    "moirai": "Moirai",
    "toto": "Toto",
    "tide": "TiDE",
    "timesfm": "TimesFM",
    "timegrad": "TimeGrad",
    "sundial": "Sundial",
}

MODEL_MARKERS: Dict[str, str] = {
    "chronos2": "o",  # circle
    "moirai": "s",  # square
    "toto": "^",  # triangle up
    "tide": "D",  # diamond
    "timesfm": "v",  # triangle down
    "timegrad": "P",  # plus (filled)
    "sundial": "X",  # x (filled)
}

DATASET_LABELS: Dict[str, str] = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
    "tamborlane_2008": "Tamborlane 2008",
}

# 5-minute cadence × 96 steps = 480 minutes = 8 hours
MINUTES_PER_STEP: int = 5
TOTAL_STEPS: int = 96

DEFAULT_BEST_CSV = "experiments/nocturnal_forecasting/best_by_model_dataset.csv"
DEFAULT_OUTDIR = "results/pit_horizon_heatmaps"


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
        help="Path to best_by_model_dataset.csv (default: %(default)s). "
        "Ignored when --eval-dirs is provided.",
    )
    p.add_argument(
        "--eval-dirs",
        nargs="+",
        metavar="DIR",
        default=None,
        help="One or more eval run directories (each containing forecasts.npz). "
        "Overrides --best-csv.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help="Model labels corresponding to --eval-dirs (same order). "
        "Required when --eval-dirs is used.",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of PIT-value buckets on the Y-axis (default: %(default)s).",
    )
    p.add_argument(
        "--step-bins",
        type=int,
        default=TOTAL_STEPS,
        help="Number of time-step groups on the X-axis (default: %(default)s = "
        "one column per step). Lower values (e.g. 12, 24, 48) pool adjacent "
        "steps and reduce per-column noise.",
    )
    p.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Directory for output figures (default: %(default)s).",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each cell with its density value (best with --step-bins <= 32).",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF output (only save PNG).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution (default: %(default)s).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_pit_matrix(run_path: str) -> Optional[np.ndarray]:
    """Load forecasts.npz and return a (n_episodes, forecast_length) PIT matrix.

    Returns None if the run is non-probabilistic (empty quantile_forecasts) or
    if forecasts.npz is missing.
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

    n_episodes, _n_q, forecast_length = q_forecasts.shape

    # compute_pit_values returns flat (n_episodes * forecast_length,)
    pit_flat = compute_pit_values(q_forecasts, actuals, list(q_levels))

    return pit_flat.reshape(n_episodes, forecast_length)


def _discover_runs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Return list of (model_label, run_path) tuples.

    Uses --eval-dirs + --labels when provided, otherwise reads --best-csv.
    """
    if args.eval_dirs is not None:
        if args.labels is None or len(args.labels) != len(args.eval_dirs):
            raise ValueError(
                "--labels must be provided and match the length of --eval-dirs"
            )
        return list(zip(args.labels, args.eval_dirs))

    df = pd.read_csv(args.best_csv)
    print(f"  {len(df)} rows, models: {sorted(df['model'].unique())}")
    return [(str(row["model"]), str(row["run_path"])) for _, row in df.iterrows()]


def load_all_pit_matrices(
    runs: List[Tuple[str, str]],
) -> Dict[str, np.ndarray]:
    """Return dict model_label → (n_total_episodes, forecast_length) PIT matrix.

    Episodes from multiple datasets for the same model are concatenated along
    axis 0.
    """
    accum: Dict[str, List[np.ndarray]] = {}

    for model_label, run_path in runs:
        pit = _load_pit_matrix(run_path)
        if pit is None:
            print(
                f"  Skipping {model_label} @ {run_path} "
                "(non-probabilistic or missing forecasts.npz)"
            )
            continue
        accum.setdefault(model_label, []).append(pit)
        print(
            f"  Loaded {model_label}: {pit.shape[0]:,} episodes, "
            f"{pit.shape[1]} steps  ({run_path})"
        )

    return {model: np.concatenate(arrays, axis=0) for model, arrays in accum.items()}


# ---------------------------------------------------------------------------
# Density computation
# ---------------------------------------------------------------------------


def _compute_step_densities(
    pit_matrix: np.ndarray,
    n_bins: int,
    step_bins: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Compute a (n_bins, step_bins) PIT density grid and per-bin MAD values.

    Args:
        pit_matrix: Shape (n_episodes, forecast_length).
        n_bins: Number of PIT-value buckets on the Y-axis.
        step_bins: Number of time-step groups on the X-axis.  Must divide
            forecast_length evenly, or the last group is dropped.

    Returns:
        density: Shape (n_bins, step_bins).  Each column is a normalised
            histogram of PIT values for that group of steps.  Uniform
            calibration → all values ≈ 1.0.
        bin_edges: Shape (n_bins + 1,) — edges of the PIT-value bins.
        mad_per_bin: List of mean absolute density deviations from 1.0 per time bin.
    """
    n_episodes, forecast_length = pit_matrix.shape

    steps_per_bin = forecast_length // step_bins
    # Truncate to a multiple of step_bins so reshape is exact
    usable_steps = steps_per_bin * step_bins
    pit_truncated = pit_matrix[:, :usable_steps]  # (n_eps, usable_steps)

    # Reshape to (n_episodes, step_bins, steps_per_bin)
    pit_grouped = pit_truncated.reshape(n_episodes, step_bins, steps_per_bin)
    # Flatten episodes × steps_per_bin → (step_bins, n_samples_per_bin)
    pit_by_bin = pit_grouped.transpose(1, 0, 2).reshape(step_bins, -1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    density = np.zeros((n_bins, step_bins), dtype=np.float64)
    mad_per_bin: List[float] = []

    for col, pit_col in enumerate(pit_by_bin):
        counts, _ = np.histogram(pit_col, bins=bin_edges)
        total = counts.sum()
        if total > 0:
            bin_width = bin_edges[1] - bin_edges[0]
            density[:, col] = counts / (total * bin_width)
        mad_per_bin.append(float(np.mean(np.abs(density[:, col] - 1.0))))

    return density, bin_edges, mad_per_bin


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_horizon_heatmap(
    pit_matrices: Dict[str, np.ndarray],
    outdir: Path,
    n_bins: int,
    step_bins: int,
    dpi: int,
    save_pdf: bool,
    annotate: bool = False,
) -> None:
    """Produce a stacked per-model PIT horizon heatmap figure.

    One subplot per model, arranged vertically.  Each subplot is a pcolormesh
    heatmap (n_bins PIT buckets × step_bins time groups).
    """
    models = sorted(pit_matrices.keys())
    n_models = len(models)
    if n_models == 0:
        print("No probabilistic models to plot — nothing to save.")
        return

    fig_height = max(2.8 * n_models, 4.0)
    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(11.0, fig_height),
        sharex=True,
        squeeze=False,
    )

    n_by_model = {m: pit_matrices[m].shape[0] for m in models}
    unique_ns = set(n_by_model.values())
    base_title = "Per-horizon PIT Density Heatmap"
    if n_models == 1:
        single_label = MODEL_LABELS.get(models[0], models[0].capitalize())
        n_str = f"n\u202f=\u202f{n_by_model[models[0]]:,} episodes"
        title = f"{base_title} \u2014 {single_label}  ({n_str})"
    elif len(unique_ns) == 1:
        n_str = f"n\u202f=\u202f{next(iter(unique_ns)):,} episodes per model"
        title = f"{base_title}  ({n_str})"
    else:
        title = base_title
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)

    # Compute all density grids first so we can set a shared colour scale
    grids: Dict[str, tuple] = {}
    for model in models:
        density, bin_edges, mad_per_bin_m = _compute_step_densities(
            pit_matrices[model], n_bins=n_bins, step_bins=step_bins
        )
        grids[model] = (density, bin_edges, mad_per_bin_m)

    uniform = 1.0
    vmin, vmax = 0.5, 1.5

    # X-axis: midpoint minute for each step-bin group
    steps_per_bin = TOTAL_STEPS // step_bins
    bin_start_steps = np.arange(step_bins) * steps_per_bin
    x_minutes = bin_start_steps * MINUTES_PER_STEP  # left edge of each bin in min
    x_mid_minutes = x_minutes + steps_per_bin * MINUTES_PER_STEP / 2.0

    # Build pcolormesh X edges (step_bins + 1 edges)
    x_edges_min = np.append(x_minutes, step_bins * steps_per_bin * MINUTES_PER_STEP)

    last_ax = axes[-1, 0]
    im = None

    for i, model in enumerate(models):
        ax = axes[i, 0]
        density, bin_edges, mad_per_bin = grids[model]
        color = MODEL_COLORS.get(model, "#7f7f7f")
        label = MODEL_LABELS.get(model, model.capitalize())

        # pcolormesh: X edges (step_bins+1,), Y edges (n_bins+1,), Z (n_bins, step_bins)
        im = ax.pcolormesh(
            x_edges_min,
            bin_edges,
            density,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        # Cell annotations — fontsize scales down for denser column grids
        if annotate:
            annotate_fs = max(2.5, 5.5 * (32 / step_bins))
            bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            col_mids = x_edges_min[:-1] + 0.5 * np.diff(x_edges_min)
            for ci, cx in enumerate(col_mids):
                for ri, ry in enumerate(bin_mids):
                    val = density[ri, ci]
                    ax.text(
                        cx,
                        ry,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=annotate_fs,
                        color="black",
                    )

        # Median reference line
        ax.axhline(0.5, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)

        ax.set_title(
            label,
            fontsize=9,
            fontweight="bold",
            color=color,
            loc="left",
            pad=16,
        )
        ax.set_ylabel("PIT value", fontsize=9)

        ax.set_ylim(0.0, 1.0)
        yticks = np.arange(0.0, 1.1, 0.1)
        ax.set_yticks(yticks)
        ax.tick_params(labelsize=8)
        ax.grid(False)
        ax.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=7.5)

        # MAD (mean |density - 1.0|) per time-bin above the heatmap
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            0.000,
            1.025,
            "MAD:",
            ha="right",
            va="bottom",
            fontsize=5.5,
            transform=ax.transAxes,
            color="#555555",
        )
        for cx, mad_val in zip(x_mid_minutes, mad_per_bin):
            ax.text(
                cx,
                1.025,
                f"{mad_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=5,
                transform=trans,
                clip_on=False,
                color="#333333",
            )

        # Per-subplot colourbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, aspect=20)
        cbar.set_label("PIT density", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.axhline(
            uniform, color="black", linewidth=0.8, linestyle="--", alpha=0.7
        )

    # X-axis labels only on the bottom subplot
    last_ax.set_xlabel("Forecast horizon (minutes)", fontsize=9)
    last_ax.set_xlim(x_edges_min[0], x_edges_min[-1])

    # Tick marks at every 60 min (every 12 steps)
    hour_ticks = np.arange(0, TOTAL_STEPS * MINUTES_PER_STEP + 1, 60)
    last_ax.set_xticks(hour_ticks)
    last_ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    last_ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout(rect=(0, 0, 1, 1.01))

    if n_models == 1:
        stem_name = f"pit_horizon_heatmap_{models[0]}"
    else:
        stem_name = "pit_horizon_heatmap"
    _save(fig, outdir / stem_name, dpi=dpi, pdf=save_pdf)


# ---------------------------------------------------------------------------
# MAD over horizon line plot
# ---------------------------------------------------------------------------


def plot_mad_over_horizon(
    pit_matrices: Dict[str, np.ndarray],
    outdir: Path,
    n_bins: int,
    step_bins: int,
    dpi: int,
    save_pdf: bool,
) -> None:
    """Line plot of mean absolute PIT density deviation (MAD) across the horizon.

    One line per model, X-axis = forecast horizon in minutes, Y-axis = MAD.
    MAD = 0 when calibration is perfectly uniform; higher = worse.
    """
    models = sorted(pit_matrices.keys())
    n_models = len(models)
    if n_models == 0:
        return

    steps_per_bin = TOTAL_STEPS // step_bins
    bin_start_steps = np.arange(step_bins) * steps_per_bin
    x_mid_minutes = (
        bin_start_steps * MINUTES_PER_STEP + steps_per_bin * MINUTES_PER_STEP / 2.0
    )

    fig, ax = plt.subplots(figsize=(10.0, 3.5))

    for model in models:
        density, _bin_edges, mad_per_bin = _compute_step_densities(
            pit_matrices[model], n_bins=n_bins, step_bins=step_bins
        )
        color = MODEL_COLORS.get(model, "#7f7f7f")
        label = MODEL_LABELS.get(model, model.capitalize())
        marker = MODEL_MARKERS.get(model, "o")
        ax.plot(
            x_mid_minutes,
            mad_per_bin,
            label=label,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=4.5,
            alpha=0.55,
        )

    ax.set_xlabel("Forecast horizon (minutes)", fontsize=10)
    ax.set_ylabel("Mean calibration deviation\n(0 = perfectly uniform)", fontsize=10)
    n_by_model = {m: pit_matrices[m].shape[0] for m in models}
    unique_ns = set(n_by_model.values())
    n_str = (
        f"n\u202f=\u202f{next(iter(unique_ns)):,} episodes per model"
        if len(unique_ns) == 1
        else ""
    )
    title = "PIT Calibration Deviation over Forecast Horizon"
    if n_str:
        title += f"  ({n_str})"
    ax.set_title(title, fontsize=11, fontweight="bold")

    hour_ticks = np.arange(0, TOTAL_STEPS * MINUTES_PER_STEP + 1, 60)
    ax.set_xticks(hour_ticks)
    ax.set_xlim(x_mid_minutes[0] - 5, x_mid_minutes[-1] + 5)
    ax.set_ylim(bottom=0)
    ax.axhline(0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(
        fontsize=9,
        framealpha=0.9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=n_models,
        borderaxespad=0.0,
    )
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    if n_models == 1:
        stem_name = f"pit_mad_over_horizon_{models[0]}"
    else:
        stem_name = "pit_mad_over_horizon"
    _save(fig, outdir / stem_name, dpi=dpi, pdf=save_pdf)


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

    if args.step_bins < 1 or TOTAL_STEPS % args.step_bins != 0:
        valid = [s for s in range(1, TOTAL_STEPS + 1) if TOTAL_STEPS % s == 0]
        raise ValueError(
            f"--step-bins {args.step_bins} does not evenly divide {TOTAL_STEPS} steps. "
            f"Valid values: {valid}"
        )

    outdir = Path(args.outdir)

    if args.eval_dirs is not None:
        print(f"Using --eval-dirs ({len(args.eval_dirs)} dirs) …")
    else:
        print(f"Reading {args.best_csv} …")

    runs = _discover_runs(args)

    print(
        f"Loading forecasts.npz and computing PIT matrices "
        f"(step-bins={args.step_bins}, n-bins={args.n_bins}) …"
    )
    pit_matrices = load_all_pit_matrices(runs)

    n_prob = len(pit_matrices)
    if n_prob == 0:
        print("No probabilistic runs found — nothing to plot.")
        return

    print(f"\nPlotting {n_prob} model(s) …")
    plot_horizon_heatmap(
        pit_matrices,
        outdir=outdir,
        n_bins=args.n_bins,
        step_bins=args.step_bins,
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
        annotate=args.annotate,
    )
    plot_mad_over_horizon(
        pit_matrices,
        outdir=outdir,
        n_bins=args.n_bins,
        step_bins=args.step_bins,
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
    )
    print("Done.")


if __name__ == "__main__":
    main()
