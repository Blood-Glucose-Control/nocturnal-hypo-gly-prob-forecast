#!/usr/bin/env python3
"""Combined PIT histogram + per-horizon heatmap (one figure, side-by-side).

For each probabilistic model one row is produced:

  LEFT  (1 grid col) : overall PIT histogram pooled across all datasets and
                       forecast steps — shows global calibration at a glance.
  RIGHT (3 grid cols): per-horizon PIT density heatmap — shows how calibration
                       evolves across the 8-hour forecast horizon.

Under a perfectly calibrated model the histogram bars are all ≈ 1 and the
heatmap is uniformly coloured.

Usage
-----
    python scripts/visualization/plot_pit_combined.py \\
        --eval-dirs <dir1> <dir2> ... \\
        --labels    <label1> <label2> ... \\
        --outdir    results/pit_combined/ \\
        --step-bins 32 --annotate --no-pdf
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
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from src.evaluation.metrics.probabilistic import compute_pit_values

# ---------------------------------------------------------------------------
# Aesthetics — kept consistent with the other visualisation scripts
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
    "chronos2": "o",
    "moirai": "s",
    "toto": "^",
    "tide": "D",
    "timesfm": "v",
    "timegrad": "P",
    "sundial": "X",
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

N_BINS: int = 10  # PIT-value buckets (Y-axis of heatmap, X-axis of histogram)
UNIFORM_DENSITY: float = 1.0

DEFAULT_BEST_CSV = "experiments/nocturnal_forecasting/best_by_model_dataset.csv"
DEFAULT_OUTDIR = "results/pit_combined"


# ---------------------------------------------------------------------------
# Layout & font configuration — tweak these instead of hunting through the code
# ---------------------------------------------------------------------------


class LAYOUT:
    # ── Figure dimensions ────────────────────────────────────────────────────
    FIG_W: float = 6.75  # width in inches  (NeurIPS text width ≈ 6.75 in)
    FIG_H: float = 8.5  # height in inches

    # ── Grid spacing ─────────────────────────────────────────────────────────
    WSPACE: float = 0.1  # horizontal gap between histogram and heatmap columns
    HSPACE: float = 0.25  # vertical gap between model rows
    WIDTH_RATIOS = [2, 3, 3, 3]  # histogram col : heatmap cols

    # ── Title / suptitle ─────────────────────────────────────────────────────
    SUPTITLE_FS: float = 8  # main figure title font size
    SUPTITLE_Y: float = 0.99  # vertical position of suptitle (figure fraction)
    SUBPLOT_TOP: float = (
        0.96  # top edge of subplot grid  (gap = SUPTITLE_Y - SUBPLOT_TOP)
    )

    # ── Model name labels (histogram title) ──────────────────────────────────
    MODEL_TITLE_FS: float = 7  # font size
    MODEL_TITLE_PAD: float = 2  # gap between label and plot top (points)

    # ── Axis labels ──────────────────────────────────────────────────────────
    AXIS_LABEL_FS: float = 7  # "PIT value", "Density", "Forecast horizon (minutes)"

    # ── Tick labels ──────────────────────────────────────────────────────────
    TICK_LABEL_FS: float = 6  # all tick labels (hist Y, hist X, heatmap X)

    # ── KS annotation box ────────────────────────────────────────────────────
    KS_FS: float = 6  # "KS=0.xxx" font size
    KS_BOX_PAD: float = 0.2  # inner padding of the rounded box (in font units)

    # ── MAD row above heatmap ─────────────────────────────────────────────────
    MAD_FS: float = 3.25  # "MAD:" label and per-bin values font size
    MAD_Y: float = 1.025  # vertical offset above heatmap (axes fraction)

    # ── Heatmap cell annotations ─────────────────────────────────────────────
    ANNOTATE_FS_BASE: float = 3.25  # scales as BASE * (32 / step_bins), min 2.0 pt

    # ── Colourbar ────────────────────────────────────────────────────────────
    CBAR_LABEL_FS: float = 6  # "PIT density" label font size
    CBAR_TICK_FS: float = 6  # tick label font size
    CBAR_FRACTION: float = 0.02
    CBAR_PAD: float = 0.01
    CBAR_ASPECT: float = 20

    # ── Heatmap colour scale ──────────────────────────────────────────────────
    VMIN: float = 0.5
    VMAX: float = 1.5


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
        help="Path to best_by_model_dataset.csv. Ignored when --eval-dirs is given.",
    )
    p.add_argument(
        "--eval-dirs",
        nargs="+",
        metavar="DIR",
        default=None,
        help="One or more eval run directories (each containing forecasts.npz).",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help="Model labels corresponding to --eval-dirs (same order). Required with --eval-dirs.",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=N_BINS,
        help="Number of PIT-value buckets on the Y-axis / histogram X-axis (default: %(default)s).",
    )
    p.add_argument(
        "--step-bins",
        type=int,
        default=TOTAL_STEPS,
        help="Number of time-step groups on the heatmap X-axis (default: %(default)s). "
        "Lower values (e.g. 32) pool adjacent steps and reduce per-column noise.",
    )
    p.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Output directory (default: %(default)s).",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each heatmap cell with its density value (best with --step-bins <= 32).",
    )
    p.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF output (only save PNG).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: %(default)s).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading  (mirrors plot_pit_horizon_heatmap.py)
# ---------------------------------------------------------------------------


def _load_pit_matrix(run_path: str) -> Optional[np.ndarray]:
    """Load forecasts.npz and return a (n_episodes, forecast_length) PIT matrix."""
    npz_path = Path(run_path) / "forecasts.npz"
    if not npz_path.exists():
        warnings.warn(f"forecasts.npz not found: {npz_path}", stacklevel=2)
        return None

    data = np.load(npz_path, allow_pickle=False)
    q_forecasts = data["quantile_forecasts"]
    q_levels = data["quantile_levels"]
    actuals = data["actuals"]

    if q_forecasts.ndim != 3 or q_forecasts.shape[1] == 0:
        return None

    n_episodes, _n_q, forecast_length = q_forecasts.shape
    pit_flat = compute_pit_values(q_forecasts, actuals, list(q_levels))
    return pit_flat.reshape(n_episodes, forecast_length)


def _discover_runs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.eval_dirs is not None:
        if args.labels is None or len(args.labels) != len(args.eval_dirs):
            raise ValueError(
                "--labels must be provided and match the length of --eval-dirs"
            )
        return list(zip(args.labels, args.eval_dirs))
    df = pd.read_csv(args.best_csv)
    print(f"  {len(df)} rows, models: {sorted(df['model'].unique())}")
    return [(str(row["model"]), str(row["run_path"])) for _, row in df.iterrows()]


def load_all_pit_matrices(runs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    accum: Dict[str, List[np.ndarray]] = {}
    for model_label, run_path in runs:
        pit = _load_pit_matrix(run_path)
        if pit is None:
            print(
                f"  Skipping {model_label} @ {run_path} (non-probabilistic or missing)"
            )
            continue
        accum.setdefault(model_label, []).append(pit)
        print(
            f"  Loaded {model_label}: {pit.shape[0]:,} episodes, {pit.shape[1]} steps  ({run_path})"
        )
    return {model: np.concatenate(arrays, axis=0) for model, arrays in accum.items()}


# ---------------------------------------------------------------------------
# Density computation  (mirrors plot_pit_horizon_heatmap.py)
# ---------------------------------------------------------------------------


def _compute_step_densities(
    pit_matrix: np.ndarray,
    n_bins: int,
    step_bins: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Return (density [n_bins × step_bins], bin_edges, mad_per_bin)."""
    n_episodes, forecast_length = pit_matrix.shape
    steps_per_bin = forecast_length // step_bins
    usable_steps = steps_per_bin * step_bins
    pit_grouped = pit_matrix[:, :usable_steps].reshape(
        n_episodes, step_bins, steps_per_bin
    )
    pit_by_bin = pit_grouped.transpose(1, 0, 2).reshape(step_bins, -1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    density = np.zeros((n_bins, step_bins), dtype=np.float64)
    mad_per_bin: List[float] = []

    for col, pit_col in enumerate(pit_by_bin):
        counts, _ = np.histogram(pit_col, bins=bin_edges)
        total = counts.sum()
        if total > 0:
            density[:, col] = counts / (total * (bin_edges[1] - bin_edges[0]))
        mad_per_bin.append(float(np.mean(np.abs(density[:, col] - 1.0))))

    return density, bin_edges, mad_per_bin


# ---------------------------------------------------------------------------
# Combined plot
# ---------------------------------------------------------------------------


def plot_combined(
    pit_matrices: Dict[str, np.ndarray],
    outdir: Path,
    n_bins: int,
    step_bins: int,
    dpi: int,
    save_pdf: bool,
    annotate: bool = False,
) -> None:
    """One row per model: PIT histogram (1 col) + heatmap (3 cols)."""
    models = sorted(pit_matrices.keys())
    n_models = len(models)
    if n_models == 0:
        print("No probabilistic models to plot.")
        return

    fig = plt.figure(figsize=(LAYOUT.FIG_W, LAYOUT.FIG_H))
    gs = GridSpec(
        n_models,
        4,
        figure=fig,
        width_ratios=LAYOUT.WIDTH_RATIOS,
        wspace=LAYOUT.WSPACE,
        hspace=LAYOUT.HSPACE,
    )

    # --- Pre-compute heatmap grids for shared colour scale ---
    grids: Dict[str, Tuple] = {}
    for model in models:
        density, bin_edges, mad_per_bin = _compute_step_densities(
            pit_matrices[model], n_bins=n_bins, step_bins=step_bins
        )
        grids[model] = (density, bin_edges, mad_per_bin)

    vmin, vmax = LAYOUT.VMIN, LAYOUT.VMAX

    # Pre-compute shared x-axis limit for histogram panels
    hist_xmax = 0.0
    _bin_edges_shared = np.linspace(0.0, 1.0, n_bins + 1)
    for model in models:
        pit_flat = pit_matrices[model].ravel()
        counts, _ = np.histogram(
            pit_flat, bins=_bin_edges_shared, range=(0.0, 1.0), density=False
        )
        hist_density_m = counts / (
            len(pit_flat) * (_bin_edges_shared[1] - _bin_edges_shared[0])
        )
        hist_xmax = max(hist_xmax, float(hist_density_m.max()))
    hist_xmax *= 1.08  # small headroom

    # X-axis geometry for heatmap (minutes)
    steps_per_bin = TOTAL_STEPS // step_bins
    bin_start_steps = np.arange(step_bins) * steps_per_bin
    x_minutes = bin_start_steps * MINUTES_PER_STEP
    x_mid_minutes = x_minutes + steps_per_bin * MINUTES_PER_STEP / 2.0
    x_edges_min = np.append(x_minutes, step_bins * steps_per_bin * MINUTES_PER_STEP)
    hour_ticks = np.arange(0, TOTAL_STEPS * MINUTES_PER_STEP + 1, 60)

    for i, model in enumerate(models):
        color = MODEL_COLORS.get(model, _DEFAULT_COLOR)
        label = MODEL_LABELS.get(model, model.capitalize())
        pit_all = pit_matrices[model]

        # ── Histogram panel (col 0) ─────────────────────────────────────────
        ax_hist = fig.add_subplot(gs[i, 0])

        pit_flat = pit_all.ravel()
        counts, edges = np.histogram(
            pit_flat, bins=n_bins, range=(0.0, 1.0), density=False
        )
        hist_density = counts / (len(pit_flat) * (edges[1] - edges[0]))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bar_width = edges[1] - edges[0]

        # Rotated: PIT value on Y-axis, density on X-axis — matches heatmap orientation
        ax_hist.barh(
            bin_centers,
            hist_density,
            height=bar_width * 0.9,
            color=color,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax_hist.axvline(
            UNIFORM_DENSITY,
            color="#333333",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            zorder=4,
        )

        overall_mad = float(np.mean(np.abs(hist_density - 1.0)))
        ax_hist.text(
            0.97,
            0.04,
            f"MAD={overall_mad:.3f}",
            transform=ax_hist.transAxes,
            fontsize=LAYOUT.KS_FS,
            ha="right",
            va="bottom",
            color="#222",
            bbox=dict(
                boxstyle=f"round,pad={LAYOUT.KS_BOX_PAD}",
                fc="white",
                alpha=0.7,
                ec="none",
            ),
        )

        ax_hist.set_title(
            label,
            fontsize=LAYOUT.MODEL_TITLE_FS,
            fontweight="bold",
            color=color,
            pad=LAYOUT.MODEL_TITLE_PAD,
        )
        ax_hist.set_ylim(0.0, 1.0)
        ax_hist.set_xlim(0.0, hist_xmax)
        ax_hist.invert_xaxis()  # bars grow leftward, adjacent to heatmap
        ax_hist.set_ylabel("PIT value", fontsize=LAYOUT.AXIS_LABEL_FS)
        yticks = np.arange(0.0, 1.1, 0.1)
        ax_hist.set_yticks(yticks)
        ax_hist.set_yticklabels(
            [f"{v:.1f}" for v in yticks], fontsize=LAYOUT.TICK_LABEL_FS
        )
        ax_hist.tick_params(labelsize=LAYOUT.TICK_LABEL_FS)
        ax_hist.grid(True, linestyle=":", alpha=0.35, zorder=0)

        # X label only on last row to reduce clutter
        if i == n_models - 1:
            ax_hist.set_xlabel("Density", fontsize=LAYOUT.AXIS_LABEL_FS)

        # ── Heatmap panel (cols 1–3) ────────────────────────────────────────
        ax_heat = fig.add_subplot(gs[i, 1:])

        density_grid, bin_edges_h, mad_per_bin = grids[model]

        im = ax_heat.pcolormesh(
            x_edges_min,
            bin_edges_h,
            density_grid,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        if annotate:
            # scale annotation font to cell size at the rendered figure dimensions
            annotate_fs = max(2.0, LAYOUT.ANNOTATE_FS_BASE * (32 / step_bins))
            bin_mids = 0.5 * (bin_edges_h[:-1] + bin_edges_h[1:])
            col_mids = x_edges_min[:-1] + 0.5 * np.diff(x_edges_min)
            for ci, cx in enumerate(col_mids):
                for ri, ry in enumerate(bin_mids):
                    ax_heat.text(
                        cx,
                        ry,
                        f"{density_grid[ri, ci]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=annotate_fs,
                        color="black",
                    )

        # Median reference line
        ax_heat.axhline(0.5, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)

        ax_heat.set_ylim(0.0, 1.0)
        yticks = np.arange(0.0, 1.1, 0.1)
        ax_heat.set_yticks(yticks)
        ax_heat.set_yticklabels(
            []
        )  # ticks present but no labels — histogram already shows PIT values
        ax_heat.tick_params(labelsize=LAYOUT.TICK_LABEL_FS)
        ax_heat.grid(False)
        ax_heat.set_xlim(x_edges_min[0], x_edges_min[-1])

        # MAD row above heatmap
        trans = blended_transform_factory(ax_heat.transData, ax_heat.transAxes)
        ax_heat.text(
            0.000,
            LAYOUT.MAD_Y,
            "MAD:",
            ha="right",
            va="bottom",
            fontsize=LAYOUT.MAD_FS,
            transform=ax_heat.transAxes,
            color="#555555",
        )
        for cx, mad_val in zip(x_mid_minutes, mad_per_bin):
            ax_heat.text(
                cx,
                LAYOUT.MAD_Y,
                f"{mad_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=LAYOUT.MAD_FS,
                transform=trans,
                clip_on=False,
                color="#333333",
            )

        # Per-row colourbar
        cbar = fig.colorbar(
            im,
            ax=ax_heat,
            fraction=LAYOUT.CBAR_FRACTION,
            pad=LAYOUT.CBAR_PAD,
            aspect=LAYOUT.CBAR_ASPECT,
        )
        cbar.set_label("PIT density", fontsize=LAYOUT.CBAR_LABEL_FS)
        cbar.ax.tick_params(labelsize=LAYOUT.CBAR_TICK_FS)
        cbar.ax.axhline(
            UNIFORM_DENSITY, color="black", linewidth=0.8, linestyle="--", alpha=0.7
        )

        # X-axis ticks: only on the last row
        if i == n_models - 1:
            ax_heat.set_xlabel(
                "Forecast horizon (minutes)", fontsize=LAYOUT.AXIS_LABEL_FS
            )
            ax_heat.set_xticks(hour_ticks)
            ax_heat.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x)}")
            )
            ax_heat.tick_params(axis="x", labelsize=LAYOUT.TICK_LABEL_FS)
        else:
            ax_heat.set_xticklabels([])
            ax_heat.set_xticks(hour_ticks)

    # Figure title
    n_by_model = {m: pit_matrices[m].shape[0] for m in models}
    unique_ns = set(n_by_model.values())
    base_title = "Overall PIT Histogram  ·  Per-Horizon PIT Density Heatmap"
    n_str = (
        f"n\u202f=\u202f{next(iter(unique_ns)):,} episodes per model"
        if len(unique_ns) == 1
        else ""
    )
    title = f"{base_title}  ({n_str})" if n_str else base_title
    fig.subplots_adjust(top=LAYOUT.SUBPLOT_TOP)
    fig.suptitle(
        title, fontsize=LAYOUT.SUPTITLE_FS, fontweight="bold", y=LAYOUT.SUPTITLE_Y
    )

    _save(fig, outdir / "pit_combined", dpi=dpi, pdf=save_pdf)


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
            f"--step-bins {args.step_bins} does not evenly divide {TOTAL_STEPS}. "
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

    if not pit_matrices:
        print("No probabilistic runs found — nothing to plot.")
        return

    print(f"\nPlotting {len(pit_matrices)} model(s) …")
    plot_combined(
        pit_matrices,
        outdir=outdir,
        n_bins=args.n_bins,
        step_bins=args.step_bins,
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
        annotate=args.annotate,
    )
    print("Done.")


if __name__ == "__main__":
    main()
