#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""Synthetic CGM variance visualization.

Creates a single-panel 24-hour synthetic CGM figure with three traces that share
an equal mean but have different variances (low, medium, high). Each trace has
three meal-related spikes and includes clinical threshold overlays.

Usage:
    # Run with defaults
    python scripts/visualization/plot_synthetic_cgm_variance_analysis.py

    # Custom output stem and seed
    python scripts/visualization/plot_synthetic_cgm_variance_analysis.py \
        --output my_synthetic_cgm_plot --seed 42

    # Custom meal times and target mean
    python scripts/visualization/plot_synthetic_cgm_variance_analysis.py \
        --meal-times 7 12 18 --target-mean 6.8
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Output
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_STEM = "synthetic_cgm_variance_analysis"
OUTPUT_FORMATS = ["pdf", "png"]
OUTPUT_DPI = 300

# Timeline
INTERVAL_MINS = 5
TOTAL_HOURS = 24
MEAL_TIMES_HOURS = [8.0, 13.0, 19.0]

# Synthetic glucose shape
TARGET_MEAN_MMOL = 6.8
CIRCADIAN_AMPLITUDE = 0.35
CIRCADIAN_PHASE_SHIFT_HR = 4.0
MEAL_SPIKE_AMPLITUDES = [2.1, 2.4, 2.2]
MEAL_SPIKE_WIDTH_HR = 0.75

# Variance settings (meal-response driven, not noise-driven)
LOW_RESPONSE_SCALE = 0.55
MEDIUM_RESPONSE_SCALE = 1.25
HIGH_RESPONSE_SCALE = 2.15

# Meal rebound dip settings (post-prandial undershoot)
REBOUND_DELAY_HR = 2.2
REBOUND_WIDTH_HR = 1.0
REBOUND_AMPLITUDE_RATIO = 0.52

# Trace constraints
HIGH_TRACE_MIN_TARGET = 3.5
HIGH_TRACE_MAX_TARGET = 10.4
TRACE_FLOOR = 2.5
TRACE_CEIL = 14.5

# Figure geometry
FIGURE_WIDTH_IN = 5.5
FIGURE_HEIGHT_IN = 2.9

SUBPLOT_LEFT = 0.12
SUBPLOT_RIGHT = 0.98
SUBPLOT_TOP = 0.90
SUBPLOT_BOTTOM = 0.43

# Clinical thresholds
HYPO_THRESHOLD = 3.9
HYPER_THRESHOLD = 10.0

# Typography
USE_LATEX = False
FONT_FAMILY = "serif"
FONT_SIZE_BASE = 7
FONT_SIZE_PANEL_TITLE = 8
FONT_SIZE_AXIS_LABEL = 7
FONT_SIZE_TICK = 6
FONT_SIZE_LEGEND = 6

# Line widths
TRACE_LW = 1.1
THRESHOLD_LW = 0.8
MEAN_LINE_LW = 0.9

# Colors
LOW_COLOR = "#0072b2"  # Blue
MEDIUM_COLOR = "#f0e442"  # Yellow
HIGH_COLOR = "#d62728"  # Red
MEAN_LINE_COLOR = "#4d4d4d"

# ══════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ══════════════════════════════════════════════════════════════════════════════


def _apply_rcparams() -> None:
    """Apply matplotlib rcParams for publication-quality figures."""
    if USE_LATEX:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{times}"

    matplotlib.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_PANEL_TITLE,
            "axes.labelsize": FONT_SIZE_AXIS_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def _gaussian_pulse(
    time_h: np.ndarray, center_h: float, width_h: float, amplitude: float
) -> np.ndarray:
    """Create a Gaussian pulse centered at center_h."""
    return amplitude * np.exp(-0.5 * ((time_h - center_h) / width_h) ** 2)


def generate_base_cgm_trace(
    time_h: np.ndarray,
    meal_times_h: List[float],
    target_mean: float,
) -> np.ndarray:
    """Generate a smooth base CGM profile with circadian drift and meal spikes."""
    base = target_mean + CIRCADIAN_AMPLITUDE * np.sin(
        2.0 * np.pi * (time_h - CIRCADIAN_PHASE_SHIFT_HR) / TOTAL_HOURS
    )

    return base


def generate_meal_response_component(
    time_h: np.ndarray,
    meal_times_h: List[float],
    response_scale: float,
) -> np.ndarray:
    """Build a meal response with peaks and delayed rebound dips.

    Variance differences are driven by response_scale (meal dynamics), not noise.
    """
    response = np.zeros_like(time_h, dtype=float)

    for meal_time_h, base_amp in zip(meal_times_h, MEAL_SPIKE_AMPLITUDES):
        pos_amp = base_amp * response_scale
        neg_amp = pos_amp * REBOUND_AMPLITUDE_RATIO

        response += _gaussian_pulse(
            time_h,
            meal_time_h,
            MEAL_SPIKE_WIDTH_HR,
            pos_amp,
        )
        response -= _gaussian_pulse(
            time_h,
            meal_time_h + REBOUND_DELAY_HR,
            REBOUND_WIDTH_HR,
            neg_amp,
        )

    return response


def _align_to_mean(trace: np.ndarray, target_mean: float) -> np.ndarray:
    """Shift trace to have a desired mean without changing variance shape."""
    return trace - np.mean(trace) + target_mean


def _add_local_adjustment(
    trace: np.ndarray,
    time_h: np.ndarray,
    center_h: float,
    amplitude: float,
    width_h: float,
) -> np.ndarray:
    """Add a smooth local Gaussian adjustment around center_h."""
    return trace + _gaussian_pulse(time_h, center_h, width_h, amplitude)


def _enforce_high_variance_constraints(
    high_trace: np.ndarray,
    time_h: np.ndarray,
    target_mean: float,
) -> np.ndarray:
    """Ensure high-variance trace crosses both hyper and hypo thresholds."""
    trace = high_trace.copy()

    # Use smooth local adjustments near meal windows to avoid abrupt overnight cliffs.
    lunch_peak_h = 13.0
    afternoon_dip_h = 15.2

    trace = np.clip(trace, TRACE_FLOOR, TRACE_CEIL)
    trace = _align_to_mean(trace, target_mean)

    max_val = float(np.max(trace))
    min_val = float(np.min(trace))

    if max_val < HIGH_TRACE_MAX_TARGET:
        trace = _add_local_adjustment(
            trace,
            time_h,
            center_h=lunch_peak_h,
            amplitude=HIGH_TRACE_MAX_TARGET - max_val,
            width_h=0.50,
        )
    if min_val > HIGH_TRACE_MIN_TARGET:
        trace = _add_local_adjustment(
            trace,
            time_h,
            center_h=afternoon_dip_h,
            amplitude=HIGH_TRACE_MIN_TARGET - min_val,
            width_h=0.65,
        )

    trace = np.clip(trace, TRACE_FLOOR, TRACE_CEIL)
    trace = _align_to_mean(trace, target_mean)

    if np.max(trace) <= HYPER_THRESHOLD:
        trace = _add_local_adjustment(
            trace,
            time_h,
            center_h=lunch_peak_h,
            amplitude=(HYPER_THRESHOLD + 0.20) - float(np.max(trace)),
            width_h=0.45,
        )
    if np.min(trace) >= HYPO_THRESHOLD:
        trace = _add_local_adjustment(
            trace,
            time_h,
            center_h=afternoon_dip_h,
            amplitude=(HYPO_THRESHOLD - 0.20) - float(np.min(trace)),
            width_h=0.60,
        )

    trace = np.clip(trace, TRACE_FLOOR, TRACE_CEIL)
    trace = _align_to_mean(trace, target_mean)

    return trace


def generate_variance_traces(
    interval_mins: int,
    target_mean: float,
    meal_times_h: List[float],
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """Generate low/medium/high variance CGM traces over 24h.

    Variance is controlled by meal response dynamics (spike/rebound magnitude).
    """
    n_steps = int((TOTAL_HOURS * 60) / interval_mins)
    time_h = np.arange(n_steps) * interval_mins / 60.0

    base = generate_base_cgm_trace(time_h, meal_times_h, target_mean)

    low = base + generate_meal_response_component(
        time_h, meal_times_h, LOW_RESPONSE_SCALE
    )
    medium = base + generate_meal_response_component(
        time_h, meal_times_h, MEDIUM_RESPONSE_SCALE
    )
    high = base + generate_meal_response_component(
        time_h, meal_times_h, HIGH_RESPONSE_SCALE
    )

    low = np.clip(_align_to_mean(low, target_mean), TRACE_FLOOR, TRACE_CEIL)
    medium = np.clip(_align_to_mean(medium, target_mean), TRACE_FLOOR, TRACE_CEIL)
    high = np.clip(_align_to_mean(high, target_mean), TRACE_FLOOR, TRACE_CEIL)

    high = _enforce_high_variance_constraints(high, time_h, target_mean)

    low = _align_to_mean(low, target_mean)
    medium = _align_to_mean(medium, target_mean)
    high = _align_to_mean(high, target_mean)

    traces = {
        "low": {
            "label": "Low Variance",
            "color": LOW_COLOR,
            "values": low,
        },
        "medium": {
            "label": "Medium Variance",
            "color": MEDIUM_COLOR,
            "values": medium,
        },
        "high": {
            "label": "High Variance",
            "color": HIGH_COLOR,
            "values": high,
        },
    }

    return time_h, traces


def plot_single_panel_cgm_variance(
    time_h: np.ndarray,
    traces: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    output_stem: str,
) -> None:
    """Plot one panel with low/medium/high synthetic CGM traces."""
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    plt.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        bottom=SUBPLOT_BOTTOM,
    )

    ax.axhline(
        HYPER_THRESHOLD,
        color="#ff7f0e",
        lw=THRESHOLD_LW,
        ls="-",
        alpha=0.9,
        zorder=1,
        label="Hyper (10.0)",
    )
    ax.axhline(
        HYPO_THRESHOLD,
        color="#d62728",
        lw=THRESHOLD_LW,
        ls="-",
        alpha=0.9,
        zorder=1,
        label="Hypo (3.9)",
    )

    mean_level = float(
        np.mean([np.mean(traces[key]["values"]) for key in ["low", "medium", "high"]])
    )
    ax.axhline(
        mean_level,
        color=MEAN_LINE_COLOR,
        lw=MEAN_LINE_LW,
        ls="--",
        alpha=0.95,
        zorder=1,
        label=f"Mean ({mean_level:.1f})",
    )

    for trace_key in ["low", "medium", "high"]:
        trace = traces[trace_key]
        ax.plot(
            time_h,
            trace["values"],
            color=trace["color"],
            lw=TRACE_LW,
            zorder=3,
            label=trace["label"],
        )

    ax.set_xlim(0.0, TOTAL_HOURS)
    ax.set_ylim(2.5, 12.0)
    ax.set_title("Synthetic CGM Traces With Matched Mean and Different Variance", pad=3)
    ax.set_xlabel("Time of day (h)")
    ax.set_ylabel("Blood Glucose (mmol/L)")

    major_ticks = np.arange(0, TOTAL_HOURS + 1, 2)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{int(t):02d}" for t in major_ticks])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=len(labels),
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        out_path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(
            out_path, dpi=OUTPUT_DPI if fmt == "png" else None, bbox_inches="tight"
        )
        print(f"Saved: {out_path}")

    plt.close(fig)


def summarize_traces(traces: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Print summary statistics for quick validation."""
    print("Trace summary statistics:")
    for key in ["low", "medium", "high"]:
        values = traces[key]["values"]
        print(
            f"  {traces[key]['label']:<15} "
            f"mean={np.mean(values):5.2f} "
            f"std={np.std(values):5.2f} "
            f"min={np.min(values):5.2f} "
            f"max={np.max(values):5.2f}"
        )

    high_vals = traces["high"]["values"]
    crosses_hyper = np.any(high_vals > HYPER_THRESHOLD)
    crosses_hypo = np.any(high_vals < HYPO_THRESHOLD)
    print(
        "High-variance trace checks: "
        f"> {HYPER_THRESHOLD}={crosses_hyper}, < {HYPO_THRESHOLD}={crosses_hypo}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a single-panel synthetic CGM variance visualization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename stem (default: synthetic_cgm_variance_analysis)",
    )
    parser.add_argument(
        "--interval-mins",
        type=int,
        default=INTERVAL_MINS,
        help="Sampling interval in minutes",
    )
    parser.add_argument(
        "--target-mean",
        type=float,
        default=TARGET_MEAN_MMOL,
        help="Target mean glucose (mmol/L) for all traces",
    )
    parser.add_argument(
        "--meal-times",
        type=float,
        nargs=3,
        default=MEAL_TIMES_HOURS,
        help="Three meal times in hours (e.g., 8 13 19)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SYNTHETIC CGM VARIANCE VISUALIZATION")
    print("=" * 80)
    print(f"Interval: {args.interval_mins} min")
    print(f"Target mean: {args.target_mean:.2f} mmol/L")
    print(f"Meal times: {args.meal_times}")
    print()

    _apply_rcparams()

    time_h, traces = generate_variance_traces(
        interval_mins=args.interval_mins,
        target_mean=args.target_mean,
        meal_times_h=args.meal_times,
    )

    summarize_traces(traces)
    print()

    output_stem = args.output or OUTPUT_STEM
    plot_single_panel_cgm_variance(
        time_h=time_h,
        traces=traces,
        output_dir=OUTPUT_DIR,
        output_stem=output_stem,
    )

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
