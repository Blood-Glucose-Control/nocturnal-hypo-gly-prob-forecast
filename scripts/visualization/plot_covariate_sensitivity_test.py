#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""TFT covariate sensitivity test — synthetic injection analysis.

Tests TFT forecaster behavior under synthetic insulin and carbohydrate
injection scenarios. Generates forecasts with various doses and timings
to assess model sensitivity to physiological covariates.

Usage:
    # Use top-ranked episode from find_stable_episodes.py
    python scripts/visualization/plot_covariate_sensitivity_test.py

    # Specify a different episode
    python scripts/visualization/plot_covariate_sensitivity_test.py \
        --patient-id ale_139 --anchor 2020-08-01

    # Custom dose/timing ranges
    python scripts/visualization/plot_covariate_sensitivity_test.py \
        --insulin-doses 2 4 6 8 10 12 \
        --carb-doses 25 50 75 100 \
        --timings 30 60 90 120
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.physiological.insulin_model.insulin_model import (  # noqa: E402
    calculate_insulin_availability_and_iob_single_delivery,
)
from src.data.physiological.carb_model.constants import TS_MIN, T_ACTION_MAX_MIN  # noqa: E402
from src.models import create_model_and_config  # noqa: E402

matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Default episode (top-ranked from find_stable_episodes.py)
DEFAULT_PATIENT_ID = "ale_15"
DEFAULT_ANCHOR = "2020-06-29"

# Model checkpoint
CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "trained_models"
    / "artifacts"
    / "tft"
    / "2026-04-29_1853_RID20260429_185322_1168187_16_iob_cob_high_lr"
)

# Data directory
DATA_DIR = PROJECT_ROOT / "cache" / "data" / "aleppo_2017" / "processed"

# Output
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_STEM = "covariate_sensitivity_test"
OUTPUT_FORMATS = ["pdf", "png"]
OUTPUT_DPI = 300

# Model parameters
CONTEXT_LENGTH = 512
FORECAST_LENGTH = 96
INTERVAL_MINS = 5
COVARIATE_COLS = ["iob", "cob"]

# Test scenarios
DEFAULT_INSULIN_DOSES = [2.0, 4.0, 8.0, 12.0]  # Units
DEFAULT_CARB_DOSES = [25.0, 50.0, 75.0, 100.0]  # Grams
DEFAULT_INJECTION_TIMING = 60  # Minutes before forecast start
DEFAULT_TIMINGS = [30, 60, 90, 120]  # Minutes for timing sweep

# Quantile levels for probabilistic forecasts
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Figure geometry (from plot_three_model_row.py)
FIGURE_WIDTH_IN = 11.0  # 3-panel layout
FIGURE_HEIGHT_IN = 3.5

SUBPLOT_LEFT = 0.06
SUBPLOT_RIGHT = 0.98
SUBPLOT_TOP = 0.88
SUBPLOT_BOTTOM = 0.15
SUBPLOT_WSPACE = 0.12

# Clinical thresholds
HYPO_THRESHOLD = 3.9  # mmol/L
HYPER_THRESHOLD = 10.0  # mmol/L

# Typography
USE_LATEX = False
FONT_FAMILY = "serif"
FONT_SIZE_BASE = 7
FONT_SIZE_PANEL_TITLE = 8
FONT_SIZE_AXIS_LABEL = 7
FONT_SIZE_TICK = 6
FONT_SIZE_LEGEND = 6

# Line widths
ACTUAL_CTX_LW = 1.1
ACTUAL_FC_LW = 1.1
MEDIAN_LW = 0.9
THRESHOLD_LW = 0.8

# Colors (colorblind-safe palette)
BASELINE_COLOR = "#000000"  # Black
DOSE_COLORS = [
    "#e69f00",  # Orange
    "#56b4e9",  # Sky blue
    "#009e73",  # Bluish green
    "#f0e442",  # Yellow
    "#0072b2",  # Blue
    "#d55e00",  # Vermillion
    "#cc79a7",  # Reddish purple
]

TIMING_COLORS = [
    "#d55e00",  # Vermillion
    "#0072b2",  # Blue
    "#009e73",  # Bluish green
    "#cc79a7",  # Reddish purple
]

# Secondary axis (IOB/COB)
IOB_COB_ALPHA = 0.6
IOB_COB_LW = 0.8

# Context window styling
CONTEXT_SHADE_COLOR = "#f2f2f2"
CONTEXT_SHADE_ALPHA = 1.0
MIDNIGHT_LINE_COLOR = "#aaaaaa"
MIDNIGHT_LINE_LW = 0.7
MIDNIGHT_LINE_LS = ":"

# PI band opacity
ALPHA_80PI = 0.15
ALPHA_50PI = 0.30

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


def load_episode_context(
    patient_id: str,
    anchor: str,
    context_length: int = CONTEXT_LENGTH,
    interval_mins: int = INTERVAL_MINS,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load context window and forecast actuals for an episode.

    Returns:
        context_df: DataFrame with datetime index and BG, insulin, carb columns
        actuals: Ground truth BG in forecast window (96 steps)
    """
    anchor_dt = pd.to_datetime(anchor)
    csv_path = DATA_DIR / f"{patient_id}_full.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Patient CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")

    # Extract context window (ending just before forecast)
    ctx_end = anchor_dt - pd.Timedelta(minutes=interval_mins)
    ctx_start = ctx_end - pd.Timedelta(minutes=(context_length - 1) * interval_mins)
    context_df = df.loc[ctx_start:ctx_end].copy()

    # Extract forecast window actuals
    fc_start = anchor_dt
    fc_end = anchor_dt + pd.Timedelta(minutes=(FORECAST_LENGTH - 1) * interval_mins)
    forecast_df = df.loc[fc_start:fc_end]
    actuals = forecast_df["bg_mM"].values

    return context_df, actuals


def create_synthetic_insulin_scenario(
    context_df: pd.DataFrame,
    dose_units: float,
    injection_time_mins: int,
    interval_mins: int = INTERVAL_MINS,
) -> pd.DataFrame:
    """Create modified context with synthetic insulin injection.

    Args:
        context_df: Original context DataFrame
        dose_units: Insulin dose in units
        injection_time_mins: Minutes before forecast start to inject

    Returns:
        Modified context DataFrame with updated dose_units, iob, insulin_availability
    """
    ctx = context_df.copy()

    # Calculate injection index (from end of context)
    inj_steps_before = injection_time_mins // interval_mins
    if inj_steps_before >= len(ctx):
        raise ValueError(
            f"Injection time {injection_time_mins}min exceeds context length"
        )

    inj_idx = len(ctx) - inj_steps_before

    # Add synthetic insulin dose at specified time
    ctx.iloc[inj_idx, ctx.columns.get_loc("dose_units")] += dose_units

    # Calculate IOB and insulin availability from synthetic dose
    ins_avail, iob, _, _, _ = calculate_insulin_availability_and_iob_single_delivery(
        dose_units, TS_MIN, T_ACTION_MAX_MIN
    )

    # Add IOB/insulin_availability from injection point forward (on top of existing)
    for i in range(inj_idx, len(ctx)):
        time_since_inj = (i - inj_idx) * interval_mins
        if time_since_inj < T_ACTION_MAX_MIN:
            idx_in_curve = time_since_inj // TS_MIN
            ctx.iloc[i, ctx.columns.get_loc("iob")] += iob[idx_in_curve]
            ctx.iloc[i, ctx.columns.get_loc("insulin_availability")] += ins_avail[
                idx_in_curve
            ]

    return ctx


def create_synthetic_carb_scenario(
    context_df: pd.DataFrame,
    carb_grams: float,
    injection_time_mins: int,
    interval_mins: int = INTERVAL_MINS,
) -> pd.DataFrame:
    """Create modified context with synthetic carbohydrate intake.

    Args:
        context_df: Original context DataFrame
        carb_grams: Carbohydrate amount in grams
        injection_time_mins: Minutes before forecast start to inject

    Returns:
        Modified context DataFrame with updated food_g, cob, carb_availability
    """
    ctx = context_df.copy()

    # Calculate injection index
    inj_steps_before = injection_time_mins // interval_mins
    if inj_steps_before >= len(ctx):
        raise ValueError(
            f"Injection time {injection_time_mins}min exceeds context length"
        )

    inj_idx = len(ctx) - inj_steps_before

    # Add synthetic carb dose at specified time (on top of existing)
    ctx.iloc[inj_idx, ctx.columns.get_loc("food_g")] += carb_grams

    # Recalculate COB and carb_availability using simplified model
    # (Carb absorption is faster than insulin: peak ~30-60min, decay over 3-4h)
    # Using exponential decay model for simplicity
    CARB_TMAX_MIN = 45  # Peak absorption time
    CARB_DECAY_HALFLIFE_MIN = 90  # Half-life after peak

    for i in range(inj_idx, len(ctx)):
        time_since_inj = (i - inj_idx) * interval_mins

        if time_since_inj <= CARB_TMAX_MIN:
            # Rising phase: linear ramp to peak
            frac = time_since_inj / CARB_TMAX_MIN
            cob = carb_grams * frac
        else:
            # Decay phase: exponential decay
            time_after_peak = time_since_inj - CARB_TMAX_MIN
            decay_frac = 2 ** (-time_after_peak / CARB_DECAY_HALFLIFE_MIN)
            cob = carb_grams * decay_frac

        ctx.iloc[i, ctx.columns.get_loc("cob")] += cob  # ADD to existing
        if "carb_availability" in ctx.columns:
            # Carb availability peaks earlier and decays faster
            avail_frac = np.exp(-time_since_inj / 60.0)  # Exponential decay
            ctx.iloc[i, ctx.columns.get_loc("carb_availability")] += (
                carb_grams * avail_frac
            )  # ADD to existing

    return ctx


def predict_scenario(
    model,
    context_df: pd.DataFrame,
    patient_id: str,
    scenario_id: str,
    debug: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate forecast for a single scenario.

    Returns:
        Dict with keys "quantiles" (9, 96), "median" (96,)
    """
    # Prepare context for model
    ctx = context_df.reset_index().copy()
    ctx["p_num"] = patient_id
    ctx["episode_id"] = scenario_id

    # Ensure required columns exist
    if "datetime" not in ctx.columns and ctx.index.name == "datetime":
        ctx = ctx.reset_index()

    if debug:
        print(f"    [DEBUG] Scenario: {scenario_id}")
        print(f"    [DEBUG] Columns: {list(ctx.columns)}")
        print(f"    [DEBUG] IOB range: {ctx['iob'].min():.3f} - {ctx['iob'].max():.3f}")
        print(f"    [DEBUG] COB range: {ctx['cob'].min():.3f} - {ctx['cob'].max():.3f}")
        print(f"    [DEBUG] Last 5 IOB values: {ctx['iob'].tail().values}")
        print(f"    [DEBUG] Last 5 COB values: {ctx['cob'].tail().values}")
        print(
            f"    [DEBUG] BG range: {ctx['bg_mM'].min():.3f} - {ctx['bg_mM'].max():.3f}"
        )

    # Call model predict_batch
    results = model.predict_batch(
        ctx,
        episode_col="episode_id",
        quantile_levels=QUANTILE_LEVELS,
    )

    if scenario_id not in results:
        raise ValueError(f"Model did not return forecast for {scenario_id}")

    q_forecast = results[scenario_id]  # Shape: (9, 96)
    median_idx = QUANTILE_LEVELS.index(0.5)
    median_forecast = q_forecast[median_idx]

    if debug:
        print(
            f"    [DEBUG] Forecast range: {median_forecast.min():.3f} - {median_forecast.max():.3f} mmol/L"
        )
        print(
            f"    [DEBUG] First/last forecast: {median_forecast[0]:.3f} / {median_forecast[-1]:.3f} mmol/L"
        )

    return {
        "quantiles": q_forecast,
        "median": median_forecast,
    }


def generate_all_scenarios(
    model,
    context_df: pd.DataFrame,
    patient_id: str,
    insulin_doses: List[float],
    carb_doses: List[float],
    timings: List[int],
    injection_timing: int,
) -> Dict[str, Dict]:
    """Generate forecasts for all scenarios.

    Returns:
        Dict[scenario_name, Dict["quantiles": array, "median": array, "context": DataFrame]]
    """
    scenarios = {}

    # Baseline (no modifications)
    print("Generating baseline scenario...")
    baseline_pred = predict_scenario(
        model, context_df, patient_id, "baseline", debug=True
    )
    scenarios["baseline"] = {
        **baseline_pred,
        "context": context_df.copy(),
        "label": "Baseline",
        "color": BASELINE_COLOR,
    }
    print(
        f"  ✓ Baseline complete (median forecast: {baseline_pred['median'][0]:.2f} - {baseline_pred['median'][-1]:.2f} mmol/L)"
    )

    # Insulin dose sweep (fixed timing)
    print(f"\nGenerating insulin dose sweep @ {injection_timing} min...")
    for i, dose in enumerate(insulin_doses):
        scenario_id = f"ins_{dose:.0f}U_{injection_timing}min"
        print(f"  {scenario_id}...", end=" ")
        ctx = create_synthetic_insulin_scenario(context_df, dose, injection_timing)
        pred = predict_scenario(model, ctx, patient_id, scenario_id, debug=True)
        scenarios[scenario_id] = {
            **pred,
            "context": ctx,
            "label": f"{dose:.0f}U",
            "color": DOSE_COLORS[i % len(DOSE_COLORS)],
            "dose": dose,
            "timing": injection_timing,
        }
        print(f"{dose:.0f}U ✓")

    # Insulin timing sweep (fixed dose = 4U)
    print("\nGenerating insulin timing sweep (4U)...")
    for i, timing in enumerate(timings):
        scenario_id = f"ins_timing_4U_{timing}min"
        print(f"  {scenario_id}...", end=" ")
        ctx = create_synthetic_insulin_scenario(context_df, 4.0, timing)
        pred = predict_scenario(model, ctx, patient_id, scenario_id, debug=True)
        scenarios[scenario_id] = {
            **pred,
            "context": ctx,
            "label": f"{timing} min",
            "color": TIMING_COLORS[i % len(TIMING_COLORS)],
            "dose": 4.0,
            "timing": timing,
        }
        print(f"{timing} min ✓")

    # Carb dose sweep (fixed timing)
    print(f"\nGenerating carb dose sweep @ {injection_timing} min...")
    for i, dose in enumerate(carb_doses):
        scenario_id = f"carb_{dose:.0f}g_{injection_timing}min"
        print(f"  {scenario_id}...", end=" ")
        ctx = create_synthetic_carb_scenario(context_df, dose, injection_timing)
        pred = predict_scenario(model, ctx, patient_id, scenario_id, debug=True)
        scenarios[scenario_id] = {
            **pred,
            "context": ctx,
            "label": f"{dose:.0f}g",
            "color": DOSE_COLORS[i % len(DOSE_COLORS)],
            "dose": dose,
            "timing": injection_timing,
        }
        print(f"{dose:.0f}g ✓")

    print(f"\nGenerated {len(scenarios)} scenarios total")
    return scenarios


def plot_three_panel_figure(
    scenarios: Dict[str, Dict],
    actuals: np.ndarray,
    output_dir: Path,
    output_stem: str,
) -> None:
    """Create three-panel visualization (insulin dose, timing, carb dose)."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        sharey=True,
    )

    plt.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
        bottom=SUBPLOT_BOTTOM,
        wspace=SUBPLOT_WSPACE,
    )

    # Time axes (hours relative to midnight)
    n_ctx = CONTEXT_LENGTH
    n_fc = FORECAST_LENGTH
    ctx_t = np.arange(-n_ctx, 0) * INTERVAL_MINS / 60.0
    fc_t = np.arange(n_fc) * INTERVAL_MINS / 60.0

    # Extract baseline BG for context
    baseline_bg = scenarios["baseline"]["context"]["bg_mM"].values
    print(
        f"Scenarios: {list(scenarios.keys())}\n"
        f"Scenario labels/colors: {[ (s['label'], s['color']) for s in scenarios.values() ]}\n"
    )
    # Panel 1: Insulin dose sweep (no baseline, sorted by dose)
    panel_scenarios_1 = sorted(
        [
            k
            for k in scenarios.keys()
            if k.startswith("ins_")
            and not k.startswith("ins_timing_")
            and "_60min" in k
        ],
        key=lambda x: float(x.split("_")[1].replace("U", "")),
    )
    plot_panel(
        axes[0],
        scenarios,
        panel_scenarios_1,
        actuals,
        baseline_bg,
        ctx_t,
        fc_t,
        title="Insulin Dose Sweep (60 min)",
        ylabel="Blood Glucose (mmol/L)",
        show_iob=True,
    )

    # Panel 2: Insulin timing sweep (sorted by timing)
    panel_scenarios_2 = sorted(
        [k for k in scenarios.keys() if k.startswith("ins_timing_4U_")],
        key=lambda x: int(x.split("_")[-1].replace("min", "")),
    )
    plot_panel(
        axes[1],
        scenarios,
        panel_scenarios_2,
        actuals,
        baseline_bg,
        ctx_t,
        fc_t,
        title="Insulin Timing Sweep (4U)",
        ylabel=None,
        show_iob=True,
    )

    # Panel 3: Carb dose sweep (no baseline)
    panel_scenarios_3 = [k for k in scenarios.keys() if k.startswith("carb_")]
    plot_panel(
        axes[2],
        scenarios,
        panel_scenarios_3,
        actuals,
        baseline_bg,
        ctx_t,
        fc_t,
        title="Carbohydrate Sweep (60 min)",
        ylabel=None,
        show_iob=False,  # Show COB instead
    )

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        out_path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(
            out_path, dpi=OUTPUT_DPI if fmt == "png" else None, bbox_inches="tight"
        )
        print(f"Saved: {out_path}")

    plt.close(fig)


def write_debug_outputs(
    scenarios: Dict[str, Dict],
    actuals: np.ndarray,
    output_dir: Path,
    output_stem: str,
    interval_mins: int = INTERVAL_MINS,
) -> None:
    """Write last 2 hours of model input and forecast medians per scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    last_steps = int(120 / interval_mins)

    for scenario_id, scenario in scenarios.items():
        ctx = scenario["context"].copy()
        ctx_last = ctx.tail(last_steps).reset_index()
        ctx_path = output_dir / f"{output_stem}_{scenario_id}_context_last2h.csv"
        ctx_last.to_csv(ctx_path, index=False)

        forecast = np.asarray(scenario["median"], dtype=float)
        forecast_steps = np.arange(len(forecast))
        forecast_df = pd.DataFrame(
            {
                "step": forecast_steps,
                "t_min": forecast_steps * interval_mins,
                "t_hr": (forecast_steps * interval_mins) / 60.0,
                "forecast_median": forecast,
                "actual_bg": actuals[: len(forecast)],
            }
        )
        forecast_path = output_dir / f"{output_stem}_{scenario_id}_forecast.csv"
        forecast_df.to_csv(forecast_path, index=False)


def plot_panel(
    ax: plt.Axes,
    scenarios: Dict[str, Dict],
    scenario_keys: List[str],
    actuals: np.ndarray,
    baseline_bg: np.ndarray,
    ctx_t: np.ndarray,
    fc_t: np.ndarray,
    title: str,
    ylabel: str = None,
    show_iob: bool = True,
) -> None:
    """Plot one panel of the three-panel figure."""
    # Context background
    ax.axvspan(
        ctx_t[0],
        0.0,
        color=CONTEXT_SHADE_COLOR,
        alpha=CONTEXT_SHADE_ALPHA,
        zorder=0,
        label="_nolegend_",
    )

    # Midnight separator
    ax.axvline(
        0.0,
        color=MIDNIGHT_LINE_COLOR,
        lw=MIDNIGHT_LINE_LW,
        ls=MIDNIGHT_LINE_LS,
        zorder=1,
        label="_nolegend_",
    )

    # Clinical thresholds
    ax.axhline(
        HYPER_THRESHOLD,
        color="#ff7f0e",
        lw=THRESHOLD_LW,
        ls="-",
        alpha=0.9,
        zorder=2,
        label="_nolegend_",
    )
    ax.axhline(
        HYPO_THRESHOLD,
        color="#d62728",
        lw=THRESHOLD_LW,
        ls="-",
        alpha=0.9,
        zorder=2,
        label="_nolegend_",
    )

    # Actual BG context (black solid)
    ax.plot(
        ctx_t,
        baseline_bg,
        color="black",
        lw=ACTUAL_CTX_LW,
        zorder=6,
        label="Actual BG",
    )

    # Actual BG forecast (black solid)
    ax.plot(
        fc_t,
        actuals,
        color="black",
        lw=ACTUAL_FC_LW,
        zorder=6,
        label="_nolegend_",
    )

    # Plot forecasts for each scenario (median only, no quantiles)
    for scenario_key in scenario_keys:
        scenario = scenarios[scenario_key]
        median_fc = scenario["median"]
        color = scenario["color"]
        label = scenario["label"]

        # Plot median forecast only
        ax.plot(
            fc_t,
            median_fc,
            color=color,
            lw=MEDIAN_LW,
            zorder=5,
            label=label,
        )

    # Formatting
    ax.set_xlim(ctx_t[0], fc_t[-1])
    ax.set_ylim(0, 12.0)
    ax.set_title(title, fontsize=FONT_SIZE_PANEL_TITLE, pad=3)
    ax.set_xlabel("Time of day (h)", fontsize=FONT_SIZE_AXIS_LABEL)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXIS_LABEL)

    # X-axis ticks (every 2 hours)
    t_start = int(ctx_t[0])
    t_end = int(np.ceil(fc_t[-1]))
    major_ticks = np.arange(t_start, t_end + 1, 2)
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(int(t % 24)) for t in major_ticks])

    # Legend
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )


def main():
    parser = argparse.ArgumentParser(
        description="TFT covariate sensitivity test with synthetic injections"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=DEFAULT_PATIENT_ID,
        help="Patient ID",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default=DEFAULT_ANCHOR,
        help="Episode anchor (midnight date, YYYY-MM-DD)",
    )
    parser.add_argument(
        "--insulin-doses",
        type=float,
        nargs="+",
        default=DEFAULT_INSULIN_DOSES,
        help="Insulin doses for dose sweep (units)",
    )
    parser.add_argument(
        "--carb-doses",
        type=float,
        nargs="+",
        default=DEFAULT_CARB_DOSES,
        help="Carbohydrate doses for carb sweep (grams)",
    )
    parser.add_argument(
        "--timings",
        type=int,
        nargs="+",
        default=DEFAULT_TIMINGS,
        help="Injection timings for timing sweep (minutes before forecast)",
    )
    parser.add_argument(
        "--injection-timing",
        type=int,
        default=DEFAULT_INJECTION_TIMING,
        help="Fixed injection timing for dose sweeps (minutes before forecast)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename stem (default: covariate_sensitivity_test)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization (useful for debugging scenarios)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TFT COVARIATE SENSITIVITY TEST")
    print("=" * 80)
    print(f"Episode: {args.patient_id} @ {args.anchor}")
    print(f"Checkpoint: {CHECKPOINT_DIR.name}")
    print()

    # Apply plotting style
    _apply_rcparams()

    # Load model
    print("Loading TFT model...")
    model, config = create_model_and_config(
        model_type="tft",
        checkpoint=str(CHECKPOINT_DIR),
        context_length=CONTEXT_LENGTH,
        forecast_length=FORECAST_LENGTH,
        covariate_cols=COVARIATE_COLS,
    )
    print(f"  Model loaded from {CHECKPOINT_DIR}")
    print()

    # Load episode context
    print("Loading episode context...")
    context_df, actuals = load_episode_context(
        args.patient_id,
        args.anchor,
        context_length=CONTEXT_LENGTH,
    )
    print(f"  Context: {len(context_df)} steps")
    print(f"  Actuals: {len(actuals)} steps")
    print(f"  BG at forecast start (t=0): {actuals[0]:.2f} mmol/L")
    print(f"  Context BG last value: {context_df['bg_mM'].iloc[-1]:.2f} mmol/L")
    print()

    # Generate scenarios and forecasts
    print("Generating scenarios and forecasts...")
    print("=" * 80)
    scenarios = generate_all_scenarios(
        model=model,
        context_df=context_df,
        patient_id=args.patient_id,
        insulin_doses=args.insulin_doses,
        carb_doses=args.carb_doses,
        timings=args.timings,
        injection_timing=args.injection_timing,
    )
    print("=" * 80)
    print()

    output_stem = args.output or OUTPUT_STEM

    # Write debug outputs
    write_debug_outputs(
        scenarios=scenarios,
        actuals=actuals,
        output_dir=OUTPUT_DIR,
        output_stem=output_stem,
    )

    # Create visualization
    if not args.skip_viz:
        print("Creating visualization...")
        plot_three_panel_figure(
            scenarios=scenarios,
            actuals=actuals,
            output_dir=OUTPUT_DIR,
            output_stem=output_stem,
        )
        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)
    else:
        print("Skipping visualization (--skip-viz)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
