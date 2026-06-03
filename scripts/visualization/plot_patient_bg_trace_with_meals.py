#!/usr/bin/env python3
"""
Patient Blood Glucose Trace with Meal Bounding Boxes

Visualize a single patient's BG trace over a 16-hour period (6AM-10PM) with
configurable bounding boxes to highlight meal spikes for paper figures.

Usage:
    python scripts/visualization/plot_patient_bg_trace_with_meals.py
    python scripts/visualization/plot_patient_bg_trace_with_meals.py --patient_id ale_42
    python scripts/visualization/plot_patient_bg_trace_with_meals.py --date 2017-03-15
    python scripts/visualization/plot_patient_bg_trace_with_meals.py --show_patient_stats
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

# ── CONFIGURATION ───────────────────────────────────────────────────────────
# All parameters for the visualization are defined here. Edit these values to
# customize the plot without modifying the code below.
# ────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -- Patient Selection -------------------------------------------------------
# Specify a patient ID or leave as None to auto-select from low-mean-BG patients
CONFIGURATION_PATIENT_ID: str | None = (
    "ale_79"  # e.g., "ale_42" or None for auto-select
)
CONFIGURATION_DATE: str | None = None  # e.g., "2017-03-15" or None for random date
CONFIGURATION_SEED: int = 42  # Random seed for reproducibility
CONFIGURATION_NUM_CANDIDATES: int = 20  # Number of low-mean-BG patients to consider

# -- Time Window -------------------------------------------------------------
CONFIGURATION_START_HOUR: int = 6  # Start time (24-hour format, 0-23)
CONFIGURATION_END_HOUR: int = 22  # End time (24-hour format, 0-23)

# -- Y-Axis Range ------------------------------------------------------------
CONFIGURATION_Y_MIN: float = 0.0  # mmol/L
CONFIGURATION_Y_MAX: float = 13.3  # mmol/L

# -- Bounding Box 1 ----------------------------------------------------------
CONFIGURATION_BOX1_X_START: float = 1.5  # Hours from start time (0-16)
CONFIGURATION_BOX1_WIDTH: float = 4.5  # Hours
CONFIGURATION_BOX1_Y_BOTTOM: float = 5.0  # mmol/L
CONFIGURATION_BOX1_HEIGHT: float = 4.25  # mmol/L

# -- Bounding Box 2 ----------------------------------------------------------
CONFIGURATION_BOX2_X_START: float = 7.0  # Hours from start time (0-16)
CONFIGURATION_BOX2_WIDTH: float = 3.5  # Hours
CONFIGURATION_BOX2_Y_BOTTOM: float = 4.0  # mmol/L
CONFIGURATION_BOX2_HEIGHT: float = 2.25  # mmol/L

# -- Bounding Box 3 ----------------------------------------------------------
CONFIGURATION_BOX3_X_START: float = 11.5  # Hours from start time (0-16)
CONFIGURATION_BOX3_WIDTH: float = 4.0  # Hours
CONFIGURATION_BOX3_Y_BOTTOM: float = 3.0  # mmol/L
CONFIGURATION_BOX3_HEIGHT: float = 6.25  # mmol/L

# -- Visual Styling ----------------------------------------------------------
CONFIGURATION_LINE_COLOR: str = "#1565c0"  # BG trace line color (blue)
CONFIGURATION_LINE_WIDTH: float = 1.5  # BG trace line width (pt)
CONFIGURATION_MARKER_SIZE: float = 3.0  # BG trace marker size (pt)
CONFIGURATION_MARKER_STYLE: str = "o"  # BG trace marker style

CONFIGURATION_BOX_EDGE_COLOR: str = "#d32f2f"  # Bounding box edge color (red)
CONFIGURATION_BOX_LINE_WIDTH: float = 2.0  # Bounding box line width (pt)
CONFIGURATION_BOX_LINE_STYLE: str = "--"  # Bounding box line style
CONFIGURATION_BOX_FILL_ALPHA: float = 0.25  # Bounding box fill transparency
CONFIGURATION_BOX_FILL_COLOR: str = "#d32f2f"  # Bounding box fill color (red)

CONFIGURATION_GRID: bool = True  # Show grid
CONFIGURATION_GRID_ALPHA: float = 0.3  # Grid transparency
CONFIGURATION_GRID_COLOR: str = "#cccccc"  # Grid color

# -- Reference Lines ---------------------------------------------------------
CONFIGURATION_HYPO_LINE: bool = True  # Show hypoglycemia threshold line
CONFIGURATION_HYPO_THRESHOLD: float = 3.9  # mmol/L - hypoglycemia threshold
CONFIGURATION_HYPO_COLOR: str = "#d32f2f"  # Red
CONFIGURATION_HYPO_LINE_WIDTH: float = 1.5  # Line width
CONFIGURATION_HYPO_LINE_STYLE: str = "--"  # Dashed

CONFIGURATION_HYPER_LINE: bool = True  # Show hyperglycemia threshold line
CONFIGURATION_HYPER_THRESHOLD: float = 10.0  # mmol/L - hyperglycemia threshold
CONFIGURATION_HYPER_COLOR: str = "#ff9800"  # Orange
CONFIGURATION_HYPER_LINE_WIDTH: float = 1.5  # Line width
CONFIGURATION_HYPER_LINE_STYLE: str = "--"  # Dashed

# -- Typography --------------------------------------------------------------
CONFIGURATION_FONT_SIZE_TITLE: int = 14  # Figure title
CONFIGURATION_FONT_SIZE_AXIS_LABEL: int = 12  # X and Y axis labels
CONFIGURATION_FONT_SIZE_TICK: int = 10  # Tick labels
CONFIGURATION_FONT_FAMILY: str = "sans-serif"  # Font family

# -- Figure Dimensions -------------------------------------------------------
CONFIGURATION_FIGURE_WIDTH: float = 7.5  # inches
CONFIGURATION_FIGURE_HEIGHT: float = 5.0  # inches
CONFIGURATION_DPI: int = 300  # Resolution for PNG output

# -- Output ------------------------------------------------------------------
CONFIGURATION_OUTPUT_DIR: Path = PROJECT_ROOT / "results" / "figures"
CONFIGURATION_OUTPUT_FILENAME: str = "patient_bg_trace_with_meals"
CONFIGURATION_OUTPUT_FORMATS: list[str] = ["png", "pdf"]

# ── END CONFIGURATION ───────────────────────────────────────────────────────


def load_patient_statistics(loader) -> pd.DataFrame:
    """Compute mean BG and SD for all patients, sorted by mean then SD."""
    print("Computing patient statistics...")
    stats = []

    for patient_id in loader.patient_ids:
        patient_df = loader.get_patient_data(patient_id)
        if patient_df is None or len(patient_df) == 0:
            continue

        # Filter for CGM measurements only
        cgm_df = patient_df[patient_df["msg_type"] == "cgm"]
        if len(cgm_df) == 0:
            continue

        mean_bg = cgm_df["bg_mM"].mean()
        std_bg = cgm_df["bg_mM"].std()
        count = len(cgm_df)

        stats.append(
            {
                "patient_id": patient_id,
                "mean_bg_mmol": mean_bg,
                "std_bg_mmol": std_bg,
                "num_measurements": count,
            }
        )

    stats_df = pd.DataFrame(stats)
    # Sort by mean BG ascending, then by std ascending
    stats_df = stats_df.sort_values(
        by=["mean_bg_mmol", "std_bg_mmol"], ascending=[True, True]
    )

    return stats_df


def select_patient(
    loader,
    patient_id: str | None,
    num_candidates: int,
    seed: int,
) -> str:
    """Select a patient ID - either specified or auto-selected from low-mean-BG patients."""
    if patient_id is not None:
        if patient_id not in loader.patient_ids:
            raise ValueError(
                f"Patient ID '{patient_id}' not found. "
                f"Available: {loader.patient_ids[:5]}..."
            )
        return patient_id

    # Auto-select from candidates with low mean BG
    stats_df = load_patient_statistics(loader)
    candidates = stats_df.head(num_candidates)

    print(f"\nTop {num_candidates} patients with lowest mean BG:")
    print("=" * 70)
    print(candidates.to_string(index=False))
    print("=" * 70)

    # Randomly select one from candidates
    random.seed(seed)
    selected_idx = random.randint(0, len(candidates) - 1)
    selected_patient = candidates.iloc[selected_idx]["patient_id"]

    print(f"\nAuto-selected patient: {selected_patient}")
    print(f"  Mean BG: {candidates.iloc[selected_idx]['mean_bg_mmol']:.2f} mmol/L")
    print(f"  Std BG:  {candidates.iloc[selected_idx]['std_bg_mmol']:.2f} mmol/L")

    return selected_patient


def load_patient_day_data(
    loader,
    patient_id: str,
    date: str | None,
    start_hour: int,
    end_hour: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    """Load patient data for a specific date and time window."""
    patient_df = loader.get_patient_data(patient_id)

    if patient_df is None or len(patient_df) == 0:
        raise ValueError(f"No data found for patient {patient_id}")

    # Filter for CGM measurements only
    cgm_df = patient_df[patient_df["msg_type"] == "cgm"].copy()

    if len(cgm_df) == 0:
        raise ValueError(f"No CGM data found for patient {patient_id}")

    # Reset index to make datetime a column (it's stored as the index)
    cgm_df = cgm_df.reset_index()

    # Parse datetime
    cgm_df["datetime"] = pd.to_datetime(cgm_df["datetime"])
    cgm_df = cgm_df.sort_values("datetime")

    # Select date
    if date is not None:
        selected_date = pd.to_datetime(date).date()
    else:
        # Randomly select a date from available dates
        available_dates = cgm_df["datetime"].dt.date.unique()
        random.seed(seed)
        selected_date = random.choice(available_dates)

    # Filter for selected date and time window
    day_df = cgm_df[cgm_df["datetime"].dt.date == selected_date].copy()

    if len(day_df) == 0:
        raise ValueError(f"No data found for patient {patient_id} on {selected_date}")

    # Filter time window
    day_df = day_df[
        (day_df["datetime"].dt.hour >= start_hour)
        & (day_df["datetime"].dt.hour < end_hour)
    ].copy()

    if len(day_df) == 0:
        raise ValueError(
            f"No data in time window {start_hour}:00-{end_hour}:00 "
            f"for patient {patient_id} on {selected_date}"
        )

    # Calculate hours from start time
    first_time = day_df["datetime"].iloc[0]
    day_df["hours_from_start"] = (
        day_df["datetime"] - first_time
    ).dt.total_seconds() / 3600

    return day_df, str(selected_date)


def create_plot(
    df: pd.DataFrame,
    patient_id: str,
    date: str,
    start_hour: int,
    end_hour: int,
) -> plt.Figure:
    """Create the main BG trace plot with bounding boxes."""
    # Set up matplotlib styling
    plt.rcParams.update(
        {
            "font.family": CONFIGURATION_FONT_FAMILY,
            "font.size": CONFIGURATION_FONT_SIZE_TICK,
            "axes.labelsize": CONFIGURATION_FONT_SIZE_AXIS_LABEL,
            "axes.titlesize": CONFIGURATION_FONT_SIZE_TITLE,
            "xtick.labelsize": CONFIGURATION_FONT_SIZE_TICK,
            "ytick.labelsize": CONFIGURATION_FONT_SIZE_TICK,
        }
    )

    fig, ax = plt.subplots(
        figsize=(CONFIGURATION_FIGURE_WIDTH, CONFIGURATION_FIGURE_HEIGHT)
    )

    # Plot BG trace
    ax.plot(
        df["hours_from_start"],
        df["bg_mM"],
        color=CONFIGURATION_LINE_COLOR,
        linewidth=CONFIGURATION_LINE_WIDTH,
        marker=CONFIGURATION_MARKER_STYLE,
        markersize=CONFIGURATION_MARKER_SIZE,
        label="Blood Glucose",
        zorder=3,
    )

    # Add reference lines for hypo/hyper thresholds
    if CONFIGURATION_HYPO_LINE:
        ax.axhline(
            y=CONFIGURATION_HYPO_THRESHOLD,
            color=CONFIGURATION_HYPO_COLOR,
            linewidth=CONFIGURATION_HYPO_LINE_WIDTH,
            linestyle=CONFIGURATION_HYPO_LINE_STYLE,
            label=f"Hypoglycemia ({CONFIGURATION_HYPO_THRESHOLD} mmol/L)",
            zorder=2,
            alpha=0.7,
        )

    if CONFIGURATION_HYPER_LINE:
        ax.axhline(
            y=CONFIGURATION_HYPER_THRESHOLD,
            color=CONFIGURATION_HYPER_COLOR,
            linewidth=CONFIGURATION_HYPER_LINE_WIDTH,
            linestyle=CONFIGURATION_HYPER_LINE_STYLE,
            label=f"Hyperglycemia ({CONFIGURATION_HYPER_THRESHOLD} mmol/L)",
            zorder=2,
            alpha=0.7,
        )

    # Add bounding boxes
    boxes = [
        {
            "x": CONFIGURATION_BOX1_X_START,
            "width": CONFIGURATION_BOX1_WIDTH,
            "y": CONFIGURATION_BOX1_Y_BOTTOM,
            "height": CONFIGURATION_BOX1_HEIGHT,
        },
        {
            "x": CONFIGURATION_BOX2_X_START,
            "width": CONFIGURATION_BOX2_WIDTH,
            "y": CONFIGURATION_BOX2_Y_BOTTOM,
            "height": CONFIGURATION_BOX2_HEIGHT,
        },
        {
            "x": CONFIGURATION_BOX3_X_START,
            "width": CONFIGURATION_BOX3_WIDTH,
            "y": CONFIGURATION_BOX3_Y_BOTTOM,
            "height": CONFIGURATION_BOX3_HEIGHT,
        },
    ]

    for i, box in enumerate(boxes, start=1):
        rect = Rectangle(
            (box["x"], box["y"]),
            box["width"],
            box["height"],
            linewidth=CONFIGURATION_BOX_LINE_WIDTH,
            edgecolor=CONFIGURATION_BOX_EDGE_COLOR,
            facecolor=CONFIGURATION_BOX_FILL_COLOR,
            alpha=CONFIGURATION_BOX_FILL_ALPHA,
            linestyle=CONFIGURATION_BOX_LINE_STYLE,
            label="Meal" if i == 1 else None,
            zorder=2,
        )
        ax.add_patch(rect)

    # Set axis limits
    time_window_hours = end_hour - start_hour
    ax.set_xlim(0, time_window_hours)
    ax.set_ylim(CONFIGURATION_Y_MIN, CONFIGURATION_Y_MAX)

    # Set up x-axis with actual times
    # Create tick positions every 2 hours
    tick_hours = list(range(0, time_window_hours + 1, 2))
    ax.set_xticks(tick_hours)

    # Create time labels (e.g., "6 AM", "8 AM", ..., "8 PM", "10 PM")
    time_labels = []
    for hour_offset in tick_hours:
        actual_hour = (start_hour + hour_offset) % 24
        if actual_hour == 0:
            time_labels.append("12 AM")
        elif actual_hour < 12:
            time_labels.append(f"{actual_hour} AM")
        elif actual_hour == 12:
            time_labels.append("12 PM")
        else:
            time_labels.append(f"{actual_hour - 12} PM")

    ax.set_xticklabels(time_labels)

    # Labels and title
    ax.set_xlabel("Time", fontsize=CONFIGURATION_FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Blood Glucose (mmol/L)", fontsize=CONFIGURATION_FONT_SIZE_AXIS_LABEL)
    ax.set_title(
        f"Replace-BG: Patient Id {patient_id} — {date}",
        fontsize=CONFIGURATION_FONT_SIZE_TITLE,
        fontweight="bold",
    )

    # Grid
    if CONFIGURATION_GRID:
        ax.grid(
            True,
            alpha=CONFIGURATION_GRID_ALPHA,
            color=CONFIGURATION_GRID_COLOR,
            linestyle="-",
            linewidth=0.5,
            zorder=1,
        )

    # Add legend
    ax.legend(loc="upper right", framealpha=0.9)

    # Tight layout
    fig.tight_layout()

    return fig


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save figure in configured formats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in CONFIGURATION_OUTPUT_FORMATS:
        output_file = output_path.parent / f"{output_path.stem}.{fmt}"
        fig.savefig(
            output_file,
            dpi=CONFIGURATION_DPI,
            bbox_inches="tight",
            format=fmt,
        )
        print(f"Saved: {output_file}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize patient BG trace with meal bounding boxes"
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient ID to visualize (e.g., ale_42). If not specified, auto-selects from low-mean-BG patients.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to visualize (YYYY-MM-DD format). If not specified, randomly selects a date.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (without extension). Default: patient_bg_trace_with_meals",
    )
    parser.add_argument(
        "--show_patient_stats",
        action="store_true",
        help="Show patient statistics and exit without generating plot.",
    )

    args = parser.parse_args()

    # Import data loader here to avoid slow imports if just viewing help
    try:
        from src.data.diabetes_datasets import Aleppo2017DataLoader
    except ImportError as e:
        print(f"Error importing data loader: {e}", file=sys.stderr)
        print("Make sure you're running from the project root.", file=sys.stderr)
        sys.exit(1)

    # Load Aleppo dataset
    print("Loading Aleppo dataset...")
    loader = Aleppo2017DataLoader(use_cached=True)
    print(f"Loaded {loader.num_patients} patients")

    # If only showing stats, compute and exit
    if args.show_patient_stats:
        stats_df = load_patient_statistics(loader)
        print("\nAll patient statistics (sorted by mean BG, then std):")
        print("=" * 70)
        print(stats_df.to_string(index=False))
        print("=" * 70)
        return

    # Select patient
    patient_id = select_patient(
        loader,
        args.patient_id or CONFIGURATION_PATIENT_ID,
        CONFIGURATION_NUM_CANDIDATES,
        CONFIGURATION_SEED,
    )

    # Load patient data for specified date/time window
    print(f"\nLoading data for patient {patient_id}...")
    df, selected_date = load_patient_day_data(
        loader,
        patient_id,
        args.date or CONFIGURATION_DATE,
        CONFIGURATION_START_HOUR,
        CONFIGURATION_END_HOUR,
        CONFIGURATION_SEED,
    )

    print(f"Selected date: {selected_date}")
    print(f"Time window: {CONFIGURATION_START_HOUR}:00 - {CONFIGURATION_END_HOUR}:00")
    print(f"Data points: {len(df)}")
    print(f"BG range: {df['bg_mM'].min():.2f} - {df['bg_mM'].max():.2f} mmol/L")

    # Create plot
    print("\nGenerating plot...")
    fig = create_plot(
        df,
        patient_id,
        selected_date,
        CONFIGURATION_START_HOUR,
        CONFIGURATION_END_HOUR,
    )

    # Save figure
    output_filename = args.output or CONFIGURATION_OUTPUT_FILENAME
    output_path = CONFIGURATION_OUTPUT_DIR / output_filename
    save_figure(fig, output_path)

    print("\nDone!")
    print("\nTo adjust bounding boxes, edit the CONFIGURATION_BOX*_* variables")
    print(f"at the top of {__file__}")


if __name__ == "__main__":
    main()
