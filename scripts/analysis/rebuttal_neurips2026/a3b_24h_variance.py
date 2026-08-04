# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A3b: 24-Hour BG Variance Analysis (NeurIPS 2026 E&D Rebuttal, submission #3091).

Computes blood glucose coefficient of variation across all 24 hours (1-hour windows)
to establish whether the midnight-anchored evaluation window (00:00-08:00) captures
a physiologically stable period compared to daytime hours.

Also computes hypoglycemia counts (BG < 3.9 mmol/L) by hour to show temporal
distribution of events.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a3b_24h_variance

Outputs:
    outputs/a3b_24h_variance.csv - BG statistics for each hour
    outputs/a3b_24h_variance.png - Combined variance plot (all datasets)
    outputs/a3b_24h_hypoglycemia_counts.png - Hypoglycemia count bar charts (per dataset)
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"
PROC = Path("/data/shared/cache/data")


@functools.lru_cache(maxsize=2048)
def _patient_bg_frame(dataset: str, pid: str) -> pd.DataFrame | None:
    """Load a processed patient CSV with datetime + BG."""
    f = PROC / dataset / "processed" / f"{pid}_full.csv"
    if not f.exists():
        return None
    # BG column varies by dataset: bg_mM (most), bg, glucose
    df = pd.read_csv(f, usecols=lambda c: c in ("datetime", "bg_mM", "bg", "glucose"))
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Find which BG column exists
    bg_col = None
    for col in ["bg_mM", "bg", "glucose"]:
        if col in df.columns:
            bg_col = col
            break

    if bg_col is None:
        return None

    df["_bg"] = pd.to_numeric(df[bg_col], errors="coerce")
    df = df.dropna(subset=["datetime", "_bg"]).set_index("datetime").sort_index()
    return df[["_bg"]]


def _window_stats(dataset: str, pids: list[str], hour: int) -> dict:
    """BG statistics and hypoglycemia counts for a single clock hour across all patients.

    Args:
        dataset: Dataset name
        pids: List of patient IDs
        hour: Hour of day (0-23). hour=1 means 01:00:00-01:59:59

    Returns:
        Dict with n, mean, std, cv, and hypo_count
    """
    vals = []
    hypo_count = 0
    hypo_threshold = 3.9  # mmol/L

    for pid in pids:
        pf = _patient_bg_frame(dataset, pid)
        if pf is None:
            continue
        # filter to exact hour (e.g., hour=1 means 01:00:00 to 01:59:59)
        mask = pf.index.hour == hour
        bg = pf.loc[mask, "_bg"].to_numpy()
        if len(bg) > 0:
            vals.extend(bg)
            hypo_count += (bg < hypo_threshold).sum()

    if len(vals) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, cv=np.nan, hypo_count=0)

    arr = np.array(vals)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    cv_val = std_val / mean_val if mean_val > 0 else np.nan

    return dict(
        n=len(arr), mean=mean_val, std=std_val, cv=cv_val, hypo_count=int(hypo_count)
    )


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    rows = []
    for dataset in C.DATASETS:
        # get patient list from any available run
        ref_rp = next(
            (
                C.best_run_path(m, dataset)
                for m in ["chronos2", "moirai", "patchtst"]
                if C.best_run_path(m, dataset)
            ),
            None,
        )
        if ref_rp is None:
            continue
        ref = C.load_run(ref_rp, dataset=dataset).episodes
        pids = sorted(ref["pid"].unique())

        print(f"\n{C.DATASET_LABEL[dataset]} (n={len(pids)} patients)")
        # Compute stats for each 1-hour window
        for hour in range(24):
            stats = _window_stats(dataset, pids, hour)
            rows.append(
                dict(
                    dataset=dataset,
                    dataset_label=C.DATASET_LABEL[dataset],
                    hour=hour,
                    window_label=f"{hour:02d}:00-{(hour+1)%24:02d}:00",
                    n=stats["n"],
                    mean_bg=stats["mean"],
                    std_bg=stats["std"],
                    cv=stats["cv"],
                    hypo_count=stats["hypo_count"],
                )
            )
            print(
                f"  {hour:02d}:00-{(hour+1)%24:02d}:00  n={stats['n']:7d}  "
                f"mean={stats['mean']:5.2f}  std={stats['std']:5.2f}  CV={stats['cv']:.4f}  "
                f"hypo_count={stats['hypo_count']:6d}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "a3b_24h_variance.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/a3b_24h_variance.csv")

    # Visualization
    _plot_variance_combined(df)
    _plot_hypoglycemia_counts(df)
    _plot_hypoglycemia_normalized(df)


def _plot_variance_combined(df: pd.DataFrame) -> None:
    """Generate a combined variance plot with all datasets on one panel."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette for datasets
    colors = {
        "Replace-BG": "#1f77b4",
        "DCLP3": "#ff7f0e",
        "IOBP2": "#2ca02c",
        "Tamborlane": "#d62728",
    }

    for dataset in C.DATASETS:
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("hour")

        # Extract x (hour) and y (CV)
        x = subset["hour"].values
        y = subset["cv"].values
        label = C.DATASET_LABEL[dataset]

        # Plot
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
            color=colors.get(label, None),
            alpha=0.8,
        )

    # Highlight evaluation window
    ax.axvspan(0, 8, color="green", alpha=0.1, label="Evaluation window (00:00-08:00)")

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 1))
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 24, 1)], rotation=45, ha="right", fontsize=9
    )
    ax.set_xlabel("Time of Day (Hour)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coefficient of Variation", fontsize=12, fontweight="bold")
    ax.set_title(
        "24-Hour Blood Glucose Variability Pattern\n"
        "Comparing Nocturnal (00:00-08:00) vs Daytime Variability",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = OUT_DIR / "a3b_24h_variance.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")


def _plot_hypoglycemia_counts(df: pd.DataFrame) -> None:
    """Generate bar charts showing hypoglycemia counts by hour for each dataset."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    # First pass: find the global max for consistent y-axis
    global_max = 0
    for dataset in C.DATASETS:
        subset = df[df["dataset"] == dataset].copy()
        if not subset.empty:
            global_max = max(global_max, subset["hypo_count"].max())

    # Add 10% padding to y-axis max
    y_max = global_max * 1.1

    for i, dataset in enumerate(C.DATASETS):
        ax = axes[i]
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("hour")

        # Extract x (hour) and y (hypo count)
        x = subset["hour"].values
        y = subset["hypo_count"].values

        # Bar plot with evaluation window highlighted
        colors_bar = ["#90EE90" if h < 8 else "#4169E1" for h in x]
        ax.bar(x, y, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=0.5)

        # Add reference patches for legend - moved to upper left
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="#90EE90",
                alpha=0.7,
                edgecolor="black",
                label="Evaluation window (00:00-08:00)",
            ),
            Patch(
                facecolor="#4169E1",
                alpha=0.7,
                edgecolor="black",
                label="Daytime (08:00-24:00)",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(0, y_max)  # Set consistent y-axis across all panels
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(
            [f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, ha="right"
        )
        ax.set_xlabel("Hour of Day", fontsize=10)
        ax.set_ylabel("Number of Hypoglycemic Readings", fontsize=10)
        ax.set_title(C.DATASET_LABEL[dataset], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add total count annotation - moved to right side, lower position
        total_hypo = y.sum()
        overnight_hypo = y[:8].sum()
        overnight_pct = 100 * overnight_hypo / total_hypo if total_hypo > 0 else 0
        ax.text(
            0.98,
            0.70,
            f"Total hypo readings: {total_hypo:,}\n"
            f"Overnight (00:00-08:00): {overnight_hypo:,} ({overnight_pct:.1f}%)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.suptitle(
        "Hypoglycemia Event Distribution by Hour of Day\n"
        "Number of BG readings < 3.9 mmol/L",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    png_path = OUT_DIR / "a3b_24h_hypoglycemia_counts.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")


def _plot_hypoglycemia_normalized(df: pd.DataFrame) -> None:
    """Generate normalized bar charts showing percentage of total hypoglycemia by hour."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    # First pass: find the global max percentage for consistent y-axis
    global_max_pct = 0
    for dataset in C.DATASETS:
        subset = df[df["dataset"] == dataset].copy()
        if not subset.empty:
            total_hypo = subset["hypo_count"].sum()
            if total_hypo > 0:
                max_pct = 100 * subset["hypo_count"].max() / total_hypo
                global_max_pct = max(global_max_pct, max_pct)

    # Add 10% padding to y-axis max
    y_max = global_max_pct * 1.1

    for i, dataset in enumerate(C.DATASETS):
        ax = axes[i]
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("hour")

        # Extract x (hour) and normalize y to percentage
        x = subset["hour"].values
        y_count = subset["hypo_count"].values
        total_hypo = y_count.sum()
        y_pct = 100 * y_count / total_hypo if total_hypo > 0 else np.zeros_like(y_count)

        # Bar plot with evaluation window highlighted
        colors_bar = ["#90EE90" if h < 8 else "#4169E1" for h in x]
        ax.bar(x, y_pct, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=0.5)

        # Add reference patches for legend - moved to upper left
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor="#90EE90",
                alpha=0.7,
                edgecolor="black",
                label="Evaluation window (00:00-08:00)",
            ),
            Patch(
                facecolor="#4169E1",
                alpha=0.7,
                edgecolor="black",
                label="Daytime (08:00-24:00)",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(0, y_max)  # Set consistent y-axis across all panels
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(
            [f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, ha="right"
        )
        ax.set_xlabel("Hour of Day", fontsize=10)
        ax.set_ylabel("Percentage of Total Hypoglycemia (%)", fontsize=10)
        ax.set_title(C.DATASET_LABEL[dataset], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add normalized annotation
        overnight_pct = y_pct[:8].sum()
        ax.text(
            0.98,
            0.70,
            f"Total hypo readings: {total_hypo:,}\n"
            f"Overnight share: {overnight_pct:.1f}%",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.suptitle(
        "Normalized Hypoglycemia Event Distribution by Hour of Day\n"
        "Percentage of total hypoglycemic readings (BG < 3.9 mmol/L) in each hour",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    png_path = OUT_DIR / "a3b_24h_hypoglycemia_normalized.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
