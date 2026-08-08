#!/usr/bin/env python3
"""
2-D coverage calibration error scatter.

X-axis : 0.50 − coverage_50  (50% PI coverage error)
Y-axis : 0.80 − coverage_80  (80% PI coverage error)

Both axes: 0 = perfectly calibrated, positive = under-coverage,
           negative = over-coverage.

Colour : model type
Marker : dataset

The ideal point is the origin (0, 0) — both levels perfectly calibrated.
Dashed cross-hairs mark the ideal origin.

Usage
-----
    python scripts/visualization/plot_coverage_error_2d.py
    python scripts/visualization/plot_coverage_error_2d.py \\
        --summary experiments/nocturnal_forecasting/summary.csv \\
        --out results/coverage_error_2d.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Aesthetics (shared with the other scatter scripts)
# ---------------------------------------------------------------------------

MODEL_COLORS: dict[str, str] = {
    "chronos2": "#e377c2",  # pink
    "moirai": "#1f77b4",
    "toto": "#ff7f0e",
    "tide": "#2ca02c",
    "timesfm": "#d62728",
    "timegrad": "#9467bd",
    "sundial": "#8c564b",
}
_DEFAULT_COLOR = "#7f7f7f"

DATASET_MARKERS: dict[str, str] = {
    "aleppo_2017": "o",
    "brown_2019": "s",
    "lynch_2022": "^",
    "tamborlane_2008": "D",
}
_DEFAULT_MARKER = "P"

DATASET_LABELS: dict[str, str] = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
    "tamborlane_2008": "Tamborlane 2008",
}

DEFAULT_SUMMARY = "experiments/nocturnal_forecasting/summary.csv"
DEFAULT_OUT = "results/coverage_error_2d.png"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    print(f"Loaded {len(df)} rows.")
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["model", "dataset", "mode", "checkpoint"], keep="last"
    )
    print(f"After dedup (unique checkpoint × dataset): {len(df)} rows.")

    # Keep rows where both coverage levels are present and finite
    mask = (
        df["coverage_50"].notna()
        & df["coverage_80"].notna()
        & np.isfinite(df["coverage_50"].astype(float))
        & np.isfinite(df["coverage_80"].astype(float))
    )
    df = df[mask].copy()
    df["err_50"] = 0.50 - df["coverage_50"].astype(float)
    df["err_80"] = 0.80 - df["coverage_80"].astype(float)
    print(f"Plotting {len(df)} rows with valid 50% and 80% coverage.")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Coverage calibration: 50% vs 80% prediction interval\n"
        "★ = ideal (both levels perfectly calibrated)",
        fontsize=13,
        fontweight="bold",
    )

    lim_full = max(abs(df["err_50"]).max(), abs(df["err_80"]).max()) * 1.05
    zoom = 0.1  # zoom panel bounds

    def _draw_panel(ax: plt.Axes, xlim: float, ylim: float, title: str) -> None:
        for (model, dataset), grp in df.groupby(["model", "dataset"]):
            ax.scatter(
                grp["err_50"],
                grp["err_80"],
                color=MODEL_COLORS.get(str(model), _DEFAULT_COLOR),
                marker=DATASET_MARKERS.get(str(dataset), _DEFAULT_MARKER),
                s=65,
                alpha=0.45,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
        ax.axvline(
            0, color="black", linestyle="--", linewidth=0.9, alpha=0.45, zorder=2
        )
        ax.axhline(
            0, color="black", linestyle="--", linewidth=0.9, alpha=0.45, zorder=2
        )
        ax.plot(
            [-xlim, xlim],
            [-ylim, ylim],
            color="#aaa",
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )
        ax.scatter(
            [0],
            [0],
            marker="*",
            s=200,
            color="gold",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ax.text(
            xlim * 0.02,
            -ylim * 0.05,
            "ideal",
            fontsize=8.5,
            ha="left",
            va="top",
            color="#333",
            style="italic",
        )
        ax.set_xlabel(
            "50% prediction interval coverage error   (0.50 − empirical)\n"
            "← over-coverage  ·  0 = perfect  ·  under-coverage →",
            fontsize=9.5,
        )
        ax.set_ylabel(
            "80% prediction interval coverage error   (0.80 − empirical)\n"
            "← over-coverage  ·  0 = perfect  ·  under-coverage →",
            fontsize=9.5,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.grid(True, linestyle=":", alpha=0.35)

    _draw_panel(ax_full, lim_full, lim_full, "Overview (all models)")
    _draw_panel(ax_zoom, zoom, zoom, "Zoomed in  (−0.1 to 0.1)")

    # Draw a rectangle on the full panel showing the zoom region
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (-zoom, -zoom),
        2 * zoom,
        2 * zoom,
        linewidth=1.4,
        edgecolor="#444",
        facecolor="none",
        linestyle="-",
        zorder=6,
    )
    ax_full.add_patch(rect)
    ax_full.text(
        zoom + lim_full * 0.02,
        zoom,
        "zoom →",
        fontsize=8,
        va="top",
        ha="left",
        color="#444",
    )

    # Legends on the zoom panel only (avoids clutter on the full view)
    model_handles = [
        mlines.Line2D(
            [],
            [],
            color=c,
            marker="o",
            linestyle="None",
            markersize=8,
            label=m.capitalize(),
        )
        for m, c in MODEL_COLORS.items()
    ]
    dataset_handles = [
        mlines.Line2D(
            [],
            [],
            color="#333",
            marker=mk,
            linestyle="None",
            markersize=8,
            label=DATASET_LABELS.get(ds, ds),
        )
        for ds, mk in DATASET_MARKERS.items()
    ]
    leg1 = ax_zoom.legend(
        handles=model_handles,
        title="Model (colour)",
        loc="upper left",
        fontsize=8.5,
        title_fontsize=9.5,
        framealpha=0.85,
    )
    ax_zoom.add_artist(leg1)
    ax_zoom.legend(
        handles=dataset_handles,
        title="Dataset (marker)",
        loc="lower right",
        fontsize=8.5,
        title_fontsize=9.5,
        framealpha=0.85,
    )

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
