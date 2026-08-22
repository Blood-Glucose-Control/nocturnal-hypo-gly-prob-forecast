"""
Plot absolute error distribution (box plots) vs forecast horizon for one or more eval runs.

What this visualization tells you
---------------------------------
- Error growth profile as forecast horizon increases.
- Distributional spread (not just mean), including tails and median.

What to look for
----------------
- Steep late-horizon growth (poor long-horizon stability).
- Wide boxes/whiskers at specific horizons (episode heterogeneity).
- Consistent median separation between model lines of evidence.

Usage:
    python scripts/visualization/plot_rmse_vs_horizon.py \
        --results path/to/run_dir [path2/run_dir ...] \
        --labels "Zero-Shot" "Fine-Tuned" \
        --output rmse_vs_horizon.svg

--results accepts three formats (auto-detected):
  1. Run directory  — prefers forecasts.npz (Tier 3), falls back to nocturnal_results.json
  2. forecasts.npz  — Tier 3 compressed arrays (new storage format)
  3. nocturnal_results.json — legacy monolithic JSON
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.visualization.nocturnal import (
    DEFAULT_BOXPLOT_QUANTILES,
    compute_horizon_rmse_quantiles,
    load_prediction_actual_arrays,
)

matplotlib.use("svg")
matplotlib.rcParams["svg.fonttype"] = "none"

COLORS = ["#E07B54", "#3A7FD5", "#5BAD6F", "#9B59B6"]


def load_horizon(
    path: str, quantiles: tuple[float, float, float, float, float]
) -> list[dict]:
    predictions, actuals = load_prediction_actual_arrays(path)
    return compute_horizon_rmse_quantiles(predictions, actuals, quantiles=quantiles)


def make_plot(results: list[list[dict]], labels: list[str], output_path: str):
    n_series = len(results)
    box_width = 0.06  # hours

    fig, axes = plt.subplots(
        1, n_series, figsize=(5 * n_series, 5), sharey=True, sharex=True
    )
    if n_series == 1:
        axes = [axes]

    # Compute shared y-axis upper limit across all series
    all_whisker_high = [d["whisker_high"] for data in results for d in data]
    y_max = max(all_whisker_high) * 1.1

    for i, (ax, data, label, color) in enumerate(zip(axes, results, labels, COLORS)):
        hours = np.array([d["horizon_minutes"] / 60 for d in data])

        stats = [
            {
                "med": d["median"],
                "q1": d["box_low"],
                "q3": d["box_high"],
                "whislo": d["whisker_low"],
                "whishi": d["whisker_high"],
                "fliers": [],
            }
            for d in data
        ]

        bp = ax.bxp(
            stats,
            positions=hours,
            widths=box_width,
            manage_ticks=False,
            patch_artist=True,
            showfliers=False,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(color)  # type: ignore[union-attr]
            patch.set_alpha(0.6)
            patch.set_edgecolor(color)  # type: ignore[union-attr]
            patch.set_gid(f"box-{label}")
        for element in bp["medians"]:
            element.set_color("white")
            element.set_linewidth(1.8)
            element.set_gid(f"median-{label}")
        for element in bp["whiskers"] + bp["caps"]:
            element.set_color(color)
            element.set_linewidth(1.2)
            element.set_gid(f"whisker-{label}")

        ax.set_title(label, fontsize=11, color=color, fontweight="bold")
        ax.set_xlabel("Forecast horizon (hours)", fontsize=10)
        if i == 0:
            ax.set_ylabel("RMSE (mmol/L)", fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=y_max)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(
        output_path, format="svg", bbox_inches="tight", metadata={"Creator": ""}
    )
    print(f"Saved: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="rmse_vs_horizon.svg")
    parser.add_argument(
        "--quantiles",
        nargs=5,
        type=float,
        default=list(DEFAULT_BOXPLOT_QUANTILES),
        metavar=("WHISKER_LOW", "BOX_LOW", "MEDIAN", "BOX_HIGH", "WHISKER_HIGH"),
        help=(
            "Five quantiles (0-100) used for whiskers/box/median, in ascending order. "
            "Default: 10 25 50 75 90."
        ),
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if len(args.results) != len(args.labels):
        raise ValueError("--results and --labels must have the same number of entries")

    quantiles = tuple(args.quantiles)
    results = [load_horizon(path, quantiles) for path in args.results]
    make_plot(results, args.labels, args.output)


if __name__ == "__main__":
    main()
