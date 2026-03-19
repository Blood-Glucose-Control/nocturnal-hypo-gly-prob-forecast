"""
Plot absolute error distribution (box plots) vs forecast horizon for one or more eval runs.

Usage:
    python scripts/analysis/plot_rmse_vs_horizon.py \
        --results path/to/nocturnal_results.json [path2.json ...] \
        --labels "Zero-Shot" "Fine-Tuned" \
        --output rmse_vs_horizon.svg
"""

import argparse
import json
import matplotlib
matplotlib.use("svg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

COLORS = ["#E07B54", "#3A7FD5", "#5BAD6F", "#9B59B6"]


def load_horizon(path: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)

    episodes = d["per_episode"]
    forecast_length = len(episodes[0]["pred"])
    sampling_interval = 5  # CGM sampling interval (minutes)

    pred_matrix = np.array([ep["pred"] for ep in episodes if len(ep["pred"]) == forecast_length])
    tgt_matrix  = np.array([ep["target_bg"] for ep in episodes if len(ep["target_bg"]) == forecast_length])

    horizon_data = []
    for idx in range(forecast_length):
        ep_rmse = (pred_matrix[:, idx] - tgt_matrix[:, idx]) ** 2
        horizon_data.append({
            "horizon_minutes": (idx + 1) * sampling_interval,
            "rmse": float(np.sqrt(np.mean(ep_rmse))),
            "q10": float(np.sqrt(np.percentile(ep_rmse, 10))),
            "q25": float(np.sqrt(np.percentile(ep_rmse, 25))),
            "q50": float(np.sqrt(np.percentile(ep_rmse, 50))),
            "q75": float(np.sqrt(np.percentile(ep_rmse, 75))),
            "q90": float(np.sqrt(np.percentile(ep_rmse, 90))),
        })
    return horizon_data


def make_plot(results: list[list[dict]], labels: list[str], output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    n_series = len(results)
    box_width = 0.06          # hours
    gap = box_width * 1.3     # space between series at the same horizon
    total_span = gap * (n_series - 1)

    for i, (data, label, color) in enumerate(zip(results, labels, COLORS)):
        hours = np.array([d["horizon_minutes"] / 60 for d in data])
        offset = -total_span / 2 + i * gap
        positions = hours + offset

        has_whiskers = "q10" in data[0]
        stats = [
            {
                "med":    d["q50"],
                "q1":     d["q25"],
                "q3":     d["q75"],
                "whislo": d["q10"] if has_whiskers else d["q25"],
                "whishi": d["q90"] if has_whiskers else d["q75"],
                "fliers": [],
            }
            for d in data
        ]

        bp = ax.bxp(
            stats,
            positions=positions,
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

        # Invisible line for legend
        ax.plot([], [], color=color, linewidth=6, alpha=0.6,
                solid_capstyle="round", label=label)

    ax.set_xlabel("Forecast horizon (hours)", fontsize=11)
    ax.set_ylabel("RMSE (mmol/L)", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

    legend = ax.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")
    legend.set_gid("legend")

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", metadata={"Creator": ""})
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="rmse_vs_horizon.svg")
    args = parser.parse_args()

    if len(args.results) != len(args.labels):
        raise ValueError("--results and --labels must have the same number of entries")

    results = [load_horizon(p) for p in args.results]
    make_plot(results, args.labels, args.output)


if __name__ == "__main__":
    main()
