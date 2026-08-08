"""
Plot absolute error distribution (box plots) vs forecast horizon for one or more eval runs.

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
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.use("svg")
matplotlib.rcParams["svg.fonttype"] = "none"

COLORS = ["#E07B54", "#3A7FD5", "#5BAD6F", "#9B59B6"]


def load_horizon(path: str) -> list[dict]:
    p = Path(path)

    # Resolve run directory: prefer Tier 3 NPZ, fall back to legacy JSON
    if p.is_dir():
        npz_candidate = p / "forecasts.npz"
        json_candidate = p / "nocturnal_results.json"
        if npz_candidate.exists():
            p = npz_candidate
        elif json_candidate.exists():
            p = json_candidate
        else:
            raise FileNotFoundError(
                f"No results found in {path!r}: expected forecasts.npz or nocturnal_results.json"
            )

    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        pred_matrix = data["predictions"]  # (n_episodes, forecast_length)
        tgt_matrix = data["actuals"]  # (n_episodes, forecast_length)
        forecast_length = pred_matrix.shape[1]
    else:
        with open(p) as f:
            d = json.load(f)
        episodes = d["per_episode"]
        forecast_length = len(episodes[0]["pred"])
        pred_matrix = np.array(
            [ep["pred"] for ep in episodes if len(ep["pred"]) == forecast_length]
        )
        tgt_matrix = np.array(
            [
                ep["target_bg"]
                for ep in episodes
                if len(ep["target_bg"]) == forecast_length
            ]
        )

    sampling_interval = 5  # CGM sampling interval (minutes)

    horizon_data = []
    for idx in range(forecast_length):
        ep_rmse = (pred_matrix[:, idx] - tgt_matrix[:, idx]) ** 2
        horizon_data.append(
            {
                "horizon_minutes": (idx + 1) * sampling_interval,
                "rmse": float(np.sqrt(np.mean(ep_rmse))),
                "q10": float(np.sqrt(np.percentile(ep_rmse, 10))),
                "q25": float(np.sqrt(np.percentile(ep_rmse, 25))),
                "q50": float(np.sqrt(np.percentile(ep_rmse, 50))),
                "q75": float(np.sqrt(np.percentile(ep_rmse, 75))),
                "q90": float(np.sqrt(np.percentile(ep_rmse, 90))),
            }
        )
    return horizon_data


def make_plot(results: list[list[dict]], labels: list[str], output_path: str):
    n_series = len(results)
    box_width = 0.06  # hours

    fig, axes = plt.subplots(
        1, n_series, figsize=(5 * n_series, 5), sharey=True, sharex=True
    )
    if n_series == 1:
        axes = [axes]

    # Compute shared y-axis upper limit across all series
    all_q90 = [d["q90"] for data in results for d in data]
    y_max = max(all_q90) * 1.1

    for i, (ax, data, label, color) in enumerate(zip(axes, results, labels, COLORS)):
        hours = np.array([d["horizon_minutes"] / 60 for d in data])

        stats = [
            {
                "med": d["q50"],
                "q1": d["q25"],
                "q3": d["q75"],
                "whislo": d["q10"],
                "whishi": d["q90"],
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
