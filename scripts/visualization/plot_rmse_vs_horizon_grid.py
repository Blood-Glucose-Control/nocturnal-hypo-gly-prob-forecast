"""
Plot mean RMSE (with IQR band) vs forecast horizon for multiple models.
One subplot per dataset (columns), all models overlaid in each panel.

Reads best run paths from best_by_model_dataset.csv. Models missing
forecasts.npz for any dataset are skipped automatically.

Usage:
    python scripts/visualization/plot_rmse_vs_horizon_grid.py
    python scripts/visualization/plot_rmse_vs_horizon_grid.py \
        --csv experiments/nocturnal_forecasting/best_by_model_dataset.csv \
        --output results/figures/rmse_vs_horizon_grid.svg
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.use("svg")
matplotlib.rcParams["svg.fonttype"] = "none"

DATASETS = ["aleppo_2017", "brown_2019", "lynch_2022"]
DATASET_LABELS = {
    "aleppo_2017": "Aleppo 2017",
    "brown_2019": "Brown 2019",
    "lynch_2022": "Lynch 2022",
}

# For models whose best CSV entry lacks forecasts.npz, point at a newer run
# that does have it. (TTM aleppo/brown entries in the CSV are from Feb 2026
# and predate the NPZ storage format.)
FALLBACK_PATHS: dict[tuple[str, str], str] = {
    ("ttm", "aleppo_2017"): (
        "experiments/nocturnal_forecasting/512ctx_96fh/ttm"
        "/2026-04-16_0922_aleppo_2017_finetuned"
    ),
    ("ttm", "brown_2019"): (
        "experiments/nocturnal_forecasting/512ctx_96fh/ttm"
        "/2026-04-16_0925_brown_2019_finetuned"
    ),
}

MODEL_COLORS: dict[str, str] = {
    "chronos2": "#E07B54",
    "moirai": "#3A7FD5",
    "tide": "#5BAD6F",
    "timesfm": "#9B59B6",
    "ttm": "#C8963E",
    "toto": "#E84B8A",
    "moment": "#3BBFBF",
    "sundial": "#9E9E9E",
    "timegrad": "#7B5033",
}

SAMPLING_INTERVAL_MIN = 5  # CGM cadence


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_npz(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, actuals) arrays from forecasts.npz."""
    data = np.load(run_dir / "forecasts.npz", allow_pickle=False)
    return data["predictions"], data["actuals"]


def per_step_stats(preds: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute per-step mean RMSE and IQR across episodes.

    Returns a dict with keys:
        hours   : (n_steps,) forecast horizon in hours
        mean    : (n_steps,) mean RMSE across episodes
        q25     : (n_steps,) 25th-percentile per-episode RMSE
        q75     : (n_steps,) 75th-percentile per-episode RMSE
    """
    n_steps = preds.shape[1]
    hours = np.array([(i + 1) * SAMPLING_INTERVAL_MIN / 60.0 for i in range(n_steps)])
    sq_err = (preds - actuals) ** 2  # (n_episodes, n_steps)
    mean_rmse = np.sqrt(np.mean(sq_err, axis=0))
    ep_rmse = np.sqrt(sq_err)  # per-episode RMSE at each step
    q25 = np.percentile(ep_rmse, 25, axis=0)
    q75 = np.percentile(ep_rmse, 75, axis=0)

    # Cumulative RMSE: sqrt of running mean of squared errors over all
    # episodes and all steps up to k.  Correct because we accumulate MSE
    # (linear) before taking the single square root.
    cum_mse = np.cumsum(np.mean(sq_err, axis=0)) / np.arange(1, n_steps + 1)
    cum_rmse = np.sqrt(cum_mse)

    return {
        "hours": hours,
        "mean": mean_rmse,
        "q25": q25,
        "q75": q75,
        "cumulative": cum_rmse,
    }


def read_best_paths(csv_path: Path) -> dict[tuple[str, str], str]:
    """Return {(model, dataset): run_path} for the best (lowest RMSE) run."""
    best: dict[tuple[str, str], tuple[float, str]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] not in DATASETS:
                continue
            key = (row["model"], row["dataset"])
            rmse = float(row["rmse"])
            if key not in best or rmse < best[key][0]:
                best[key] = (rmse, row["run_path"])
    return {k: v[1] for k, v in best.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_grid_plot(
    stats: dict[str, dict[str, dict]],
    ordered_models: list[str],
    output_path: str,
    show_iqr: bool = True,
    cumulative: bool = False,
) -> None:
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(
        1, n_datasets, figsize=(5 * n_datasets, 4.5), sharey=True, sharex=True
    )

    for col, ds in enumerate(DATASETS):
        ax = axes[col]

        for model in ordered_models:
            color = MODEL_COLORS.get(model, "#888888")
            s = stats[model][ds]
            y = s["cumulative"] if cumulative else s["mean"]
            if show_iqr and not cumulative:
                ax.fill_between(
                    s["hours"], s["q25"], s["q75"], alpha=0.12, color=color, zorder=2
                )
            ax.plot(
                s["hours"],
                y,
                color=color,
                linewidth=1.8,
                label=model,
                zorder=3,
            )

        ax.set_title(DATASET_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast horizon (hours)", fontsize=10)
        if col == 0:
            ylabel = "Cumulative RMSE (mmol/L)" if cumulative else "RMSE (mmol/L)"
            ax.set_ylabel(ylabel, fontsize=10)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    # Single legend on the last panel, models listed best-first
    axes[-1].legend(
        fontsize=8,
        framealpha=0.85,
        loc="upper left",
        title="Model (best → worst)",
        title_fontsize=8,
    )

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), format="svg", bbox_inches="tight", metadata={"Creator": ""})
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid plot of RMSE vs horizon — one column per dataset, "
        "multiple models per panel."
    )
    parser.add_argument(
        "--csv",
        default="experiments/nocturnal_forecasting/best_by_model_dataset.csv",
        help="Path to best_by_model_dataset.csv",
    )
    parser.add_argument(
        "--output",
        default="results/figures/rmse_vs_horizon_grid.svg",
        help="Output SVG path",
    )
    parser.add_argument(
        "--no-iqr",
        action="store_true",
        dest="no_iqr",
        help="Hide the per-episode IQR shaded band (cleaner when many models overlap)",
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Plot cumulative RMSE (sqrt of running mean MSE) instead of per-step RMSE",
    )
    args = parser.parse_args()

    best_paths = read_best_paths(Path(args.csv))
    for key, path in FALLBACK_PATHS.items():
        best_paths[key] = path

    # Keep only models that have forecasts.npz for every dataset
    all_models = sorted({m for m, _ in best_paths})
    complete_models = [
        m
        for m in all_models
        if all(
            (m, ds) in best_paths
            and (Path(best_paths[(m, ds)]) / "forecasts.npz").exists()
            for ds in DATASETS
        )
    ]
    print(f"Models with complete NPZ coverage: {complete_models}")

    # Load per-step stats
    run_stats: dict[str, dict[str, dict]] = {}
    for model in complete_models:
        run_stats[model] = {}
        for ds in DATASETS:
            preds, actuals = load_npz(Path(best_paths[(model, ds)]))
            run_stats[model][ds] = per_step_stats(preds, actuals)

    # Order best → worst by mean RMSE averaged across all datasets and steps
    def mean_overall_rmse(model: str) -> float:
        return float(
            np.mean([np.mean(run_stats[model][ds]["mean"]) for ds in DATASETS])
        )

    ordered = sorted(complete_models, key=mean_overall_rmse)
    print("Model order (best→worst):", ordered)

    make_grid_plot(
        run_stats,
        ordered,
        args.output,
        show_iqr=not args.no_iqr,
        cumulative=args.cumulative,
    )


if __name__ == "__main__":
    main()
