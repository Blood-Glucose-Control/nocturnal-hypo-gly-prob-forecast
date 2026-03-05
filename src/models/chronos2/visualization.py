# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

"""
Evaluation visualization for Chronos-2 forecasts.

Generates best-N and worst-N episode plots from midnight-anchored
evaluation results. Each subplot shows true BG, predicted BG, and IOB
on a dual-axis layout — same visual style as
scripts/visualize_chronos2_best30_v4.py.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Clinical threshold for hypoglycemia (mmol/L)
HYPO_THRESHOLD = 3.9


def plot_evaluation_episodes(
    episodes: List[Dict[str, Any]],
    per_episode: List[Dict[str, Any]],
    output_dir: str,
    model_label: str = "Chronos-2",
    forecast_length: int = 72,
    interval_mins: int = 5,
    n_display: int = 30,
    covariate_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Generate best-N and worst-N evaluation plots.

    Creates two PNG files in output_dir:
    - best_{n}_episodes.png — episodes with lowest RMSE (model strengths)
    - worst_{n}_episodes.png — episodes with highest RMSE (failure modes)

    Each subplot shows a 6-hour forecast window with:
    - True BG (black solid) on the left axis
    - Predicted BG (steelblue solid) on the left axis
    - IOB trajectory (gray dashed) on the right axis
    - Hypo threshold at 3.9 mmol/L (red dotted)

    Args:
        episodes: Episode dicts from model.evaluate() with return_predictions=True.
            Each has keys: anchor, context_df, target_bg, future_covariates.
        per_episode: Per-episode results from evaluate_with_covariates.
            Each has keys: pred (numpy array), rmse (float).
        output_dir: Directory to save PNG files.
        model_label: Label for the model in the plot title.
        forecast_length: Number of forecast timesteps.
        interval_mins: Sampling interval in minutes.
        n_display: Number of episodes to show per plot (default 30).
        covariate_cols: Covariate column names to plot on secondary axis.
            First available covariate is used. Defaults to ["iob"].

    Returns:
        Dict mapping plot name to file path (e.g. {"best_30": "/path/to/best_30_episodes.png"}).
    """
    import matplotlib

    matplotlib.use("Agg")

    if covariate_cols is None:
        covariate_cols = ["iob"]

    os.makedirs(output_dir, exist_ok=True)

    # Build sortable list of (episode, per_episode_result) pairs
    indexed = []
    for i, (ep, pe) in enumerate(zip(episodes, per_episode)):
        if np.isnan(pe["rmse"]) or len(pe["pred"]) == 0:
            continue

        # Compute mean covariate value for display
        mean_cov = 0.0
        cov_label = covariate_cols[0] if covariate_cols else "cov"
        if "future_covariates" in ep:
            for col in covariate_cols:
                vals = ep["future_covariates"].get(col)
                if vals is not None and len(vals) > 0:
                    mean_cov = float(np.mean(vals))
                    cov_label = col.upper()
                    break

        indexed.append(
            {
                "idx": i,
                "episode": ep,
                "pred": pe["pred"],
                "rmse": pe["rmse"],
                "mean_cov": mean_cov,
                "cov_label": cov_label,
            }
        )

    if not indexed:
        logger.warning("No valid episodes for plotting")
        return {}

    plot_paths = {}

    # Best N (ascending RMSE) and Worst N (descending RMSE)
    selections = [
        {
            "name": f"best_{n_display}",
            "title": f"Best {n_display} Episodes (Lowest RMSE)",
            "ascending": True,
            "filename": f"best_{n_display}_episodes.png",
        },
        {
            "name": f"worst_{n_display}",
            "title": f"Worst {n_display} Episodes (Highest RMSE)",
            "ascending": False,
            "filename": f"worst_{n_display}_episodes.png",
        },
    ]

    for sel in selections:
        sorted_eps = sorted(
            indexed, key=lambda x: x["rmse"], reverse=not sel["ascending"]
        )
        selected = sorted_eps[: min(n_display, len(sorted_eps))]

        fig_path = _plot_episode_grid(
            selected,
            output_dir=output_dir,
            filename=sel["filename"],
            title=f"{model_label}: {sel['title']}",
            forecast_length=forecast_length,
            interval_mins=interval_mins,
            covariate_cols=covariate_cols,
        )
        plot_paths[sel["name"]] = fig_path
        logger.info("Saved %s plot: %s", sel["name"], fig_path)

    return plot_paths


def _plot_episode_grid(
    selected_episodes: List[Dict[str, Any]],
    output_dir: str,
    filename: str,
    title: str,
    forecast_length: int,
    interval_mins: int,
    covariate_cols: List[str],
) -> str:
    """Plot episodes in a grid with dual axes (BG + covariate).

    Returns path to saved PNG.
    """
    import matplotlib.pyplot as plt

    n = len(selected_episodes)
    ncols = 5
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 5 * nrows))

    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = axes.flatten()

    time_hours = np.arange(forecast_length) * interval_mins / 60

    # Compute aggregate RMSE for the title
    avg_rmse = float(np.mean([e["rmse"] for e in selected_episodes]))

    for plot_idx, entry in enumerate(selected_episodes):
        ax = axes_flat[plot_idx]
        ax2 = ax.twinx()

        ep = entry["episode"]
        pred = entry["pred"]

        # True BG (always on top)
        actual = ep["target_bg"][:forecast_length]
        ax.plot(
            time_hours[: len(actual)],
            actual,
            "k-",
            linewidth=2.5,
            label="Actual BG",
            zorder=10,
        )

        # Predicted BG
        ax.plot(
            time_hours[: len(pred)],
            pred,
            color="steelblue",
            linestyle="-",
            linewidth=1.8,
            label=f"Predicted (RMSE={entry['rmse']:.2f})",
            zorder=5,
        )

        # Covariate on secondary axis (first available)
        if "future_covariates" in ep:
            for col in covariate_cols:
                vals = ep["future_covariates"].get(col)
                if vals is not None and len(vals) > 0:
                    cov_vals = vals[:forecast_length]
                    ax2.plot(
                        time_hours[: len(cov_vals)],
                        cov_vals,
                        color="gray",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.5,
                        label=col.upper(),
                    )
                    ax2.set_ylabel(f"{col.upper()} (U)", fontsize=7, color="gray")
                    ax2.tick_params(axis="y", labelcolor="gray", labelsize=6)
                    max_val = max(cov_vals) if len(cov_vals) > 0 else 1
                    ax2.set_ylim(0, max(max_val * 1.5, 1))
                    break

        # Hypo threshold
        ax.axhline(y=HYPO_THRESHOLD, color="red", linestyle=":", alpha=0.3, linewidth=1)

        ax.set_xlabel("Hours", fontsize=7)
        ax.set_ylabel("BG (mmol/L)", fontsize=7)

        # Subplot title
        anchor_str = (
            ep["anchor"].strftime("%Y-%m-%d")
            if hasattr(ep["anchor"], "strftime")
            else str(ep["anchor"])
        )
        ax.set_title(
            f"#{plot_idx + 1} ({anchor_str}) "
            f"{entry['cov_label']}={entry['mean_cov']:.1f}U "
            f"RMSE={entry['rmse']:.2f}",
            fontsize=7,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, forecast_length * interval_mins / 60)
        ax.tick_params(axis="both", labelsize=6)

        # Legend only on first subplot
        if plot_idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=5)

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(
        f"{title}\nAvg RMSE: {avg_rmse:.3f} | {n} episodes",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
