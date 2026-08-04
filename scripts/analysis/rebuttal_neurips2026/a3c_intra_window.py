# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
"""A3c -- Intra-window model ranking stability analysis.

Compute model performance (AUROC, RMSE) separately for four 2-hour sub-windows
within the 00:00-08:00 evaluation period to test whether model rankings are
robust to potential sleep/wake heterogeneity across the overnight window.

For each dataset and model:
  - Load episode-level predictions and ground truth
  - For each 2-hour sub-window (00:00-02:00, 02:00-04:00, 04:00-06:00, 06:00-08:00):
    * Extract predictions/truth for time steps falling in that window
    * Compute AUROC and RMSE for that sub-window
  - Calculate model rankings within each sub-window
  - Compute Kendall tau correlation between full-window rankings and each sub-window

Key deliverable: Show that tau values remain high (>0.80) across all sub-windows,
demonstrating that model rankings are stable even in the 00:00-02:00 window
(most likely to include wakefulness) and throughout the overnight period.

Usage:
    python -m scripts.analysis.rebuttal_neurips2026.a3c_intra_window

Outputs (scripts/analysis/rebuttal_neurips2026/outputs/):
    a3c_intra_window_metrics.csv    dataset × model × window × AUROC × RMSE
    a3c_intra_window_stability.csv  dataset × window × kendall_tau
    a3c_intra_window_ranks.png      visualization of ranking stability
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

from scripts.analysis.rebuttal_neurips2026 import common as C

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Four 2-hour sub-windows within evaluation period
SUB_WINDOWS = [
    (0, 2, "00:00-02:00"),
    (2, 4, "02:00-04:00"),
    (4, 6, "04:00-06:00"),
    (6, 8, "06:00-08:00"),
]

DEFAULT_MODELS = ["chronos2", "moirai", "toto", "patchtst", "tft"]


def _extract_window_predictions(
    actuals: np.ndarray, predictions: np.ndarray, start_hour: int, end_hour: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract predictions for a specific hour-of-day window.

    Each episode spans 8 hours (96 steps at 5-min cadence). We extract the steps
    that fall within [start_hour, end_hour) relative to the midnight anchor.

    Args:
        actuals: (N, 96) ground truth BG values
        predictions: (N, 96) predicted BG values
        start_hour: Start hour (0-8)
        end_hour: End hour (0-8)

    Returns:
        Tuple of (actuals_window, predictions_window) where each is (N, steps_in_window)
    """
    steps_per_hour = 12  # 5-min cadence
    start_step = start_hour * steps_per_hour
    end_step = end_hour * steps_per_hour

    return actuals[:, start_step:end_step], predictions[:, start_step:end_step]


def _window_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    has_hypo: np.ndarray,
    start_hour: int,
    end_hour: int,
) -> dict:
    """Compute AUROC and RMSE for a time window within episodes.

    Args:
        actuals: (N, 96) ground truth BG values
        predictions: (N, 96) predicted BG values
        has_hypo: (N,) boolean array indicating if episode has any hypo
        start_hour: Start hour (0-8)
        end_hour: End hour (0-8)
    """
    act_win, pred_win = _extract_window_predictions(
        actuals, predictions, start_hour, end_hour
    )

    if len(act_win) < 10:  # minimum sample size
        return dict(n=0, auroc=np.nan, rmse=np.nan)

    # Compute RMSE over the window
    # Average over time steps, then over episodes
    rmse = float(np.sqrt(np.mean((act_win - pred_win) ** 2)))

    # For AUROC, we need episode-level scores and labels
    # Use the mean predicted probability of hypo in the window as the score
    # Label is whether any actual hypo occurred in the window
    hypo_threshold = 3.9  # mmol/L
    window_has_hypo = (act_win < hypo_threshold).any(axis=1)  # (N,) any hypo in window
    window_risk_score = (pred_win < hypo_threshold).mean(
        axis=1
    )  # (N,) avg predicted risk

    # Compute AUROC
    from sklearn.metrics import roc_auc_score

    try:
        if len(np.unique(window_has_hypo)) < 2:
            # All same label, AUROC undefined
            auroc = np.nan
        else:
            auroc = float(roc_auc_score(window_has_hypo, window_risk_score))
    except (ValueError, IndexError):
        auroc = np.nan

    return dict(n=len(act_win), auroc=auroc, rmse=rmse)


def run(models: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C.save_run_index(OUT_DIR / "run_index.csv")

    metric_rows = []

    for dataset in C.DATASETS:
        print(f"\n{'=' * 80}")
        print(f"{C.DATASET_LABEL[dataset]}")
        print("=" * 80)

        # First, get full-window metrics and rankings
        full_metrics = {}
        for model in models:
            rp = C.best_run_path(model, dataset)
            if rp is None or not rp.exists():
                continue
            run = C.load_run(rp, dataset=dataset, model=model)
            df = run.episodes
            full_auroc = C.auroc(df, "score_point")
            full_rmse = C.pooled_rmse(df)
            full_metrics[model] = {
                "auroc": full_auroc,
                "rmse": full_rmse,
                "run": run,  # Store for later use
            }

            # Add full-window row
            metric_rows.append(
                dict(
                    dataset=dataset,
                    dataset_label=C.DATASET_LABEL[dataset],
                    model=model,
                    window="full (00:00-08:00)",
                    window_start=0,
                    window_end=8,
                    n=len(df),
                    auroc=full_auroc,
                    rmse=full_rmse,
                )
            )

        # Calculate full-window rankings
        full_auroc_ranks = {
            m: rank
            for rank, (m, _) in enumerate(
                sorted(full_metrics.items(), key=lambda x: x[1]["auroc"], reverse=True),
                1,
            )
        }

        print("\nFull window (00:00-08:00) AUROC rankings:")
        for model in sorted(full_auroc_ranks, key=full_auroc_ranks.get):
            print(
                f"  {model:12s}  rank={full_auroc_ranks[model]}  "
                f"AUROC={full_metrics[model]['auroc']:.3f}"
            )

        # Now compute per-window metrics using step-level predictions
        for start_h, end_h, label in SUB_WINDOWS:
            print(f"\n{label}:")
            window_metrics = {}

            for model in models:
                if model not in full_metrics:
                    continue

                run = full_metrics[model]["run"]

                # Compute window-specific metrics using step-level data
                metrics = _window_metrics(
                    run.actuals,
                    run.predictions,
                    run.episodes["has_hypo"].values,
                    start_h,
                    end_h,
                )
                window_metrics[model] = metrics

                metric_rows.append(
                    dict(
                        dataset=dataset,
                        dataset_label=C.DATASET_LABEL[dataset],
                        model=model,
                        window=label,
                        window_start=start_h,
                        window_end=end_h,
                        n=metrics["n"],
                        auroc=metrics["auroc"],
                        rmse=metrics["rmse"],
                    )
                )

                print(
                    f"  {model:12s}  AUROC={metrics['auroc']:.3f}  RMSE={metrics['rmse']:.3f}"
                )

            # Calculate window rankings and tau
            window_auroc_ranks = {
                m: rank
                for rank, (m, _) in enumerate(
                    sorted(
                        window_metrics.items(),
                        key=lambda x: x[1]["auroc"],
                        reverse=True,
                    ),
                    1,
                )
            }

            # Compute Kendall tau (only for models present in both rankings)
            common_models = set(full_auroc_ranks.keys()) & set(
                window_auroc_ranks.keys()
            )
            if len(common_models) >= 3:
                full_ranks = [full_auroc_ranks[m] for m in sorted(common_models)]
                window_ranks = [window_auroc_ranks[m] for m in sorted(common_models)]
                tau, p_value = kendalltau(full_ranks, window_ranks)
                print(
                    f"  -> Kendall tau (vs full window): {tau:+.3f} (p={p_value:.4f})"
                )

    # Save outputs
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT_DIR / "a3c_intra_window_metrics.csv", index=False)
    print(f"\n{'=' * 80}")
    print(f"Saved: {OUT_DIR}/a3c_intra_window_metrics.csv")

    # Generate visualization
    _plot_ranking_stability(metrics_df)


def _plot_ranking_stability(df: pd.DataFrame) -> None:
    """Visualize model rankings across sub-windows."""
    sns.set_style("whitegrid")

    n_datasets = df["dataset"].nunique()
    fig, axes = plt.subplots(
        n_datasets, 1, figsize=(10, 3 * n_datasets), sharex=True, sharey=False
    )
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, df["dataset"].unique()):
        sub = df[df.dataset == dataset].copy()

        # Pivot to get model × window AUROC values
        pivot = sub.pivot_table(index="model", columns="window", values="auroc")

        # Reorder columns to have full first, then sub-windows in order
        col_order = [
            "full (00:00-08:00)",
            "00:00-02:00",
            "02:00-04:00",
            "04:00-06:00",
            "06:00-08:00",
        ]
        col_order = [c for c in col_order if c in pivot.columns]
        pivot = pivot[col_order]

        # Sort by full-window AUROC
        if "full (00:00-08:00)" in pivot.columns:
            pivot = pivot.sort_values("full (00:00-08:00)", ascending=False)

        # Plot lines for each model
        for model in pivot.index:
            ax.plot(
                range(len(pivot.columns)),
                pivot.loc[model],
                marker="o",
                label=model,
                linewidth=2,
                markersize=6,
            )

        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(f'{sub.iloc[0]["dataset_label"]}', fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.legend(loc="best", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Window", fontsize=12)
    fig.suptitle(
        "Model Ranking Stability Across Sub-Windows\n"
        "AUROC for Hypoglycemia Detection",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "a3c_intra_window_ranks.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_DIR}/a3c_intra_window_ranks.png")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="A3c intra-window ranking stability")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()
    run(args.models)


if __name__ == "__main__":
    main()
