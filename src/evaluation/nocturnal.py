# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
Nocturnal hypoglycemia evaluation utilities.

Provides midnight-anchored evaluation of glucose forecasting models — the primary
clinical evaluation mode for this project. Context ends at midnight (00:00), and
the model forecasts 6 hours overnight when hypoglycemia risk is highest.

These functions are shared between:
  - scripts/experiments/nocturnal_hypo_eval.py  (full evaluation script)
  - scripts/experiments/per_patient_finetune.py (Stage 2 fine-tuning script)

Distinct from sliding-window evaluation (general forecast accuracy across all
times of day): these two modes produce different RMSE numbers and must never
be compared on the same leaderboard.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.utils import get_patient_column
from src.evaluation.episode_builders import build_midnight_episodes
from src.evaluation.metrics import compute_regression_metrics

# Constants
SAMPLING_INTERVAL_MINUTES = 5
STEPS_PER_HOUR = 60 // SAMPLING_INTERVAL_MINUTES
HYPO_THRESHOLD_MMOL = 3.9

logger = logging.getLogger(__name__)


def evaluate_nocturnal_forecasting(
    model,
    holdout_data: pd.DataFrame,
    context_length: int,
    forecast_length: int,
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
    interval_mins: int = SAMPLING_INTERVAL_MINUTES,
) -> Dict[str, Any]:
    """Evaluate a model on the midnight-anchored nocturnal forecasting task.

    Builds midnight episodes per patient and evaluates each episode individually.
    This per-episode approach ensures compatibility with models like TTM that
    use ForecastDFDataset internally (which creates sliding windows rather than
    treating pre-windowed inputs as separate samples).

    Args:
        model: Model implementing predict(data) -> np.ndarray.
        holdout_data: Flat DataFrame with one or more patients.
            Must have 'bg_mM', 'datetime', and a patient ID column.
        context_length: Context window size in steps.
        forecast_length: Forecast horizon in steps.
        target_col: BG column name.
        covariate_cols: Covariate column names (e.g., ["iob"]).
        interval_mins: Sampling interval in minutes.

    Returns:
        Dict with keys:
            overall_rmse: float
            total_episodes: int
            per_patient: List[Dict] — one entry per patient with rmse, mae, episodes
            per_episode: List[Dict] — one entry per episode with pred, target_bg, context_bg
    """
    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()

    # --- Phase 1: Build midnight episodes for all patients ---
    episode_metadata = []
    context_dfs = []

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

        # Ensure DatetimeIndex
        if not isinstance(patient_df.index, pd.DatetimeIndex):
            time_col = "datetime" if "datetime" in patient_df.columns else None
            if time_col:
                patient_df[time_col] = pd.to_datetime(patient_df[time_col])
                patient_df = patient_df.set_index(time_col).sort_index()
            else:
                logger.warning(
                    "Patient %s: no datetime column or DatetimeIndex, skipping",
                    patient_id,
                )
                continue

        episodes, skip_stats = build_midnight_episodes(
            patient_df,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            covariate_cols=covariate_cols,
            interval_mins=interval_mins,
        )

        if not episodes:
            logger.info("  Patient %s: no valid midnight episodes", patient_id)
            if skip_stats["skipped_bg_nan"] > 0:
                logger.debug(
                    "    Skipped %d/%d anchors due to missing BG",
                    skip_stats["skipped_bg_nan"],
                    skip_stats["total_anchors"],
                )
            continue

        for i, ep in enumerate(episodes):
            ep_id = f"{patient_id}::ep{i:03d}"

            ctx = ep["context_df"].copy()
            # Convert DatetimeIndex to column for models that expect it (e.g., TTM)
            ctx = ctx.reset_index(names="datetime")
            ctx["episode_id"] = ep_id
            ctx["group"] = ep_id
            context_dfs.append(ctx)

            episode_metadata.append(
                {
                    "episode_id": ep_id,
                    "patient_id": str(patient_id),
                    "anchor": ep["anchor"],
                    "target_bg": ep["target_bg"],
                }
            )

    if not episode_metadata:
        logger.warning("No valid midnight episodes found across all patients")
        return {
            "overall_rmse": float("nan"),
            "total_episodes": 0,
            "per_patient": [],
            "per_episode": [],
        }

    # --- Phase 2: Predict per episode ---
    # Iterate per-episode rather than batching: models like TTM use ForecastDFDataset
    # internally (sliding windows), so batching episodes would cause window leakage.
    logger.info(
        "Evaluating %d midnight episodes across %d patients",
        len(episode_metadata),
        len(set(m["patient_id"] for m in episode_metadata)),
    )

    all_episode_results = []
    patient_episodes: Dict[str, list] = {}

    for ctx_df, meta in zip(context_dfs, episode_metadata):
        pred = np.asarray(model.predict(ctx_df))
        if pred.ndim == 3:
            pred = pred[0, :, 0]
        elif pred.ndim == 2:
            pred = pred.flatten()

        target = meta["target_bg"]
        if len(pred) > len(target):
            pred = pred[: len(target)]

        ep_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        context_bg = ctx_df[target_col].values if target_col in ctx_df.columns else None

        all_episode_results.append(
            {
                "patient_id": meta["patient_id"],
                "anchor": meta["anchor"].isoformat(),
                "rmse": ep_rmse,
                "pred": pred.tolist(),
                "target_bg": target.tolist(),
                "context_bg": context_bg.tolist() if context_bg is not None else None,
            }
        )

        pid = meta["patient_id"]
        patient_episodes.setdefault(pid, []).append((pred, target))

    # Per-patient aggregates
    all_patient_results = []
    for pid, ep_list in patient_episodes.items():
        preds = np.concatenate([p for p, _ in ep_list])
        targets = np.concatenate([t for _, t in ep_list])
        metrics = compute_regression_metrics(preds, targets)

        all_patient_results.append(
            {
                "patient_id": pid,
                "episodes": len(ep_list),
                **metrics,
            }
        )

        logger.info(
            "  Patient %s: RMSE=%.3f, MAE=%.3f (%d midnight episodes)",
            pid,
            metrics["rmse"],
            metrics["mae"],
            len(ep_list),
        )

    # Overall RMSE (concatenated across all episodes)
    all_preds = np.concatenate([np.array(ep["pred"]) for ep in all_episode_results])
    all_targets = np.concatenate(
        [np.array(ep["target_bg"]) for ep in all_episode_results]
    )
    overall_rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    logger.info(
        "Nocturnal evaluation complete: %.4f RMSE over %d midnight episodes",
        overall_rmse,
        len(all_episode_results),
    )

    return {
        "overall_rmse": overall_rmse,
        "total_episodes": len(all_episode_results),
        "per_patient": all_patient_results,
        "per_episode": all_episode_results,
    }


def plot_best_worst_episodes(
    per_episode: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    dataset_name: str,
    is_finetuned: bool,
    n_best: int = 4,
    n_worst: int = 4,
    context_hours_to_show: int = 3,
) -> None:
    """Plot the best and worst RMSE midnight episodes for visual inspection.

    Args:
        per_episode: List of episode result dicts (from evaluate_nocturnal_forecasting).
        output_path: Directory to save plot.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        is_finetuned: Whether using fine-tuned or zero-shot model.
        n_best: Number of best episodes to plot.
        n_worst: Number of worst episodes to plot.
        context_hours_to_show: Hours of context to display in plot.
    """
    episodes_with_context = [ep for ep in per_episode if ep.get("context_bg")]
    if not episodes_with_context:
        logger.warning("No episodes with context data available for plotting")
        return

    sorted_episodes = sorted(episodes_with_context, key=lambda x: x["rmse"])
    best_episodes = sorted_episodes[:n_best]
    worst_episodes = sorted_episodes[-n_worst:]

    examples = []
    for ep in best_episodes:
        examples.append({"label": "BEST", **ep})
    for ep in worst_episodes:
        examples.append({"label": "WORST", **ep})

    n = len(examples)
    if n == 0:
        return

    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    context_steps_to_show = context_hours_to_show * STEPS_PER_HOUR

    for i, ex in enumerate(examples):
        if i >= len(axes):
            break
        ax = axes[i]
        ctx_full = np.array(ex["context_bg"])
        tgt = np.array(ex["target_bg"])
        pred = np.array(ex["pred"])

        ctx = ctx_full[-context_steps_to_show:] if len(ctx_full) > context_steps_to_show else ctx_full

        t_ctx = (np.arange(len(ctx)) - len(ctx)) / STEPS_PER_HOUR
        t_pred = np.arange(len(tgt)) / STEPS_PER_HOUR

        ax.plot(t_ctx, ctx, "b-", lw=1.5, label="BG (context)")
        ax.plot(t_pred, tgt, "g-", lw=2, label="BG (actual)")
        ax.plot(t_pred, pred, "r--", lw=2, alpha=0.8, label="BG (forecast)")
        ax.axvline(0, color="gray", ls=":", lw=1.5, label="Midnight")
        ax.axhline(
            HYPO_THRESHOLD_MMOL, color="crimson", ls="--", alpha=0.4, lw=1,
            label="Hypo threshold",
        )

        ax.set_ylabel("BG (mmol/L)", fontsize=9)
        ax.set_ylim(0, 18)
        ax.set_xlabel("Hours from midnight", fontsize=9)

        label_color = "green" if ex["label"] == "BEST" else "red"
        ax.set_title(
            f"{ex['label']}: {ex['patient_id']}\nRMSE={ex['rmse']:.2f} mmol/L",
            fontsize=10,
            color=label_color,
        )
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if n > 0:
        axes[0].legend(fontsize=7, loc="upper right")

    mode_str = "Fine-tuned" if is_finetuned else "Zero-Shot"
    fig.suptitle(
        f"{model_name.upper()} {mode_str} Nocturnal Forecasts - {dataset_name}\n"
        f"Top {n_best} Best and {n_worst} Worst RMSE Episodes",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    plot_file = Path(output_path) / "best_worst_forecasts.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to: %s", plot_file)
