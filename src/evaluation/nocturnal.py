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
from src.evaluation.metrics.probabilistic import (
    compute_wql,
    compute_brier_score,
    compute_coverage,
    compute_sharpness,
    compute_coverage_by_step,
    compute_sharpness_by_step,
    compute_mace,
)
from src.evaluation.metrics.shape import compute_dilate_metrics, DILATE_COLUMNS

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
    probabilistic: bool = False,
    compute_dilate: bool = True,
) -> Dict[str, Any]:
    """Evaluate model on midnight-anchored nocturnal forecasting task.

    Builds midnight episodes per patient and calls model.predict_batch() to
    forecast all episodes in a single call. When probabilistic=True, passes
    quantile_levels to predict_batch(), extracts the median (0.5 quantile) as
    the point forecast for RMSE, and computes WQL and Brier@3.9 from the full
    quantile distribution.

    Also computes per-episode discontinuity (absolute jump between last context
    BG and first predicted BG).

    Args:
        model: Model implementing predict_batch(panel_df, episode_col).
            Must support probabilistic forecasting (supports_probabilistic_forecast)
            when probabilistic=True.
        holdout_data: Flat DataFrame with all holdout patients.
        context_length: Context window size in steps.
        forecast_length: Forecast horizon in steps.
        target_col: BG column name.
        covariate_cols: Covariate column names (e.g., ["iob"]).
        interval_mins: Sampling interval in minutes.
        probabilistic: If True, pass quantile_levels to predict_batch() and
            compute WQL/Brier alongside RMSE.
        compute_dilate: If False, skip Soft-DTW/DILATE metrics (saves
            O(n_episodes * forecast_length^2) work on large runs).

    Returns:
        Dict with overall_rmse, mean_discontinuity, total_episodes, per_patient,
        per_episode. When probabilistic=True, also includes overall_wql,
        overall_brier, and quantile_levels.
    """
    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()

    # --- Phase 1: Build episodes for all patients ---
    # Collect episodes and track which patient each belongs to.
    episode_col = "episode_id"
    episode_metadata = []
    context_dfs = []

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

        # Set DatetimeIndex if not already set
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

        # In multi-target mode, joint_target_cols (e.g. ["bg_mM", "iob"]) must be
        # included as covariates so they appear in context_df for stacking.
        effective_covs = list(covariate_cols) if covariate_cols else []
        if hasattr(model, "config") and getattr(model.config, "is_multitarget", False):
            for col in getattr(model.config, "joint_target_cols", []):
                if col != target_col and col not in effective_covs:
                    effective_covs.append(col)
        episodes, _ = build_midnight_episodes(
            patient_df,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            covariate_cols=effective_covs or None,
            interval_mins=interval_mins,
        )

        if not episodes:
            logger.info("  Patient %s: no valid midnight episodes", patient_id)
            continue

        for i, ep in enumerate(episodes):
            if len(ep["target_bg"]) < forecast_length:
                continue
            ep_id = f"{patient_id}::ep{i:03d}"
            ctx = ep["context_df"].copy().reset_index(names="datetime")
            ctx["p_num"] = patient_id
            ctx[episode_col] = ep_id
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
            "mean_discontinuity": float("nan"),
            "total_episodes": 0,
            "per_patient": [],
            "per_episode": [],
        }

    # --- Phase 2: Predict ---
    n_unique_patients = len(set(m["patient_id"] for m in episode_metadata))
    logger.info(
        "Evaluating %d midnight episodes across %d patients",
        len(episode_metadata),
        n_unique_patients,
    )

    panel_df = pd.concat(context_dfs, ignore_index=True)

    if probabilistic:
        quantile_levels = (
            getattr(model.config, "quantile_levels", None)
            or model.DEFAULT_QUANTILE_LEVELS
        )
        # Ensure 0.5 is present so we can extract the median as point forecast.
        if 0.5 not in quantile_levels:
            quantile_levels = sorted(set(quantile_levels) | {0.5})
        median_idx = quantile_levels.index(0.5)

        # Determine which coverage levels (50/90/95) the model's quantile
        # range actually supports.  A level L requires bounds at (1-L)/2 and
        # (1+L)/2; if those fall outside [min_q, max_q] the metric would be
        # silently clamped, so we skip it entirely.
        _CANDIDATE_LEVELS = (0.5, 0.80, 0.90, 0.95)
        q_min, q_max = min(quantile_levels), max(quantile_levels)
        supported_levels = []
        for lvl in _CANDIDATE_LEVELS:
            lo, hi = (1.0 - lvl) / 2.0, (1.0 + lvl) / 2.0
            if lo >= q_min - 1e-9 and hi <= q_max + 1e-9:
                supported_levels.append(lvl)
            else:
                logger.info(
                    "Skipping %d%% coverage/sharpness — needs quantiles "
                    "%.4f/%.4f but model range is [%.3f, %.3f]",
                    int(lvl * 100), lo, hi, q_min, q_max,
                )

        logger.info(
            "Probabilistic mode — computing WQL and Brier@%.1f (quantiles: %s)",
            HYPO_THRESHOLD_MMOL,
            quantile_levels,
        )
        batch_results = model.predict_batch(
            panel_df, episode_col=episode_col, quantile_levels=quantile_levels
        )
    else:
        batch_results = model.predict_batch(panel_df, episode_col=episode_col)

    # Build episode_id -> context-BG lookups for quick access
    ctx_bg_by_id = {
        m[episode_col]: ctx[target_col].values
        for ctx, m in zip(context_dfs, episode_metadata)
        if target_col in ctx.columns
    }

    all_episode_results = []
    patient_episodes: Dict[str, list] = {}
    discontinuities = []

    # Accumulators for Tier 3 raw arrays
    all_q_forecasts: List[np.ndarray] = []  # probabilistic only
    all_predictions: List[
        np.ndarray
    ] = []  # point forecasts (median when probabilistic)
    all_actuals_arrays: List[np.ndarray] = []
    all_episode_ids: List[str] = []

    for ctx_df, meta in zip(context_dfs, episode_metadata):
        ep_id = meta["episode_id"]
        target = np.asarray(meta["target_bg"])
        context_bg = ctx_bg_by_id.get(ep_id)

        raw = batch_results.get(ep_id)
        if raw is None:
            logger.warning("Episode %s: no prediction returned, skipping", ep_id)
            continue
        raw = np.asarray(raw)

        if probabilistic:
            q_forecast = raw[:, : len(target)]
            pred = q_forecast[median_idx]  # median as point forecast
            ep_wql = float(compute_wql(q_forecast, target, quantile_levels))
            ep_brier = float(compute_brier_score(q_forecast, target, quantile_levels))

            ep_prob = {}
            for lvl in supported_levels:
                suffix = str(int(lvl * 100))
                ep_prob[f"coverage_{suffix}"] = float(
                    compute_coverage(q_forecast, target, quantile_levels, level=lvl)
                )
                ep_prob[f"sharpness_{suffix}"] = float(
                    compute_sharpness(q_forecast, quantile_levels, level=lvl)
                )

            all_q_forecasts.append(q_forecast)
        else:
            pred = raw[: len(target)]

        # Accumulate arrays for Tier 3 storage (every episode, all modes)
        all_predictions.append(pred)
        all_actuals_arrays.append(target)
        all_episode_ids.append(ep_id)

        ep_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

        # Shape-aware metrics (DILATE at 3 gamma values → 9 scalars)
        ep_dilate = compute_dilate_metrics(pred, target) if compute_dilate else {}

        # Discontinuity: absolute BG jump at the context-forecast boundary.
        disc = float("nan")
        if context_bg is not None and len(context_bg) > 0 and len(pred) > 0:
            disc = abs(float(context_bg[-1]) - float(pred[0]))
        discontinuities.append(disc)

        ep_result = {
            "patient_id": meta["patient_id"],
            "anchor": meta["anchor"].isoformat(),
            "rmse": ep_rmse,
            "discontinuity": disc,
            **ep_dilate,
            "pred": pred.tolist(),
            "target_bg": target.tolist(),
            "context_bg": context_bg.tolist() if context_bg is not None else None,
        }
        if probabilistic:
            ep_result["wql"] = ep_wql
            ep_result["brier"] = ep_brier
            ep_result.update(ep_prob)

        all_episode_results.append(ep_result)

        pid = meta["patient_id"]
        patient_episodes.setdefault(pid, []).append((pred, target))

    if not all_episode_results:
        logger.warning("All episodes were dropped — no predictions")
        return {
            "overall_rmse": float("nan"),
            "mean_discontinuity": float("nan"),
            "total_episodes": 0,
            "per_patient": [],
            "per_episode": [],
        }

    # Per-patient aggregate
    all_patient_results = []
    for pid, ep_list in patient_episodes.items():
        preds = np.concatenate([p for p, _ in ep_list])
        targets = np.concatenate([t for _, t in ep_list])
        metrics = compute_regression_metrics(preds, targets)

        patient_result = {
            "patient_id": pid,
            "episodes": len(ep_list),
            **metrics,
        }

        if probabilistic:
            pid_eps = [ep for ep in all_episode_results if ep["patient_id"] == pid]
            patient_result["wql"] = float(np.mean([ep["wql"] for ep in pid_eps]))
            patient_result["brier"] = float(np.mean([ep["brier"] for ep in pid_eps]))

        all_patient_results.append(patient_result)

        log_msg = (
            f"  Patient {pid}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}"
        )
        if probabilistic:
            log_msg += f", WQL={patient_result['wql']:.4f}, Brier={patient_result['brier']:.4f}"
        log_msg += f" ({len(ep_list)} midnight episodes)"
        logger.info(log_msg)

    # Overall metrics (concatenated predictions, consistent with per-patient)
    all_preds = np.concatenate([ep["pred"] for ep in all_episode_results])
    all_targets = np.concatenate([ep["target_bg"] for ep in all_episode_results])
    overall_rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    # Mean discontinuity (ignoring NaN)
    valid_discs = [d for d in discontinuities if not np.isnan(d)]
    mean_disc = float(np.mean(valid_discs)) if valid_discs else float("nan")

    results = {
        "overall_rmse": overall_rmse,
        "mean_discontinuity": mean_disc,
        "total_episodes": len(all_episode_results),
        "per_patient": all_patient_results,
        "per_episode": all_episode_results,
        # Internal: raw arrays for Tier 3 storage (underscore = not JSON-serializable)
        "_predictions": np.stack(all_predictions),  # (n_eps, fh)
        "_actuals_array": np.stack(all_actuals_arrays),  # (n_eps, fh)
        "_episode_ids": all_episode_ids,
    }

    # Overall DILATE means (computed for all modes — only needs point forecasts)
    if compute_dilate:
        for col in DILATE_COLUMNS:
            vals = [
                ep[col]
                for ep in all_episode_results
                if not np.isnan(ep.get(col, float("nan")))
            ]
            results[f"overall_{col}"] = float(np.mean(vals)) if vals else float("nan")

    log_msg = f"Nocturnal evaluation: {overall_rmse:.4f} RMSE, {mean_disc:.4f} mean discontinuity"
    if probabilistic:
        results["overall_wql"] = float(
            np.mean([ep["wql"] for ep in all_episode_results])
        )
        results["overall_brier"] = float(
            np.mean([ep["brier"] for ep in all_episode_results])
        )
        for lvl in supported_levels:
            suffix = str(int(lvl * 100))
            results[f"overall_coverage_{suffix}"] = float(
                np.mean([ep[f"coverage_{suffix}"] for ep in all_episode_results])
            )
            results[f"overall_sharpness_{suffix}"] = float(
                np.mean([ep[f"sharpness_{suffix}"] for ep in all_episode_results])
            )
        # MACE computed from stacked arrays (all timesteps, all quantiles)
        q_stacked = np.concatenate(all_q_forecasts, axis=1)  # (n_q, total_timesteps)
        actuals_concat = np.concatenate(all_actuals_arrays)  # (total_timesteps,)
        results["overall_mace"] = float(
            compute_mace(q_stacked, actuals_concat, quantile_levels)
        )
        results["quantile_levels"] = quantile_levels
        results["_q_forecasts"] = np.stack(all_q_forecasts)  # (n_eps, n_q, fh)

        # Per-step coverage and sharpness across all episodes (shape: (fh,) each).
        # Stored in Tier 3 for calibration plots; not in summary.csv.
        # Only computed for levels the model's quantile range supports.
        q_batch = results["_q_forecasts"]  # (n_eps, n_q, fh)
        act_batch = results["_actuals_array"]  # (n_eps, fh)
        for lvl in supported_levels:
            suffix = str(int(lvl * 100))
            results[f"_coverage_by_step_{suffix}"] = compute_coverage_by_step(
                q_batch, act_batch, quantile_levels, level=lvl
            )
            results[f"_sharpness_by_step_{suffix}"] = compute_sharpness_by_step(
                q_batch, quantile_levels, level=lvl
            )

        log_msg += (
            f", {results['overall_wql']:.4f} WQL"
            f", {results['overall_brier']:.4f} Brier@{HYPO_THRESHOLD_MMOL}"
            f", {results['overall_mace']:.4f} MACE"
        )
        for lvl in supported_levels:
            suffix = str(int(lvl * 100))
            log_msg += f", cov{suffix}={results[f'overall_coverage_{suffix}']:.3f}"
    log_msg += f" over {len(all_episode_results)} midnight episodes"
    logger.info(log_msg)

    return results


def plot_stage_comparison_auto(
    stage1_per_episode: List[Dict[str, Any]],
    stage2_per_episode: List[Dict[str, Any]],
    stage1_rmse: float,
    stage2_rmse: float,
    output_path: Path,
    model_name: str,
    dataset_name: str,
    patient_id: str,
    context_hours_to_show: int = 3,
) -> None:
    """Generate Stage 1 vs Stage 2 comparison plot.

    Args:
        stage1_per_episode: Episode results from Stage 1 evaluate_nocturnal_forecasting.
        stage2_per_episode: Episode results from Stage 2 evaluate_nocturnal_forecasting.
        stage1_rmse: Overall Stage 1 RMSE.
        stage2_rmse: Overall Stage 2 RMSE.
        output_path: Directory to save plot.
        model_name: Model type string (e.g. "chronos2").
        dataset_name: Dataset name.
        patient_id: Target patient ID.
        context_hours_to_show: Hours of context to display.
    """
    s1_by_anchor = {
        ep["anchor"]: ep for ep in stage1_per_episode if ep.get("context_bg")
    }
    s2_by_anchor = {
        ep["anchor"]: ep for ep in stage2_per_episode if ep.get("context_bg")
    }
    common_anchors = sorted(set(s1_by_anchor.keys()) & set(s2_by_anchor.keys()))

    if not common_anchors:
        logger.warning("No common episodes with context data for comparison plot")
        return

    n_episodes = len(common_anchors)
    ncols = min(4, n_episodes)
    nrows = (n_episodes + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False
    )
    axes = axes.flatten()

    context_steps = context_hours_to_show * STEPS_PER_HOUR

    for i, anchor in enumerate(common_anchors):
        if i >= len(axes):
            break
        ax = axes[i]
        s1 = s1_by_anchor[anchor]
        s2 = s2_by_anchor[anchor]

        ctx_full = np.array(s2["context_bg"])
        ctx = ctx_full[-context_steps:] if len(ctx_full) > context_steps else ctx_full
        tgt = np.array(s2["target_bg"])
        pred_s1 = np.array(s1["pred"])
        pred_s2 = np.array(s2["pred"])

        t_ctx = (np.arange(len(ctx)) - len(ctx)) / STEPS_PER_HOUR
        t_pred = np.arange(len(tgt)) / STEPS_PER_HOUR

        ax.plot(t_ctx, ctx, "b-", lw=1.5, label="Context")
        ax.plot(t_pred, tgt, "k-", lw=2, label="Actual")
        ax.plot(
            t_pred,
            pred_s1,
            color="orange",
            lw=1.5,
            ls="--",
            alpha=0.9,
            label=f"S1 ({s1['rmse']:.2f})",
        )
        ax.plot(
            t_pred,
            pred_s2,
            color="#1f77b4",
            lw=2,
            alpha=0.9,
            label=f"S2 ({s2['rmse']:.2f})",
        )

        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.axhline(HYPO_THRESHOLD_MMOL, color="crimson", ls="--", alpha=0.3, lw=1)
        ax.set_ylim(0, 18)
        ax.set_xlabel("Hours from midnight", fontsize=9)
        ax.set_ylabel("BG (mmol/L)", fontsize=9)
        ax.set_title(anchor[:10], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    for j in range(n_episodes, len(axes)):
        axes[j].set_visible(False)

    delta = stage1_rmse - stage2_rmse
    direction = "improvement" if delta > 0 else "regression"
    fig.suptitle(
        f"{model_name.upper()} Stage 1 vs Stage 2 — {patient_id} ({dataset_name})\n"
        f"Population RMSE: {stage1_rmse:.3f}  |  Personalized RMSE: {stage2_rmse:.3f}  |  "
        f"Delta: {abs(delta):.3f} ({direction})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    plot_file = Path(output_path) / "stage_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Comparison plot saved to: %s", plot_file)


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

        ctx = (
            ctx_full[-context_steps_to_show:]
            if len(ctx_full) > context_steps_to_show
            else ctx_full
        )

        t_ctx = (np.arange(len(ctx)) - len(ctx)) / STEPS_PER_HOUR
        t_pred = np.arange(len(tgt)) / STEPS_PER_HOUR

        ax.plot(t_ctx, ctx, "b-", lw=1.5, label="BG (context)")
        ax.plot(t_pred, tgt, "g-", lw=2, label="BG (actual)")
        ax.plot(t_pred, pred, "r--", lw=2, alpha=0.8, label="BG (forecast)")
        ax.axvline(0, color="gray", ls=":", lw=1.5, label="Midnight")
        ax.axhline(
            HYPO_THRESHOLD_MMOL,
            color="crimson",
            ls="--",
            alpha=0.4,
            lw=1,
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
