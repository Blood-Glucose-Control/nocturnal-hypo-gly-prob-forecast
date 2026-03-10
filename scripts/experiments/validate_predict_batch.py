#!/usr/bin/env python3
r"""
Validate that predict_batch() produces results consistent with per-episode predict().

Runs the same midnight-episode evaluation **twice** back-to-back on the same
model instance:

  A) Sequential: model.predict(ctx_df) per episode   ← current production path
  B) Batch:      model.predict_batch(panel_df)        ← new predict_batch path

Reports per-episode max-absolute-difference and exits non-zero if any episode
exceeds the configured tolerance, making this suitable as a CI smoke-test.

Why the results should agree
-----------------------------
Fine-tuned TiDE / Chronos2 / TTM all delegate ``_predict_batch()`` to
``super()._predict_batch()``, which is the same sequential
``groupby + self.predict()`` loop as path A.  Results should be **bit-for-bit
identical** (max_diff == 0.0).

Zero-shot TTM calls ``TimeSeriesForecastingPipeline`` once with all episodes
as a panel — batching may cause tiny floating-point differences (~1e-6).

Usage
-----
    # Fine-tuned TTM on aleppo_2017 (50 episodes, strict tolerance)
    python scripts/experiments/validate_predict_batch.py \
        --model ttm \
        --dataset aleppo_2017 \
        --checkpoint trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt \
        --context-length 512 --forecast-length 96

    # Fine-tuned Chronos2 on aleppo_2017
    python scripts/experiments/validate_predict_batch.py \
        --model chronos2 \
        --dataset aleppo_2017 \
        --checkpoint trained_models/artifacts/chronos2/2026-02-28_05:54_RID20260228_055400_391511_holdout_workflow/resumed_training/model.pt \
        --context-length 512 --forecast-length 96

    # Fine-tuned TiDE on aleppo_2017
    python scripts/experiments/validate_predict_batch.py \
        --model tide \
        --dataset aleppo_2017 \
        --checkpoint trained_models/artifacts/tide/2026-02-28_21:28_RID20260228_212852_496983_holdout_workflow/model.pt \
        --context-length 512 --forecast-length 96

    # Zero-shot TTM (looser tolerance expected)
    python scripts/experiments/validate_predict_batch.py \
        --model ttm \
        --dataset aleppo_2017 \
        --context-length 512 --forecast-length 96 \
        --tolerance 1e-4

    # All three fine-tuned, 100 episodes each
    python scripts/experiments/validate_predict_batch.py \
        --model ttm --dataset aleppo_2017 --max-episodes 100 \
        --checkpoint trained_models/artifacts/ttm/2026-02-27_03:53_RID20260227_035316_193673_holdout_workflow/model.pt \
        --context-length 512 --forecast-length 96
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.utils import get_patient_column
from src.evaluation.episode_builders import build_midnight_episodes
from src.evaluation.metrics import compute_regression_metrics
from src.models import create_model_and_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SAMPLING_INTERVAL_MINUTES = 5


# ---------------------------------------------------------------------------
# Episode building (mirrors Phase 1 of evaluate_nocturnal_forecasting)
# ---------------------------------------------------------------------------

def build_episode_panel(
    holdout_data: pd.DataFrame,
    context_length: int,
    forecast_length: int,
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
    max_episodes: Optional[int] = None,
) -> tuple[list, pd.DataFrame]:
    """Build midnight episodes and return metadata + concatenated panel DataFrame.

    The panel has an ``episode_id`` column (``"<patient_id>::ep<NNN>"``) that
    ``predict_batch()`` uses to group per-episode slices.

    Returns
    -------
    episode_metadata: list of dicts with keys episode_id, patient_id, anchor, target_bg
    panel_df:         pd.DataFrame with all context windows concatenated
    """
    patient_col = get_patient_column(holdout_data)
    patients = holdout_data[patient_col].unique()

    episode_metadata = []
    context_dfs = []

    for patient_id in patients:
        patient_df = holdout_data[holdout_data[patient_col] == patient_id].copy()

        if not isinstance(patient_df.index, pd.DatetimeIndex):
            time_col = "datetime" if "datetime" in patient_df.columns else None
            if time_col:
                patient_df[time_col] = pd.to_datetime(patient_df[time_col])
                patient_df = patient_df.set_index(time_col).sort_index()
            else:
                logger.warning("Patient %s: no datetime column, skipping", patient_id)
                continue

        episodes, _ = build_midnight_episodes(
            patient_df,
            context_length=context_length,
            forecast_length=forecast_length,
            target_col=target_col,
            covariate_cols=covariate_cols,
            interval_mins=SAMPLING_INTERVAL_MINUTES,
        )

        if not episodes:
            continue

        for i, ep in enumerate(episodes):
            if len(ep["target_bg"]) < forecast_length:
                continue
            ep_id = f"{patient_id}::ep{i:03d}"
            ctx = ep["context_df"].copy().reset_index(names="datetime")
            ctx["p_num"] = patient_id
            ctx["episode_id"] = ep_id
            context_dfs.append(ctx)
            episode_metadata.append(
                {
                    "episode_id": ep_id,
                    "patient_id": str(patient_id),
                    "anchor": ep["anchor"],
                    "target_bg": ep["target_bg"],
                }
            )

            if max_episodes is not None and len(episode_metadata) >= max_episodes:
                logger.info("Reached --max-episodes %d, stopping episode build.", max_episodes)
                panel_df = pd.concat(context_dfs, ignore_index=True)
                return episode_metadata, panel_df

    if not context_dfs:
        return [], pd.DataFrame()

    panel_df = pd.concat(context_dfs, ignore_index=True)
    return episode_metadata, panel_df


# ---------------------------------------------------------------------------
# Run A: sequential predict()
# ---------------------------------------------------------------------------

def run_sequential(model, episode_metadata: list, panel_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Call model.predict(ctx_df) once per episode — the current production path."""
    results: Dict[str, np.ndarray] = {}
    ep_ids = [m["episode_id"] for m in episode_metadata]

    for ep_id in ep_ids:
        ctx_df = panel_df[panel_df["episode_id"] == ep_id].copy()
        pred = np.asarray(model.predict(ctx_df))
        if pred.ndim == 3:
            pred = pred[0, :, 0]
        elif pred.ndim == 2:
            pred = pred.flatten()
        results[ep_id] = pred

    return results


# ---------------------------------------------------------------------------
# Run B: predict_batch()
# ---------------------------------------------------------------------------

def run_batch(model, panel_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Call model.predict_batch(panel_df) — exercises the new predict_batch path."""
    raw = model.predict_batch(panel_df, episode_col="episode_id")
    # Normalise shape: same as run_sequential
    normalised: Dict[str, np.ndarray] = {}
    for ep_id, pred in raw.items():
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0, :, 0]
        elif pred.ndim == 2:
            pred = pred.flatten()
        normalised[ep_id] = pred
    return normalised


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_results(
    seq_preds: Dict[str, np.ndarray],
    batch_preds: Dict[str, np.ndarray],
    tolerance: float,
) -> bool:
    """Compare sequential vs batch predictions.

    Returns True if all episodes pass within tolerance, False otherwise.
    Logs a summary table and per-episode failures.
    """
    ep_ids = sorted(seq_preds.keys())

    missing_in_batch = set(ep_ids) - set(batch_preds.keys())
    extra_in_batch = set(batch_preds.keys()) - set(ep_ids)
    if missing_in_batch:
        logger.error("FAIL — predict_batch() dropped %d episode(s): %s",
                     len(missing_in_batch), sorted(missing_in_batch)[:5])
    if extra_in_batch:
        logger.warning("predict_batch() returned %d unexpected episode(s): %s",
                       len(extra_in_batch), sorted(extra_in_batch)[:5])

    max_diffs: list[float] = []
    failures: list[str] = []

    for ep_id in ep_ids:
        if ep_id not in batch_preds:
            max_diffs.append(float("nan"))
            failures.append(ep_id)
            continue

        a = seq_preds[ep_id]
        b = batch_preds[ep_id]

        # Truncate to the shorter of the two (model may return different lengths)
        min_len = min(len(a), len(b))
        if len(a) != len(b):
            logger.debug(
                "Episode %s: length mismatch seq=%d batch=%d — comparing first %d steps",
                ep_id, len(a), len(b), min_len,
            )
        a, b = a[:min_len], b[:min_len]

        max_diff = float(np.max(np.abs(a - b)))
        max_diffs.append(max_diff)
        if max_diff > tolerance:
            failures.append(ep_id)

    passed = len(ep_ids) - len(missing_in_batch) - len([f for f in failures if f not in missing_in_batch])
    failed = len(failures)

    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info("Episodes compared : %d", len(ep_ids))
    logger.info("Tolerance         : %.2e", tolerance)
    logger.info("Max diff (overall): %.6e", max(max_diffs, default=float("nan")))
    logger.info("Mean max diff     : %.6e", float(np.nanmean(max_diffs)) if max_diffs else float("nan"))
    logger.info("Passed            : %d", len(ep_ids) - failed)
    logger.info("Failed            : %d", failed)

    if failures:
        logger.error("")
        logger.error("FAILED EPISODES (max_diff > %.2e):", tolerance)
        for ep_id in failures[:20]:
            if ep_id in batch_preds and ep_id in seq_preds:
                a = seq_preds[ep_id]
                b = batch_preds[ep_id]
                min_len = min(len(a), len(b))
                diff = float(np.max(np.abs(a[:min_len] - b[:min_len])))
                logger.error("  %s  max_diff=%.6e", ep_id, diff)
            else:
                logger.error("  %s  MISSING in batch output", ep_id)
        if len(failures) > 20:
            logger.error("  ... and %d more", len(failures) - 20)

    # Also compare overall RMSE (informational)
    _compare_rmse(seq_preds, batch_preds, ep_ids)

    return failed == 0


def _compare_rmse(seq_preds, batch_preds, ep_ids):
    """Log the RMSE from both runs for sanity-check against the prior eval baseline."""
    # We don't have ground truth here — just compare that both runs agree numerically.
    # Compute a pseudo-metric: std of predictions across episodes.
    all_seq = np.concatenate([seq_preds[ep] for ep in ep_ids if ep in seq_preds])
    all_bat = np.concatenate([batch_preds[ep] for ep in ep_ids if ep in batch_preds])
    logger.info("")
    logger.info("Prediction stats (seq  ): mean=%.4f  std=%.4f  n=%d",
                float(np.mean(all_seq)), float(np.std(all_seq)), len(all_seq))
    logger.info("Prediction stats (batch): mean=%.4f  std=%.4f  n=%d",
                float(np.mean(all_bat)), float(np.std(all_bat)), len(all_bat))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate predict_batch() == predict()-per-episode on midnight episodes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ttm",
        choices=["sundial", "ttm", "chronos2", "moirai", "timegrad", "timesfm", "tide"],
    )
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config YAML file")
    parser.add_argument("--dataset", type=str, default="aleppo_2017")
    parser.add_argument("--config-dir", type=str, default="configs/data/holdout_10pct")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned checkpoint (omit for zero-shot)")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--forecast-length", type=int, default=96)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=50,
        help="Number of midnight episodes to process (default: 50 for speed)."
             " Set to 0 to run all episodes.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Max absolute difference per prediction step to consider a pass "
             "(default: 1e-6; use ~1e-4 for zero-shot TTM batched pipeline).",
    )
    parser.add_argument("--cuda-device", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    logger.info("=" * 60)
    logger.info("PREDICT_BATCH VALIDATION")
    logger.info("=" * 60)
    logger.info("Model      : %s", args.model)
    logger.info("Mode       : %s", "fine-tuned" if args.checkpoint else "zero-shot")
    logger.info("Dataset    : %s", args.dataset)
    logger.info("Checkpoint : %s", args.checkpoint or "(none)")
    logger.info("Max eps    : %s", args.max_episodes if args.max_episodes else "all")
    logger.info("Tolerance  : %.2e", args.tolerance)

    # --- Load model ---
    from src.utils import load_yaml_config
    config_dict = load_yaml_config(args.model_config) if args.model_config else {}
    model_kwargs = {
        **config_dict,
        "context_length": args.context_length,
        "forecast_length": args.forecast_length,
    }
    model, config = create_model_and_config(
        args.model, checkpoint=args.checkpoint, **model_kwargs
    )
    logger.info("Model loaded. is_fitted=%s", model.is_fitted)

    # --- Load holdout data ---
    registry = DatasetRegistry(holdout_config_dir=args.config_dir)
    holdout_data = registry.load_holdout_data_only(args.dataset)
    logger.info("Holdout rows: %d", len(holdout_data))

    # --- Build episodes ---
    max_eps = args.max_episodes if args.max_episodes > 0 else None
    logger.info("\n--- Building midnight episodes (max=%s) ---", max_eps or "all")
    episode_metadata, panel_df = build_episode_panel(
        holdout_data,
        context_length=config.context_length,
        forecast_length=config.forecast_length,
        max_episodes=max_eps,
    )

    if not episode_metadata:
        logger.error("No valid episodes found — check dataset and context/forecast lengths.")
        sys.exit(2)

    n_patients = len({m["patient_id"] for m in episode_metadata})
    logger.info("Built %d episodes across %d patients", len(episode_metadata), n_patients)
    logger.info("Panel shape: %s", panel_df.shape)

    # --- Run A: sequential predict() ---
    logger.info("\n--- Run A: sequential model.predict() per episode ---")
    seq_preds = run_sequential(model, episode_metadata, panel_df)
    logger.info("Sequential done. %d predictions.", len(seq_preds))

    # --- Run B: predict_batch() ---
    logger.info("\n--- Run B: model.predict_batch(panel_df) ---")
    batch_preds = run_batch(model, panel_df)
    logger.info("Batch done. %d predictions.", len(batch_preds))

    # --- Compare ---
    logger.info("\n--- Comparing results ---")
    passed = compare_results(seq_preds, batch_preds, args.tolerance)

    if passed:
        logger.info("")
        logger.info("PASS — predict_batch() matches predict()-per-episode within %.2e", args.tolerance)
        sys.exit(0)
    else:
        logger.error("")
        logger.error("FAIL — predict_batch() diverges from predict()-per-episode")
        sys.exit(1)


if __name__ == "__main__":
    main()
