# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Ensemble the top-K Chronos-2 trials and evaluate / write a submission.

"Winning uses the most models": this averages the point forecasts of the best K
kept predictors from an Optuna study and reports the ensemble's RMSE/MARD/DTS at
+30/+60/+90/+120 vs the single best model, on the internal temporal validation
split. It also exposes :func:`write_submission_parquet` to emit the challenge's
``pred_30/60/90/120`` columns for a template.

Requires the study to have been run with ``keep_models: true`` so each trial's
predictor still exists under ``<models_root>/trial_XXXXX``.

Example::

    python -m src.workflows.sweeps.optuna_chronos2.ensemble_submit \\
        --storage sqlite:///results/optuna/chronos2_metabonet_18h.db \\
        --study-name chronos2_metabonet_18h \\
        --models-root results/optuna/chronos2_metabonet_18h_models \\
        --config path/to/study_config.yaml \\
        --top-k 10 --patient-limit 60 --out results/optuna/ensemble_top10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import optuna
import pandas as pd

from .data import MetabonetData
from .metrics import HORIZON_MINUTES, HORIZON_STEPS, evaluate_horizons
from .study import load_config

logger = logging.getLogger(__name__)


def _predict_matrix(
    predictor, context, known_future, item_ids: List[str]
) -> np.ndarray:
    """Return an (n_windows, 24) mean-forecast matrix ordered by ``item_ids``."""
    horizon = max(HORIZON_STEPS.values())
    forecast = predictor.predict(context, known_covariates=known_future)
    pred_df = forecast["mean"].reset_index()
    rows: List[np.ndarray] = []
    for item_id in item_ids:
        sl = pred_df.loc[pred_df["item_id"] == item_id].sort_values("timestamp")
        values = sl["mean"].to_numpy()[:horizon]
        if values.size < horizon:
            values = np.pad(values, (0, horizon - values.size), constant_values=np.nan)
        rows.append(values)
    return np.vstack(rows)


def select_top_trials(
    study: optuna.Study, top_k: int
) -> List[optuna.trial.FrozenTrial]:
    """Return the top-K completed trials sorted by objective (RMSE@120)."""
    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    complete.sort(key=lambda t: t.value)
    return complete[:top_k]


def load_predictor(models_root: Path, trial_number: int):
    """Load a saved AutoGluon predictor for a trial, or None if missing."""
    from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
        TimeSeriesPredictor,
    )

    path = models_root / f"trial_{trial_number:05d}"
    if not path.exists():
        logger.warning("Predictor for trial %d not found at %s", trial_number, path)
        return None
    return TimeSeriesPredictor.load(str(path))


def ensemble_forecasts(
    data: MetabonetData,
    trials: Sequence[optuna.trial.FrozenTrial],
    models_root: Path,
) -> Dict[str, object]:
    """Average point forecasts across trials on the internal val split.

    Returns a dict with the ensemble prediction matrix, the ground-truth matrix,
    per-model matrices, and the item ids. Anchors/targets are identical across
    models (they depend only on data settings), so matrices align row-wise.
    """
    per_model: List[np.ndarray] = []
    used: List[int] = []
    true_matrix: Optional[np.ndarray] = None
    item_ids: Optional[List[str]] = None

    for trial in trials:
        predictor = load_predictor(models_root, trial.number)
        if predictor is None:
            continue
        past_cols = list(trial.user_attrs.get("past_cols", []))
        known_cols = list(trial.user_attrs.get("known_cols", []))
        static_cols = list(trial.user_attrs.get("static_cols", []))
        val = data.build_val_windows(
            past_cols, known_cols=known_cols, static_cols=static_cols
        )
        matrix = _predict_matrix(predictor, val.context, val.known_future, val.item_ids)
        per_model.append(matrix)
        used.append(trial.number)
        true_matrix = val.true_matrix
        item_ids = val.item_ids

    if not per_model:
        raise ValueError(
            "No usable predictors found. Was the study run with keep_models: true?"
        )

    ensemble = np.nanmean(np.stack(per_model, axis=0), axis=0)
    return {
        "ensemble": ensemble,
        "per_model": per_model,
        "true_matrix": true_matrix,
        "item_ids": item_ids,
        "trial_numbers": used,
    }


def write_submission_parquet(
    template: pd.DataFrame,
    pred_matrix: np.ndarray,
    anchor_index: Sequence[int],
    out_path: str | Path,
) -> Path:
    """Write challenge-format predictions (``pred_30/60/90/120``) to parquet.

    Args:
        template: Challenge template rows (columns ``id``, ``source_file``,
            ``date``) in submission order; extra columns are preserved.
        pred_matrix: (n_anchors, 24) mean forecasts (mg/dL) for a subset of
            template rows identified by ``anchor_index``.
        anchor_index: Row positions in ``template`` that ``pred_matrix`` fills;
            all other rows are left NaN (a valid partial-horizon submission
            requires each populated column to be fully filled — see toolkit).
        out_path: Output parquet path.
    """
    out = template.copy()
    for minutes in HORIZON_MINUTES:
        out[f"pred_{minutes}"] = np.nan
    for row_pos, anchor in enumerate(anchor_index):
        for minutes, step in HORIZON_STEPS.items():
            out.iloc[anchor, out.columns.get_loc(f"pred_{minutes}")] = pred_matrix[
                row_pos, step - 1
            ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Study YAML (to rebuild the internal validation data).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--patient-limit",
        type=int,
        default=None,
        help="Override patient_limit for faster ensemble evaluation.",
    )
    parser.add_argument("--out", default=None, help="Optional CSV summary path stem.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.patient_limit is not None:
        config["data"]["patient_limit"] = args.patient_limit
    data = MetabonetData(config["processed_path"], **config["data"]).load()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    trials = select_top_trials(study, args.top_k)
    if not trials:
        print("No completed trials to ensemble.")
        return 1

    result = ensemble_forecasts(data, trials, Path(args.models_root))
    true_matrix = result["true_matrix"]
    trial_numbers = list(result["trial_numbers"])  # type: ignore[call-overload]
    ensemble_metrics = evaluate_horizons(result["ensemble"], true_matrix)  # type: ignore[arg-type]
    best_metrics = evaluate_horizons(result["per_model"][0], true_matrix)  # type: ignore[index,arg-type]

    print("=" * 64)
    print(f"  Ensemble of {len(trial_numbers)} models (trials {trial_numbers})")
    print("=" * 64)
    print(f"{'horizon':>8} {'best_single':>12} {'ensemble':>10} {'delta':>8}")
    for minutes in HORIZON_MINUTES:
        b = best_metrics[f"rmse_{minutes}"]
        e = ensemble_metrics[f"rmse_{minutes}"]
        print(f"{minutes:>6}m  {b:>12.2f} {e:>10.2f} {e - b:>8.2f}")
    print()
    print(
        f"Ensemble MARD@120: {ensemble_metrics['mard_120']:.2f}%  "
        f"DTS A+B@120: {ensemble_metrics['dts_a_pct_120'] + ensemble_metrics['dts_b_pct_120']:.1f}%"
    )

    if args.out:
        summary = pd.DataFrame(
            [
                {"model": "best_single", **best_metrics},
                {"model": "ensemble", **ensemble_metrics},
            ]
        )
        out = Path(args.out).with_suffix(".csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
