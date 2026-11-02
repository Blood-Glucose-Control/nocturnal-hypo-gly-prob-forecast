# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna objective: fine-tune Chronos-2 (LoRA) and score on internal validation.

One trial:

1. Sample a :class:`SearchPoint` (LoRA + optimizer + covariate selection).
2. Build a :class:`Chronos2Config` and its AutoGluon hyperparameters.
3. Fit an AutoGluon ``TimeSeriesPredictor`` on the MetaboNet train panel.
4. Forecast the challenge-shaped validation windows (context <= 288 -> 24 steps).
5. Compute RMSE/MARD/DTS at +30/+60/+90/+120 minutes (mg/dL).
6. Record every metric as a trial user attribute and return the primary
   objective (RMSE at +120 min).

The predictor is driven directly (rather than through ``Chronos2Forecaster``) so
the validation loop matches the challenge's fixed-horizon evaluation exactly,
while still reusing ``Chronos2Config`` for hyperparameter assembly.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna

from .data import MetabonetData
from .metrics import HORIZON_STEPS, evaluate_horizons
from .pruning import (
    install_pruning_patch,
    make_early_stopping_callback,
    make_pruning_callback,
    make_step_recorder_callback,
    pruning_context,
)
from .search_space import SearchPoint, SearchSpaceBounds, suggest_search_point

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "rmse_120"


def _read_completed_steps(trial_dir: Path) -> Optional[int]:
    """Best-effort read of the max HF Trainer global_step under a trial dir.

    AutoGluon/HF write ``trainer_state.json`` alongside checkpoints; we scan for
    any and return the largest ``global_step``. Returns None if not found (the
    fit-time + hit_time_limit telemetry still indicates truncation).
    """
    try:
        best = None
        for state_path in trial_dir.rglob("trainer_state.json"):
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            step = data.get("global_step")
            if isinstance(step, int) and (best is None or step > best):
                best = step
        return best
    except Exception:
        return None


class Chronos2Objective:
    """Callable Optuna objective bound to a preloaded MetaboNet dataset."""

    def __init__(
        self,
        data: MetabonetData,
        output_root: str | Path,
        *,
        bounds: Optional[SearchSpaceBounds] = None,
        time_limit: Optional[int] = None,
        keep_models: bool = False,
        gpu_label: str = "",
        primary_metric: str = PRIMARY_METRIC,
        objective_metrics: Optional[Sequence[str]] = None,
        enable_pruning: bool = False,
        report_intermediate: bool = True,
        early_stopping_patience: Optional[int] = None,
    ) -> None:
        self.data = data
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.bounds = bounds or SearchSpaceBounds()
        self.time_limit = time_limit
        self.keep_models = keep_models
        self.gpu_label = gpu_label
        # Single-objective metric Optuna minimizes, e.g. "rmse_120" or
        # "mard_120". All horizon metrics are still recorded per trial.
        self.primary_metric = primary_metric
        # Multi-objective: a list of >=2 logged metric names returned as a tuple
        # (directions are configured on the study). None -> single objective.
        self.objective_metrics = list(objective_metrics) if objective_metrics else None
        self.is_multi = bool(self.objective_metrics and len(self.objective_metrics) > 1)
        # Report the HF validation loss to Optuna during fine-tuning and prune
        # weak trials. Disabled for multi-objective studies (Optuna cannot prune
        # them) and if the guarded patch can't install.
        self.enable_pruning = (
            enable_pruning and not self.is_multi and install_pruning_patch()
        )
        # Log intermediate values (training curves) even when not pruning.
        # NOTE: Optuna does NOT support trial.report() for MULTI-objective
        # studies (it raises NotImplementedError), so intermediate values are
        # only available for single-objective runs. Piggybacks on evals already
        # run for best-checkpoint selection, so it adds no extra eval cost.
        self.report_intermediate = report_intermediate
        self._can_report = self.enable_pruning or (
            report_intermediate and not self.is_multi and install_pruning_patch()
        )
        # HF early stopping: stop fine-tuning once eval_loss plateaus for this
        # many evals (independent of Optuna pruning; works for multi-objective).
        self.early_stopping_patience = (
            int(early_stopping_patience)
            if early_stopping_patience and int(early_stopping_patience) > 0
            else None
        )

    # ------------------------------------------------------------------
    def _build_config(self, point: SearchPoint):
        from src.models.chronos2.config import Chronos2Config

        return Chronos2Config(**point.config_kwargs)

    def _predict_matrix(self, forecast, item_ids: List[str]) -> np.ndarray:
        """Extract an (n_windows, 24) mean-forecast matrix ordered by item_ids."""
        horizon = max(HORIZON_STEPS.values())
        pred_df = forecast["mean"].reset_index()
        rows: List[np.ndarray] = []
        for item_id in item_ids:
            item_slice = pred_df.loc[pred_df["item_id"] == item_id]
            item_slice = item_slice.sort_values("timestamp")
            values = item_slice["mean"].to_numpy()[:horizon]
            if values.size < horizon:
                values = np.pad(
                    values, (0, horizon - values.size), constant_values=np.nan
                )
            rows.append(values)
        return np.vstack(rows)

    # ------------------------------------------------------------------
    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
            TimeSeriesPredictor,
        )

        point = suggest_search_point(trial, self.bounds)
        config = self._build_config(point)

        past_cols = self.data.resolve_past_covariates(point.past_cols)
        known_cols = point.known_cols
        static_cols = self.data.couple_availability_flags(past_cols, point.static_cols)
        trial.set_user_attr("past_cols", past_cols)
        trial.set_user_attr("known_cols", known_cols)
        trial.set_user_attr("static_cols", static_cols)
        trial.set_user_attr("gpu", self.gpu_label)

        trial_dir = self.output_root / f"trial_{trial.number:05d}"

        train_tsdf = self.data.build_train_panel(
            past_cols, known_cols=known_cols, static_cols=static_cols
        )
        val = self.data.build_val_windows(
            past_cols, known_cols=known_cols, static_cols=static_cols
        )

        predictor_kwargs: Dict[str, Any] = dict(
            prediction_length=config.forecast_length,
            target="target",
            freq=f"{config.interval_mins}min",
            eval_metric=config.eval_metric,
            path=str(trial_dir),
        )
        if known_cols:
            predictor_kwargs["known_covariates_names"] = list(known_cols)

        predictor = TimeSeriesPredictor(**predictor_kwargs)
        fit_kwargs = {
            "train_data": train_tsdf,
            "hyperparameters": config.get_autogluon_hyperparameters(),
            "enable_ensemble": config.enable_ensemble,
        }
        if self.time_limit is not None:
            fit_kwargs["time_limit"] = self.time_limit

        # Attach a callback that logs Optuna intermediate values (training
        # curves) and, when pruning is enabled, stops weak trials early. Only
        # when eval runs during fine-tuning (otherwise there is nothing to
        # report and forcing eval would add real cost).
        # Build the extra Trainer callbacks. Only meaningful when eval runs
        # during fine-tuning (early stopping / reporting both need evals).
        eval_on = config.eval_during_fine_tune
        pruning_callback = (
            make_pruning_callback(trial, prune=self.enable_pruning)
            if (self._can_report and eval_on)
            else None
        )
        step_holder = None
        callbacks: list = []
        if eval_on:
            if pruning_callback is not None:
                callbacks.append(pruning_callback)
            if self.early_stopping_patience:
                callbacks.append(
                    make_early_stopping_callback(self.early_stopping_patience)
                )
            recorder, step_holder = make_step_recorder_callback()
            callbacks.append(recorder)
        # The callbacks are injected via the guarded pipeline.fit patch.
        if callbacks and not install_pruning_patch():
            callbacks = []  # patch unavailable -> run without extra callbacks
        try:
            fit_start = time.time()
            if callbacks:
                with pruning_context(callbacks):
                    predictor.fit(**fit_kwargs)
            else:
                predictor.fit(**fit_kwargs)
            fit_seconds = time.time() - fit_start
            # Prefer the recorder's actual final step; fall back to the surviving
            # checkpoint (which under-reports when save_total_limit prunes).
            completed_steps = (
                step_holder["steps"] if step_holder and step_holder["steps"] else None
            ) or _read_completed_steps(trial_dir)
            if pruning_callback is not None and pruning_callback.pruned:
                # Bad trial stopped early during fine-tuning; skip the (costly)
                # validation forecast and let Optuna record it as pruned.
                raise optuna.TrialPruned(
                    f"Pruned at step ~{completed_steps} "
                    f"(val loss {pruning_callback.last_value})."
                )
            forecast = predictor.predict(val.context, known_covariates=val.known_future)
            pred_matrix = self._predict_matrix(forecast, val.item_ids)
            metrics = evaluate_horizons(pred_matrix, val.true_matrix)
        finally:
            if not self.keep_models and trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)

        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("n_val_windows", len(val.item_ids))
        # Training-progress telemetry: how long fine-tuning ran, whether the
        # per-trial time_limit truncated it, and the actual completed steps
        # (the requested fine_tune_steps is an upper bound the deep configs
        # rarely reach before time_limit cuts them off).
        trial.set_user_attr("fit_seconds", round(fit_seconds, 1))
        trial.set_user_attr("requested_steps", config.fine_tune_steps)
        if completed_steps is not None:
            # Actual steps trained (from the recorder), which may exceed the
            # best-validation checkpoint that becomes the final model.
            trial.set_user_attr("completed_steps", completed_steps)
        best_ckpt_step = _read_completed_steps(trial_dir)
        if best_ckpt_step is not None:
            trial.set_user_attr("best_checkpoint_step", best_ckpt_step)
        if self.early_stopping_patience:
            # Early-stopped iff training halted before the requested budget.
            trial.set_user_attr(
                "early_stopped",
                bool(completed_steps and completed_steps < config.fine_tune_steps),
            )
        if self.time_limit is not None:
            trial.set_user_attr("hit_time_limit", fit_seconds >= 0.9 * self.time_limit)

        if self.is_multi:
            assert self.objective_metrics is not None
            values = [metrics.get(m, float("nan")) for m in self.objective_metrics]
            if not all(np.isfinite(v) for v in values):
                raise optuna.TrialPruned(
                    f"A multi-objective metric is not finite: "
                    f"{dict(zip(self.objective_metrics, values))}."
                )
            return tuple(float(v) for v in values)

        primary = metrics.get(self.primary_metric, float("nan"))
        if not np.isfinite(primary):
            raise optuna.TrialPruned(
                f"Primary metric {self.primary_metric} is not finite."
            )
        return float(primary)
