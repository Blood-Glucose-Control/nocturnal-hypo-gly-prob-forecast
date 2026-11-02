# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna objective for the GBM (gradient-boosted tree) sweep.

Reuses the Chronos-2 sweep's :class:`MetabonetData` (panels/val windows and
known-future covariate assembly) and challenge metrics
(:func:`evaluate_horizons`), and trains an AutoGluon tabular reduction
(``RecursiveTabular`` / ``DirectTabular``) with a LightGBM/CatBoost/XGBoost
backend per trial. Runs on CPU so it can execute concurrently with GPU-bound
sweeps.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna

from ..optuna_chronos2.data import MetabonetData
from ..optuna_chronos2.metrics import HORIZON_STEPS, evaluate_horizons
from .search_space import GBMSearchSpaceBounds, suggest_gbm_point

PRIMARY_METRIC = "mard_avg"

# Objective value assigned to a trial whose fit fails. A large finite penalty
# makes TPE steer away from bad regions instead of ignoring hard failures.
FAILURE_PENALTY = 100.0


class GBMObjective:
    """Callable Optuna objective that fits + scores one GBM config per trial."""

    def __init__(
        self,
        data: MetabonetData,
        output_root: str | Path,
        *,
        bounds: Optional[GBMSearchSpaceBounds] = None,
        time_limit: Optional[int] = None,
        keep_models: bool = False,
        gpu_label: str = "",
        num_cpus: Optional[int] = None,
        primary_metric: str = PRIMARY_METRIC,
        objective_metrics: Optional[Sequence[str]] = None,
    ) -> None:
        self.data = data
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.bounds = bounds or GBMSearchSpaceBounds()
        self.time_limit = time_limit
        self.keep_models = keep_models
        self.gpu_label = gpu_label
        self.num_cpus = num_cpus
        self.primary_metric = primary_metric
        self.objective_metrics = list(objective_metrics) if objective_metrics else None
        self.is_multi = bool(self.objective_metrics and len(self.objective_metrics) > 1)

    def _predict_matrix(self, forecast, item_ids: List[str]) -> np.ndarray:  # noqa: ANN001
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

    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
            TimeSeriesPredictor,
        )

        point = suggest_gbm_point(trial, self.bounds)

        past_cols: List[str] = []
        known_cols = point.known_cols
        static_cols = point.static_cols
        trial.set_user_attr("strategy", point.strategy)
        trial.set_user_attr("backend", point.backend)
        trial.set_user_attr("known_cols", known_cols)
        trial.set_user_attr("static_cols", static_cols)
        trial.set_user_attr("eval_metric", point.eval_metric)
        trial.set_user_attr("worker", self.gpu_label)

        trial_dir = self.output_root / f"trial_{trial.number:05d}"

        train_tsdf = self.data.build_train_panel(
            past_cols, known_cols=known_cols, static_cols=static_cols
        )
        val = self.data.build_val_windows(
            past_cols, known_cols=known_cols, static_cols=static_cols
        )

        predictor_kwargs: Dict[str, Any] = dict(
            prediction_length=point.forecast_length,
            target="target",
            freq=f"{point.interval_mins}min",
            eval_metric=point.eval_metric,
            path=str(trial_dir),
        )
        if known_cols:
            predictor_kwargs["known_covariates_names"] = list(known_cols)

        model_hyperparameters = dict(point.hyperparameters)
        if self.num_cpus is not None:
            # AutoGluon TimeSeries controls per-model resources via ag_args_fit
            # (not fit()-level kwargs). Pin CPUs so the CPU-only tree backends
            # don't starve a co-running GPU sweep, and force num_gpus=0.
            model_hyperparameters["ag_args_fit"] = {
                "num_cpus": self.num_cpus,
                "num_gpus": 0,
            }

        predictor = TimeSeriesPredictor(**predictor_kwargs)
        fit_kwargs: Dict[str, Any] = {
            "train_data": train_tsdf,
            "hyperparameters": {point.model_key: model_hyperparameters},
            "enable_ensemble": False,
        }
        if self.time_limit is not None:
            fit_kwargs["time_limit"] = self.time_limit

        fit_failed = False
        fail_reason = ""
        fit_seconds = 0.0
        metrics: Dict[str, float] = {}
        try:
            fit_start = time.time()
            predictor.fit(**fit_kwargs)
            fit_seconds = time.time() - fit_start
            forecast = predictor.predict(val.context, known_covariates=val.known_future)
            pred_matrix = self._predict_matrix(forecast, val.item_ids)
            metrics = evaluate_horizons(pred_matrix, val.true_matrix)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully; penalize trial
            fit_failed = True
            fail_reason = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            if not self.keep_models and trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)

        if fit_failed:
            trial.set_user_attr("fit_failed", True)
            trial.set_user_attr("fail_reason", fail_reason)
            penalty = float(FAILURE_PENALTY)
            if self.is_multi:
                assert self.objective_metrics is not None
                return tuple(penalty for _ in self.objective_metrics)
            return penalty

        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("n_val_windows", len(val.item_ids))
        trial.set_user_attr("fit_seconds", round(fit_seconds, 1))
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
