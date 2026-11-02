# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna objective for the Moirai-1.x (native fine-tune) sweep.

Each trial natively fine-tunes a Moirai-1.x checkpoint (uni2ts ``MoiraiFinetune``,
driven through the shared ``src/models/moirai`` wrapper) on IOBP2 training-region
windows, then scores an internal temporal validation split. Insulin is fed as a
KNOWN-FUTURE covariate (``feat_dynamic_real`` spanning context + horizon), with
optional past-only insulin-kinetics channels.

Pipeline per trial:

1. Build an IOBP2 train-region DataFrame (``patient_id`` / ``datetime`` /
   ``bg_mM`` = CGM/18 mmol/L + selected covariate columns) from the shared
   :class:`~..optuna_chronos2.data.MetabonetData` engineered per-patient frames,
   restricted to each patient's training region (timestamps <= ``split_ts``).
2. Fit a :class:`~src.models.moirai.model.MoiraiForecaster` (the wrapper slides
   non-overlapping context+horizon windows and routes known-future covariates as
   ``feat_dynamic_real``).
3. Build temporal-val windows whose horizon anchors in the held-out region, as a
   GluonTS ``ListDataset`` (context CGM + past/known covariates), predict the
   mean forecast, convert mmol/L -> mg/dL, and score with
   :func:`~..optuna_chronos2.metrics.evaluate_horizons` -> ``mard_avg``.
4. Free GPU memory before the next trial.

Fit failures degrade gracefully to a large finite ``FAILURE_PENALTY`` so TPE
steers away from bad regions instead of aborting the worker.
"""

from __future__ import annotations

import gc
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
import pandas as pd

from ..optuna_chronos2.data import MetabonetData
from ..optuna_chronos2.features import TARGET_COL
from ..optuna_chronos2.metrics import HORIZON_STEPS, evaluate_horizons
from .search_space import (
    MoiraiSearchPoint,
    MoiraiSearchSpaceBounds,
    suggest_moirai_point,
)

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "mard_avg"

# Objective value assigned to a trial whose fit fails. A large finite penalty
# makes TPE steer away from bad regions instead of ignoring hard failures.
FAILURE_PENALTY = 100.0

# mg/dL per mmol/L (CGM unit conversion; the Moirai target is mmol/L).
MGDL_PER_MMOL = 18.0

# Full clinical horizon length (steps): 24 -> +120 min at 5-minute sampling.
_HORIZON_LEN = max(HORIZON_STEPS.values())


# Per-patient cache entry: (patient_id, engineered frame, train/val split index,
# interpolated CGM target in mg/dL over the whole series).
_PatientCache = Tuple[str, pd.DataFrame, int, np.ndarray]


class MoiraiObjective:
    """Optuna objective: fine-tune Moirai-1.x and score ``mard_avg`` on val."""

    def __init__(
        self,
        data: MetabonetData,
        output_root: str,
        *,
        bounds: Optional[MoiraiSearchSpaceBounds] = None,
        time_limit: Optional[int] = None,
        keep_models: bool = False,
        gpu_label: str = "",
        primary_metric: str = PRIMARY_METRIC,
        objective_metrics: Optional[Sequence[str]] = None,
        report_intermediate: bool = False,
        early_stopping_patience: Optional[int] = None,
        train_tail_days: Optional[int] = None,
        val_windows_per_patient: int = 4,
        max_val_windows_total: int = 2000,
        val_stride_steps: int = 6,
        seed: int = 42,
        **_ignored: Any,
    ) -> None:
        self.data = data
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.bounds = bounds or MoiraiSearchSpaceBounds()
        # time_limit is accepted for interface parity but unused: Moirai
        # fine-tuning is epoch-bounded (num_epochs), not wall-clock-bounded.
        self.time_limit = time_limit
        self.keep_models = keep_models
        self.gpu_label = gpu_label
        self.primary_metric = primary_metric
        self.objective_metrics = list(objective_metrics) if objective_metrics else None
        self.is_multi = bool(self.objective_metrics and len(self.objective_metrics) > 1)
        self.report_intermediate = report_intermediate
        self.early_stopping_patience = early_stopping_patience
        # Use only the last ``train_tail_days`` of each patient's training region
        # (None -> full history). Bounds per-trial fit time so a 4-hour run
        # explores more configs; the winning config is retrained on full data.
        self.train_tail_days = train_tail_days
        self.val_windows_per_patient = int(val_windows_per_patient)
        self.max_val_windows_total = int(max_val_windows_total)
        self.val_stride_steps = int(val_stride_steps)
        self.seed = int(seed)
        self.forecast_length = self.bounds.forecast_length
        self.interval_mins = self.bounds.interval_mins
        self._patient_cache: Optional[List[_PatientCache]] = None

    # ------------------------------------------------------------------
    # Imports (kept lazy so ``search_space`` stays importable without torch)
    # ------------------------------------------------------------------
    @staticmethod
    def _import_wrapper():
        from src.models.moirai.config import (  # pyright: ignore[reportMissingImports]
            MoiraiConfig,
        )
        from src.models.moirai.model import (  # pyright: ignore[reportMissingImports]
            MoiraiForecaster,
        )

        return MoiraiConfig, MoiraiForecaster

    # ------------------------------------------------------------------
    # Per-patient cache (target + split index), built once, reused per trial
    # ------------------------------------------------------------------
    def _ensure_patient_cache(self) -> List[_PatientCache]:
        if self._patient_cache is not None:
            return self._patient_cache
        cache: List[_PatientCache] = []
        for patient in self.data.patients:
            frame = patient.frame
            target = frame[TARGET_COL].where(frame[TARGET_COL] > 0)
            target = target.interpolate(limit_area="inside")
            tvals = target.to_numpy(dtype=np.float32)
            split_idx = int(frame.index.searchsorted(patient.split_ts, side="right"))
            cache.append((patient.patient_id, frame, split_idx, tvals))
        self._patient_cache = cache
        return cache

    def _covariate_series(self, frame: pd.DataFrame, col: str) -> np.ndarray:
        """Finite covariate channel (forward-fill interior gaps, zero the rest)."""
        if col not in frame.columns:
            return np.zeros(len(frame), dtype=np.float32)
        return frame[col].ffill().fillna(0.0).to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # Train-region DataFrame for MoiraiForecaster.fit
    # ------------------------------------------------------------------
    def _build_train_frame(
        self, past_cols: Sequence[str], known_cols: Sequence[str]
    ) -> pd.DataFrame:
        cov_cols = list(dict.fromkeys([*past_cols, *known_cols]))
        tail_steps = (
            None
            if self.train_tail_days is None
            else int(self.train_tail_days * 24 * 60 / self.interval_mins)
        )
        parts: List[pd.DataFrame] = []
        for patient_id, frame, split_idx, tvals in self._ensure_patient_cache():
            start = 0 if tail_steps is None else max(0, split_idx - tail_steps)
            sl = slice(start, split_idx)
            bg = tvals[sl] / MGDL_PER_MMOL
            block: Dict[str, Any] = {
                "patient_id": patient_id,
                "datetime": frame.index[sl],
                "bg_mM": bg,
            }
            for col in cov_cols:
                block[col] = self._covariate_series(frame, col)[sl]
            part = pd.DataFrame(block).dropna(subset=["bg_mM"])
            if len(part) >= self.forecast_length + 1:
                parts.append(part)
        if not parts:
            raise ValueError("No IOBP2 training rows built for Moirai fine-tune.")
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    # Temporal-val windows -> GluonTS ListDataset + truth matrix (mg/dL)
    # ------------------------------------------------------------------
    def _build_val_dataset(
        self,
        past_cols: Sequence[str],
        known_cols: Sequence[str],
        context_length: int,
    ) -> Tuple[Any, np.ndarray]:
        from gluonts.dataset.common import (  # pyright: ignore[reportMissingImports]
            ListDataset,
        )

        horizon = self.forecast_length
        freq = f"{self.interval_mins}min"
        entries: List[Dict[str, Any]] = []
        truth_rows: List[np.ndarray] = []

        for _, frame, split_idx, tvals in self._ensure_patient_cache():
            total = len(tvals)
            if total < context_length + horizon:
                continue
            index = frame.index
            past_mat = (
                np.vstack([self._covariate_series(frame, c) for c in past_cols])
                if past_cols
                else None
            )
            known_mat = (
                np.vstack([self._covariate_series(frame, c) for c in known_cols])
                if known_cols
                else None
            )

            anchor_start = max(split_idx, context_length)
            anchor_end = total - horizon
            taken = 0
            for anchor in range(anchor_start, anchor_end + 1, self.val_stride_steps):
                if (
                    taken >= self.val_windows_per_patient
                    or len(entries) >= self.max_val_windows_total
                ):
                    break
                ctx = tvals[anchor - context_length : anchor]
                if np.isnan(ctx).any():
                    continue
                future = tvals[anchor : anchor + horizon]
                if not np.isfinite(future).any():
                    continue
                entry: Dict[str, Any] = {
                    "start": index[anchor - context_length],
                    "target": (ctx / MGDL_PER_MMOL).astype(np.float32),
                }
                if past_mat is not None:
                    entry["past_feat_dynamic_real"] = past_mat[
                        :, anchor - context_length : anchor
                    ].astype(np.float32)
                if known_mat is not None:
                    entry["feat_dynamic_real"] = known_mat[
                        :, anchor - context_length : anchor + horizon
                    ].astype(np.float32)
                entries.append(entry)
                truth_rows.append(future.astype(np.float32))
                taken += 1
            if len(entries) >= self.max_val_windows_total:
                break

        if not entries:
            raise ValueError(
                f"No IOBP2 val windows built for context_length={context_length}."
            )
        dataset = ListDataset(entries, freq=freq)
        true_matrix = np.vstack(truth_rows)
        return dataset, true_matrix

    # ------------------------------------------------------------------
    def _suggest(self, trial: optuna.Trial) -> MoiraiSearchPoint:
        """Sample a search point. Subclasses may override to use a different
        search space (e.g. the Moirai-2/MoE LoRA sweep)."""
        return suggest_moirai_point(trial, self.bounds)

    def __call__(self, trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        MoiraiConfig, MoiraiForecaster = self._import_wrapper()
        point: MoiraiSearchPoint = self._suggest(trial)

        trial.set_user_attr("model_path", point.config_kwargs["model_path"])
        trial.set_user_attr("past_cols", point.past_cols)
        trial.set_user_attr("known_cols", point.known_cols)
        trial.set_user_attr("gpu", self.gpu_label)

        trial_dir = self.output_root / f"trial_{trial.number:05d}"
        forecaster = None
        try:
            train_frame = self._build_train_frame(point.past_cols, point.known_cols)
            trial.set_user_attr("n_train_rows", int(len(train_frame)))

            config = MoiraiConfig(**point.config_kwargs)
            forecaster = MoiraiForecaster(config)

            def _report_epoch(epoch: int, avg_loss: float) -> None:
                # Report the per-epoch training loss as the pruning signal
                # (Moirai fine-tune has no cheap per-epoch validation; lower
                # training loss broadly tracks a better fit). The median pruner
                # compares these across trials at the same epoch.
                trial.report(float(avg_loss), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            epoch_callback: Optional[Callable[[int, float], None]] = (
                _report_epoch if self.report_intermediate else None
            )

            fit_start = time.time()
            fit_out = forecaster.fit(
                train_frame,
                output_dir=str(trial_dir),
                epoch_callback=epoch_callback,
            )
            fit_seconds = time.time() - fit_start
            trial.set_user_attr("fit_seconds", round(fit_seconds, 1))
            trial.set_user_attr("n_train_windows", int(fit_out.get("samples", 0)))
            trial.set_user_attr("best_loss", fit_out.get("best_loss"))

            val_dataset, true_matrix = self._build_val_dataset(
                point.past_cols, point.known_cols, point.context_length
            )
            # gluonts ListDataset is a plain list of dicts, which the wrapper's
            # predict-input normalizer would misroute to the episode-dict path;
            # drive the fitted predictor directly instead.
            predictor = forecaster._get_or_create_predictor(
                batch_size=config.batch_size
            )
            forecasts = list(predictor.predict(val_dataset))
            preds_mmol = np.stack([f.mean for f in forecasts], axis=0)
            pred_matrix = np.asarray(preds_mmol, dtype=float) * MGDL_PER_MMOL
            if pred_matrix.ndim == 1:
                pred_matrix = pred_matrix[None, :]
            pred_matrix = pred_matrix[:, :_HORIZON_LEN]
            metrics = evaluate_horizons(pred_matrix, true_matrix)
        except optuna.TrialPruned:
            self._cleanup(forecaster, None)
            raise
        except Exception as exc:  # noqa: BLE001 - degrade to a finite penalty
            logger.warning("Trial %d failed: %s", trial.number, exc)
            trial.set_user_attr("error", str(exc)[:500])
            self._cleanup(forecaster, trial_dir)
            if self.is_multi:
                return tuple(FAILURE_PENALTY for _ in self.objective_metrics or [])
            return FAILURE_PENALTY
        finally:
            if not self.keep_models and trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)

        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("n_val_windows", int(true_matrix.shape[0]))

        self._cleanup(forecaster, None)

        if self.is_multi:
            return tuple(
                float(metrics.get(name, FAILURE_PENALTY))
                for name in self.objective_metrics or []
            )
        return float(metrics.get(self.primary_metric, FAILURE_PENALTY))

    # ------------------------------------------------------------------
    def _cleanup(self, forecaster: Any, trial_dir: Optional[Path]) -> None:
        try:
            import torch  # pyright: ignore[reportMissingImports]

            if forecaster is not None:
                forecaster.model = None
                forecaster.predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            pass
        if trial_dir is not None and not self.keep_models and trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
