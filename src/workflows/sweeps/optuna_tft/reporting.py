# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

# pyright: reportMissingImports=false
"""Optuna intermediate-value reporting for the AutoGluon TFT sweep.

AutoGluon trains the TFT inside a black-box ``predictor.fit()``; to surface
per-epoch training curves to Optuna we subclass the model and append a Lightning
callback in ``_get_callbacks`` (which AutoGluon *flushes before pickling*, so the
non-picklable callback + Optuna trial never hit model serialization).

This module imports AutoGluon + Lightning at import time, so it is imported
**lazily** (only inside the objective when ``report_intermediate`` is enabled and
the study is single-objective). The subclass is module-level (auto-registered by
AutoGluon's model-registry metaclass) so trained model instances remain picklable.
"""

from __future__ import annotations

from typing import Optional

import optuna
from autogluon.timeseries.models import TemporalFusionTransformerModel

# The trial whose training is currently in progress in THIS worker process. Set
# around each fit; the callback reads it so it holds no (unpicklable) trial state.
_CURRENT_REPORT_TRIAL: Optional[optuna.Trial] = None


def set_report_trial(trial: Optional[optuna.Trial]) -> None:
    global _CURRENT_REPORT_TRIAL
    _CURRENT_REPORT_TRIAL = trial


def _make_callback():
    import lightning.pytorch as pl

    class _OptunaEpochReport(pl.Callback):
        """Report the TFT's per-epoch validation loss to the active Optuna trial."""

        def __init__(self) -> None:
            self._step = 0

        def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
            if getattr(trainer, "sanity_checking", False):
                return
            trial = _CURRENT_REPORT_TRIAL
            if trial is None:
                return
            metric = trainer.callback_metrics.get("val_loss")
            if metric is None:
                return
            self._step += 1
            try:
                trial.report(float(metric), step=self._step)
            except Exception:  # noqa: BLE001 - never let reporting break training
                pass

    return _OptunaEpochReport()


class ReportingTFTModel(TemporalFusionTransformerModel):
    """TFT that additionally reports per-epoch val_loss to the active Optuna trial.

    Auto-registered by AutoGluon's metaclass under the key ``ReportingTFT``. The
    reporting callback is added to ``self.callbacks`` (flushed on save), so model
    pickling is unaffected.
    """

    def _get_callbacks(self, time_limit, early_stopping_patience=None):  # noqa: ANN001
        callbacks = super()._get_callbacks(time_limit, early_stopping_patience)
        if _CURRENT_REPORT_TRIAL is not None:
            callbacks.append(_make_callback())
        return callbacks
