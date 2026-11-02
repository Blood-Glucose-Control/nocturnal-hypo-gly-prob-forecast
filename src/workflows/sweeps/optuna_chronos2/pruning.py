# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Optuna pruning for Chronos-2 fine-tuning (guarded AutoGluon integration).

AutoGluon's ``Chronos2Model._fine_tune`` builds the HuggingFace Trainer callback
list internally and exposes no public hook, so we inject an Optuna pruning
callback by patching ``chronos.chronos2.pipeline.Chronos2Pipeline.fit`` to append
a callback pulled from a context variable. When ``eval_during_fine_tune`` is on,
the Trainer evaluates every 100 steps; on each evaluation we report the
validation loss to Optuna and stop training early if the trial should be pruned.

Everything is guarded: if the AutoGluon/chronos internals differ (e.g. a version
bump), :func:`install_pruning_patch` logs a warning and returns False, and the
objective simply runs without pruning.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Holds the list of extra Trainer callbacks (pruning/reporting, early stopping,
# step telemetry) for the trial currently being fine-tuned in this process. The
# patched pipeline.fit reads it and appends them all to the Trainer.
_ACTIVE_CALLBACKS: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "active_trainer_callbacks", default=None
)
_PATCH_INSTALLED = False


def make_early_stopping_callback(patience: int):
    """HF ``EarlyStoppingCallback`` that stops fine-tuning once the validation
    metric (``eval_loss``, best-checkpoint metric) has not improved for
    ``patience`` evaluations. Requires ``load_best_model_at_end`` (set by the
    Chronos-2 pipeline when ``eval_during_fine_tune`` is on).
    """
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=int(patience))


def make_step_recorder_callback():
    """Return ``(callback, holder)`` where ``holder['steps']`` is updated to the
    latest HF ``global_step`` so telemetry reflects the ACTUAL steps trained
    (the surviving best checkpoint alone under-reports it when
    ``save_total_limit`` deletes later checkpoints).
    """
    from transformers.trainer_callback import TrainerCallback

    holder: dict[str, Optional[int]] = {"steps": None}

    class _StepRecorder(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            holder["steps"] = int(state.global_step)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            holder["steps"] = int(state.global_step)
            return control

    return _StepRecorder(), holder


def make_pruning_callback(
    trial, report_metric: str = "eval_loss", *, prune: bool = True
):
    """Build a HuggingFace ``TrainerCallback`` that reports to Optuna.

    Always reports the validation metric as an Optuna intermediate value (so the
    dashboard shows per-trial training curves). When ``prune`` is True it also
    stops the trial early if ``trial.should_prune()``; when False it is a
    reporting-only callback (used for single-objective runs that want training
    curves without pruning). NOTE: this must never be attached for
    multi-objective studies — Optuna's ``trial.report()`` raises
    ``NotImplementedError`` for them.
    """
    import optuna
    from transformers.trainer_callback import TrainerCallback

    class OptunaPruningCallback(TrainerCallback):
        def __init__(self) -> None:
            self.trial = trial
            self.report_metric = report_metric
            self.prune = prune
            self.pruned = False
            self.last_value: Optional[float] = None
            self.n_reports = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return control
            value = metrics.get(self.report_metric)
            if value is None:
                for key, candidate in metrics.items():
                    if key.startswith("eval_") and isinstance(candidate, (int, float)):
                        value = candidate
                        break
            if value is None:
                return control
            self.last_value = float(value)
            self.n_reports += 1
            self.trial.report(float(value), step=int(state.global_step))
            if self.prune and self.trial.should_prune():
                self.pruned = True
                control.should_training_stop = True
            return control

    _ = optuna  # ensure optuna import is required at call time
    return OptunaPruningCallback()


def install_pruning_patch() -> bool:
    """Patch ``Chronos2Pipeline.fit`` to inject the active pruning callback.

    Idempotent and guarded; returns True if pruning is available, else False.
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return True
    try:
        from chronos.chronos2.pipeline import (  # pyright: ignore[reportMissingImports]
            Chronos2Pipeline,
        )
    except Exception as exc:  # pragma: no cover - version/import guard
        logger.warning("Pruning unavailable (chronos import failed): %s", exc)
        return False

    _PATCH_INSTALLED = _patch_pipeline_fit(Chronos2Pipeline)
    return _PATCH_INSTALLED


def _accepts_callbacks(func) -> bool:
    """True if ``func`` accepts a ``callbacks`` keyword (named or via **kwargs)."""
    import inspect

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - unintrospectable
        return True
    if "callbacks" in params:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in params.values())


def _patch_pipeline_fit(pipeline_cls) -> bool:
    """Wrap ``pipeline_cls.fit`` to append the active pruning callback.

    Returns False (and does not patch) if ``fit`` cannot accept ``callbacks`` or
    is already patched-idempotent. Separated from :func:`install_pruning_patch`
    so it can be unit-tested against a stub pipeline class.
    """
    original_fit = pipeline_cls.fit
    if getattr(original_fit, "_optuna_pruning_patch", False):
        return True
    if not _accepts_callbacks(original_fit):
        logger.warning(
            "%s.fit does not accept 'callbacks'; pruning disabled.",
            pipeline_cls.__name__,
        )
        return False

    def patched_fit(self, *args, **kwargs):
        active = _ACTIVE_CALLBACKS.get()
        if active:
            callbacks = list(kwargs.get("callbacks") or [])
            callbacks.extend(active)
            kwargs["callbacks"] = callbacks
        return original_fit(self, *args, **kwargs)

    patched_fit._optuna_pruning_patch = True  # type: ignore[attr-defined]
    pipeline_cls.fit = patched_fit
    logger.info("Installed Optuna pruning patch on %s.fit", pipeline_cls.__name__)
    return True


class pruning_context:
    """Context manager binding extra Trainer callbacks for the wrapped fit call.

    Accepts a single callback or a list; they are appended to the Trainer's
    callbacks by the patched ``pipeline.fit``.
    """

    def __init__(self, callbacks) -> None:
        if callbacks is None:
            self.callbacks: list = []
        elif isinstance(callbacks, (list, tuple)):
            self.callbacks = list(callbacks)
        else:
            self.callbacks = [callbacks]
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE_CALLBACKS.set(self.callbacks)
        return self.callbacks

    def __exit__(self, *exc) -> bool:
        if self._token is not None:
            _ACTIVE_CALLBACKS.reset(self._token)
        return False


def build_pruner(config: dict):
    """Construct an Optuna pruner from a config dict, or None if disabled."""
    import optuna

    if not config.get("enable_pruning"):
        return None
    spec = config.get("pruner", {}) or {}
    kind = str(spec.get("type", "median")).lower()
    if kind in ("asha", "successive_halving", "successivehalving"):
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=spec.get("min_resource", 100),
            reduction_factor=spec.get("reduction_factor", 3),
            min_early_stopping_rate=spec.get("min_early_stopping_rate", 0),
        )
    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=spec.get("min_resource", 100),
            max_resource=spec.get("max_resource", "auto"),
            reduction_factor=spec.get("reduction_factor", 3),
        )
    if kind == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=spec.get("n_startup_trials", 5),
            n_warmup_steps=spec.get("n_warmup_steps", 0),
        )
    raise ValueError(
        f"Unknown pruner type {spec.get('type')!r}; use 'median', 'asha', "
        f"or 'hyperband'."
    )
