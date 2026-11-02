# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for pruner construction (no chronos/GPU required)."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.sweeps.optuna_chronos2.pruning import build_pruner  # noqa: E402


def test_build_pruner_disabled_returns_none():
    assert build_pruner({"enable_pruning": False}) is None
    assert build_pruner({}) is None


def test_build_pruner_median_default():
    pruner = build_pruner({"enable_pruning": True})
    assert isinstance(pruner, optuna.pruners.MedianPruner)


def test_build_pruner_asha():
    pruner = build_pruner(
        {
            "enable_pruning": True,
            "pruner": {"type": "asha", "min_resource": 200, "reduction_factor": 4},
        }
    )
    assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)


def test_build_pruner_hyperband():
    pruner = build_pruner({"enable_pruning": True, "pruner": {"type": "hyperband"}})
    assert isinstance(pruner, optuna.pruners.HyperbandPruner)


def test_build_pruner_unknown_type_raises():
    with pytest.raises(ValueError):
        build_pruner({"enable_pruning": True, "pruner": {"type": "definitely-typo"}})


def test_patch_appends_active_callback_and_calls_original():
    from src.workflows.sweeps.optuna_chronos2 import pruning

    calls = {}

    class FakePipeline:
        def fit(self, *args, callbacks=None, **kwargs):
            calls["callbacks"] = callbacks
            return "fitted"

    assert pruning._patch_pipeline_fit(FakePipeline) is True

    sentinel = object()
    with pruning.pruning_context(sentinel):
        result = FakePipeline().fit(inputs=1)
    assert result == "fitted"
    assert calls["callbacks"] == [sentinel]

    # Outside the context no callback is injected.
    calls.clear()
    FakePipeline().fit(inputs=2)
    assert sentinel not in (calls["callbacks"] or [])


def test_patch_skips_when_fit_rejects_callbacks():
    from src.workflows.sweeps.optuna_chronos2 import pruning

    class NoCallbacks:
        def fit(self, inputs):  # no callbacks param, no **kwargs
            return inputs

    assert pruning._patch_pipeline_fit(NoCallbacks) is False
    # Original method is left untouched.
    assert NoCallbacks().fit(5) == 5
