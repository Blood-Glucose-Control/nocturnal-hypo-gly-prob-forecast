# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for single/multi-objective config parsing."""

import pytest

pytest.importorskip("optuna")
pytest.importorskip("yaml")

from src.workflows.sweeps.optuna_chronos2.study import (  # noqa: E402
    _parse_objectives,
)


def test_parse_single_objective_default():
    metrics, directions = _parse_objectives({"primary_metric": "mard_120"})
    assert metrics == ["mard_120"]
    assert directions == ["minimize"]


def test_parse_single_objective_direction():
    metrics, directions = _parse_objectives(
        {"primary_metric": "dts_a_pct_120", "direction": "maximize"}
    )
    assert metrics == ["dts_a_pct_120"]
    assert directions == ["maximize"]


def test_parse_multi_objective():
    config = {
        "objectives": [
            {"metric": "mard_120", "direction": "minimize"},
            {"metric": "dts_a_pct_120", "direction": "maximize"},
        ]
    }
    metrics, directions = _parse_objectives(config)
    assert metrics == ["mard_120", "dts_a_pct_120"]
    assert directions == ["minimize", "maximize"]
