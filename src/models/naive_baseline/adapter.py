# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Reference ModelAdapter port for NaiveBaselineForecaster."""

from src.models.adapter import ModelAdapter, assert_model_adapter

from .config import NaiveBaselineConfig
from .model import NaiveBaselineForecaster


def build_naive_baseline_adapter(
    config: NaiveBaselineConfig,
) -> ModelAdapter[NaiveBaselineConfig]:
    """Return a protocol-typed NaiveBaselineForecaster instance.

    This is the Phase A reference port proving one existing model family can be
    consumed through the common ModelAdapter contract without behavior changes.
    """
    return assert_model_adapter(NaiveBaselineForecaster(config))
