"""
Base model framework for Time Series Foundation Models.

This module provides the foundational classes and utilities for implementing
time series foundation models in a unified, scalable framework.
"""

from .base_model import (
    BaseTimeSeriesFoundationModel,
    ModelConfig,
    TrainingBackend,
    create_model_from_config,
)
from .registry import ModelRegistry

__all__ = [
    # Base model classes
    "BaseTimeSeriesFoundationModel",
    "ModelConfig",
    "TrainingBackend",
    # Factory functions
    "create_model_from_config",
    # Registry
    "ModelRegistry",
]
