"""
TTM (TinyTimeMixer) model implementation.

This package provides a unified interface for TTM models following the
base TSFM framework.
"""

from .model import TTMForecaster
from .config import (
    TTMConfig,
    TTMTrainingConfig,
    TTMDataConfig,
    create_default_ttm_config,
    create_ttm_fine_tuning_config,
    create_ttm_zero_shot_config,
)

__all__ = [
    "TTMForecaster",
    "TTMConfig",
    "TTMTrainingConfig",
    "TTMDataConfig",
    "create_default_ttm_config",
    "create_ttm_fine_tuning_config",
    "create_ttm_zero_shot_config",
]
