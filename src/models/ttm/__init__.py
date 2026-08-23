"""
TTM (TinyTimeMixer) model implementation.

This package provides a unified interface for TTM models following the
base TSFM framework.
"""

from .config import TTMConfig
from .model import TTMForecaster

__all__ = [
    "TTMForecaster",
    "TTMConfig",
]
