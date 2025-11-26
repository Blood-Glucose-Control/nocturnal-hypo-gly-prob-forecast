"""
TTM (TinyTimeMixer) model implementation.

This package provides a unified interface for TTM models following the
base TSFM framework.
"""

from .model import TTMForecaster, TTMConfig, create_ttm_model

__all__ = ["TTMForecaster", "TTMConfig", "create_ttm_model"]
