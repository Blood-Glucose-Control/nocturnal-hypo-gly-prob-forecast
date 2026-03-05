"""TimesFM model for time series forecasting.

This module provides a wrapper around Google's TimesFM (Time Series Foundation Model)
for glucose forecasting tasks. Integrated with the BaseTimeSeriesFoundationModel framework for consistency.
"""

from .config import TimesFMConfig
from .model import TimesFMForecaster, create_timesfm_model

__all__ = ["TimesFMConfig", "TimesFMForecaster", "create_timesfm_model"]
