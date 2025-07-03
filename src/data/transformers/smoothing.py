"""
Smoothing transformers for time series data.

This module provides transformers for smoothing time series data using methods
such as moving averages, exponential smoothing, and other techniques to reduce
noise and highlight trends.
"""

import pandas as pd
from typing import Union, Optional

from src.data.transformers.base import BaseTransformer


class MovingAverageTransformer(BaseTransformer):
    """
    Apply a moving/rolling average transformation to time series data.

    This transformer smooths data using a sliding window approach, replacing each
    value with the average of itself and surrounding values.
    """

    def __init__(
        self,
        window: int = 5,
        center: bool = True,
        min_periods: Optional[int] = None,
        win_type: Optional[str] = None,
    ):
        """
        Initialize the moving average transformer.

        Args:
            window: Size of the moving window (number of observations)
            center: If True, the window is centered on the current value.
                   If False, previous values are used.
            min_periods: Minimum number of observations in window required to have a value.
                       Default is window size if None.
            win_type: Window type for weighted averages (None for simple moving average)
                    Options include: 'boxcar', 'triang', 'blackman', 'hamming', 'bartlett',
                    'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann', 'kaiser',
                    'gaussian', 'general_gaussian', 'exponential'.
        """
        self.window = window
        self.center = center
        self.min_periods = min_periods if min_periods is not None else window
        self.win_type = win_type

    def fit(
        self, X: Union[pd.Series, pd.DataFrame], y=None
    ) -> "MovingAverageTransformer":
        """
        No fitting necessary for moving average.

        Args:
            X: Input data (unused in this transformer)
            y: Ignored

        Returns:
            Self for method chaining
        """
        return self

    def transform(
        self, X: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply moving average to the input data.

        Args:
            X: Input data to transform (Series or DataFrame)

        Returns:
            Transformed data with same type as input
        """
        # If X is a Series
        if isinstance(X, pd.Series):
            return X.rolling(
                window=self.window,
                min_periods=self.min_periods,
                center=self.center,
                win_type=self.win_type,
            ).mean()

        # If X is a DataFrame
        elif isinstance(X, pd.DataFrame):
            return X.rolling(
                window=self.window,
                min_periods=self.min_periods,
                center=self.center,
                win_type=self.win_type,
            ).mean()

        else:
            raise TypeError("Input must be a pandas Series or DataFrame")


class ExponentialSmoothingTransformer(BaseTransformer):
    """
    Apply exponential smoothing to time series data.

    Exponential smoothing assigns exponentially decreasing weights to past observations,
    giving more weight to recent observations while still considering older values.
    """

    def __init__(self, alpha: float = 0.3, adjust: bool = True):
        """
        Initialize the exponential smoothing transformer.

        Args:
            alpha: Smoothing factor between 0 and 1. Higher values give more weight to recent data.
            adjust: If True, use the adjust algorithm to compute weights
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        self.adjust = adjust

    def fit(
        self, X: Union[pd.Series, pd.DataFrame], y=None
    ) -> "ExponentialSmoothingTransformer":
        """
        No fitting necessary for exponential smoothing.

        Args:
            X: Input data (unused in this transformer)
            y: Ignored

        Returns:
            Self for method chaining
        """
        return self

    def transform(
        self, X: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply exponential smoothing to the input data.

        Args:
            X: Input data to transform (Series or DataFrame)

        Returns:
            Transformed data with same type as input
        """
        return X.ewm(alpha=self.alpha, adjust=self.adjust).mean()
