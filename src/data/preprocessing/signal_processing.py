"""
Signal processing utilities for continuous glucose monitoring data.

This module provides functions for processing and transforming signal data from glucose
monitoring systems. It includes techniques for noise reduction, feature extraction,
and signal transformation specifically designed for blood glucose time series data.

The utilities help prepare raw sensor data for analysis by:
- Smoothing noisy signals through moving averages
- Extracting relevant features from time series data
- Normalizing signals for cross-patient comparison

These operations are essential preprocessing steps before applying machine learning
algorithms or statistical analysis to glucose monitoring data.

Functions:
    apply_moving_average: Apply a rolling average to smooth glucose measurements
"""

import pandas as pd


def apply_moving_average(
    df: pd.DataFrame, window_size: int = 36, bg_col="bg_mM"
) -> pd.DataFrame:
    """
    Takes the moving average of the dataframe over bg_col

    Args:
        df: the dataframe
        window_size: the moving average window size
        bg_col: the column take moving average over
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")
    if bg_col not in df.columns:
        raise ValueError(f"Column '{bg_col}' not found in the DataFrame")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Apply moving average
    result_df[bg_col] = (
        result_df[bg_col].rolling(window=window_size, min_periods=1).mean()
    )
    return result_df
