"""
Time series sampling and resampling utilities for continuous glucose monitoring data.

This module provides functions for handling irregular sampling in time series data,
particularly focused on glucose monitoring applications. It includes functionality for:

1. Detecting and fixing irregular time intervals
2. Resampling data to standard frequencies
3. Handling missing values during resampling operations
4. Ensuring consistent sampling across patient datasets

These utilities help prepare time-series data for analysis by standardizing sampling
frequencies, which is essential for many time series algorithms and cross-patient
comparisons.

Functions:
    ensure_regular_time_intervals: Normalize data to have consistent time intervals
    resample_to_frequency: Resample time series data to a specified frequency
"""

import pandas as pd
from typing import Literal, Tuple
from src.data.preprocessing.time_processing import get_most_common_time_interval
import logging

logger = logging.getLogger(__name__)


def ensure_regular_time_intervals(
    df: pd.DataFrame, direction: Literal["backward", "forward", "nearest"] = "forward"
) -> Tuple[pd.DataFrame, int]:
    """Ensures regular time intervals exist in the dataframe by adding rows with NaN values
    where timestamps are missing, and maps shifted timestamps to the nearest regular interval.

    Args:
        df (pd.DataFrame): Input dataframe with datetime index

    Returns:
        pd.DataFrame: DataFrame with regular time intervals, missing times filled with NaN,
                     and shifted data mapped to nearest regular intervals
    """
    logger.info("ensure_regular_time_intervals(): Ensuring regular time intervals...")
    # Validate inputs
    if df.empty:
        return df.copy(), 0  # Return empty DataFrame with same structure

    if df.shape[0] <= 1:
        raise ValueError("DataFrame must contain more than 1 row")

    if "p_num" not in df.columns:
        raise ValueError("DataFrame must contain 'p_num' column")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")

    original_data = df.copy()
    freq = get_most_common_time_interval(df)
    logger.info(f"\tMost common time interval: {freq} minutes")

    # Create complete time range for this patient
    full_time_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{freq}min",
    )

    # Create a DataFrame with the complete time range
    full_df = pd.DataFrame(index=full_time_range)
    full_df.index.name = "datetime"

    # For each data point, find the nearest regular timestamp
    tolerance = pd.Timedelta(
        minutes=freq * (2 / 3)
    )  # Allow up to two-thirds of the interval as tolerance

    # Use merge_asof to match each original timestamp to the nearest regular timestamp
    # First, prepare the data
    # original_data = result_df.reset_index()
    regular_times = pd.DataFrame({"datetime": full_time_range})

    # Sort both dataframes by datetime
    # original_data = result_df.sort_values('datetime')
    # regular_times = regular_times.sort_values('datetime')

    # Use merge_asof with tolerance to map shifted timestamps to regular intervals
    mapped_data = pd.merge_asof(
        regular_times,
        original_data,
        on="datetime",
        tolerance=tolerance,
        direction=direction,
    )

    # Set datetime back as index
    mapped_data = mapped_data.set_index("datetime")
    mapped_data.index.name = "datetime"

    # For any regular timestamps that didn't get matched, we'll have NaN values
    # This preserves the regular time grid while capturing shifted data

    logger.info(
        f"Post-ensure_regular_time_intervals(): \n\t\t\tPatient {df['p_num'].iloc[0]} \n\t\t\t - old index length: {len(df.index)}, \n\t\t\t - new index length: {len(mapped_data.index)}"
    )

    return mapped_data, freq
