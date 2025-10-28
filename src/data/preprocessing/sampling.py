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
from src.data.models import ColumnNames
from src.data.preprocessing.time_processing import get_most_common_time_interval
import logging
import numpy as np

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

    # Use merge_asof with tolerance to map shifted timestamps to regular intervals
    mapped_data = pd.merge_asof(
        regular_times,  # left
        original_data,  # right
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


# TODO: Write UnitTest for this
def ensure_regular_time_intervals_with_aggregation(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Ensures regular time intervals by aggregating all data points within each interval.

    Unlike ensure_regular_time_intervals which only takes one row, this function
    aggregates multiple rows that fall within the same regular interval window.

    - Blood glucose (ColumnNames.BG.value) is averaged
    - All other numerical columns are summed
    - Categorical columns take the first value

    Args:
        df: Input dataframe with datetime index

    Returns:
        Tuple[pd.DataFrame, int]: (DataFrame with regular intervals, frequency in minutes)
    """
    logger.info(
        "ensure_regular_time_intervals_with_aggregation(): Ensuring regular time intervals with aggregation..."
    )

    # Validate inputs
    if df.empty:
        return df.copy(), 0

    if df.shape[0] <= 1:
        raise ValueError("DataFrame must contain more than 1 row")

    if "p_num" not in df.columns:
        raise ValueError("DataFrame must contain 'p_num' column")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")

    freq = get_most_common_time_interval(df)
    logger.info(f"\tMost common time interval: {freq} minutes")

    # TODO: switch back to 2/3 or do we consider all data
    # tolerance = pd.Timedelta(minutes=freq * (3 / 3))

    # Create complete regular time range
    full_time_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{freq}min",
    )

    # Identify numerical vs non-numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove(ColumnNames.P_NUM.value)
    numerical_cols.remove("isf")
    numerical_cols.remove("cr")
    numerical_cols.remove("iob")

    # Prepare aggregation dict
    agg_dict = {}
    for col in df.columns:
        if col in numerical_cols:
            if col == ColumnNames.BG.value or col == ColumnNames.RATE.value:
                # Average blood glucose or basal rate
                agg_dict[col] = "mean"
            else:
                # Sum all other numerical values
                agg_dict[col] = "sum"
        else:
            # Take first non-numerical value (categorical columns)
            # TODO: First value can be Nan so might need to take the first non-Nan value?
            agg_dict[col] = lambda x: x.iloc[0] if len(x) > 0 else None

    logger.info(f"\tAggregation strategy: {agg_dict}")

    aggregated_rows = []
    for regular_time in full_time_range:
        time_diffs = df.index - regular_time

        # TODO: or should we use tolerance/2?
        # Take +- half of the interval around the regular time
        mask = abs(time_diffs) <= pd.Timedelta(minutes=freq // 2)

        candidates = df[mask]

        if len(candidates) > 0:
            # Aggregate the candidates
            aggregated_row = candidates.agg(agg_dict)
            aggregated_row.name = regular_time
            aggregated_rows.append(aggregated_row)
        else:
            # No data in this window, create NaN row
            nan_row = pd.Series(index=df.columns, name=regular_time, dtype=float)
            nan_row["p_num"] = df["p_num"].iloc[0]  # Keep p_num from original data
            aggregated_rows.append(nan_row)

    result_df = pd.DataFrame(aggregated_rows)
    result_df.index.name = "datetime"

    logger.info(
        f"Post-ensure_regular_time_intervals_with_aggregation(): \n\t\t\t"
        f"Patient {df['p_num'].iloc[0]} \n\t\t\t "
        f"- old index length: {len(df.index)}, \n\t\t\t "
        f"- new index length: {len(result_df.index)}"
    )

    return result_df, freq
