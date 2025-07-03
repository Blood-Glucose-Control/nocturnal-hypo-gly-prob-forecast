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
import numpy as np
from typing import Optional, Literal
from src.data.preprocessing.time_processing import get_most_common_time_interval


def ensure_regular_time_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures regular time intervals exist in the dataframe by adding rows with NaN values
    where timestamps are missing.

    Args:
        df (pd.DataFrame): Input dataframe with datetime column

    Returns:
        pd.DataFrame: DataFrame with regular time intervals, missing times filled with NaN
    """
    result_df = df.copy()

    # Process each patient separately
    reindexed_dfs = []
    for patient_id, patient_df in result_df.groupby("p_num"):
        freq = get_most_common_time_interval(patient_df)

        # Create complete time range for this patient
        full_time_range = pd.date_range(
            start=patient_df["datetime"].min(),
            end=patient_df["datetime"].max(),
            freq=f"{freq}min",
        )

        # Create a DataFrame with the complete time range and ensure datetime type
        full_df = pd.DataFrame({"datetime": full_time_range})
        full_df["datetime"] = pd.to_datetime(full_df["datetime"])

        # Ensure patient_df datetime is also datetime type
        patient_df["datetime"] = pd.to_datetime(patient_df["datetime"])

        # Merge with original data
        reindexed_df = pd.merge(full_df, patient_df, on="datetime", how="left")

        # Restore patient ID for all rows
        reindexed_df["p_num"] = patient_id

        # Generate new sequential IDs for all rows
        sequence_numbers = range(len(reindexed_df))
        reindexed_df["id"] = [f"{patient_id}_{i}" for i in sequence_numbers]

        reindexed_dfs.append(reindexed_df)

    # Combine all reindexed patient data
    result_df = pd.concat(reindexed_dfs)

    return result_df

#TODO: Evaluate whether this function should replace `ensure_regular_time_intervals` or serve as a complementary utility.
def ensure_regular_time_intervals_with_interpolation(
    df: pd.DataFrame, 
    datetime_col: str = "datetime", 
    target_interval_minutes: Optional[int] = None,
    interpolation_method: str = "linear"
) -> pd.DataFrame:
    """
    Ensures that the time series data has regular intervals.
    
    This function resamples the data to create a regular time series with consistent
    intervals between measurements. If no target interval is provided, it uses the
    most common interval found in the data.
    
    Args:
        df: DataFrame containing time series data
        datetime_col: Name of column containing datetime values
        target_interval_minutes: Desired interval in minutes between measurements
                                If None, uses the most common interval in the data
        interpolation_method: Method to use for interpolating missing values
                            Options: 'linear', 'time', 'nearest', 'zero', 'slinear',
                            'quadratic', 'cubic', 'polynomial'
    
    Returns:
        DataFrame with regular time intervals
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Make sure datetime column is datetime type
    result_df[datetime_col] = pd.to_datetime(result_df[datetime_col])
    
    # Determine target interval if not provided
    if target_interval_minutes is None:
        # Calculate time differences in minutes
        result_df["_time_diff"] = (result_df[datetime_col].diff().dt.total_seconds() / 60)
        # Get the most common interval (excluding NaN values)
        time_diffs = result_df["_time_diff"].dropna()
        if len(time_diffs) > 0:
            # Round to nearest minute to account for small variations
            rounded_diffs = time_diffs.round().astype(int)
            target_interval_minutes = rounded_diffs.mode().iloc[0]
        else:
            target_interval_minutes = 5  # Default to 5 minutes if can't determine
        
        # Clean up temporary column
        result_df = result_df.drop(columns=["_time_diff"])
    
    # Set datetime as index for resampling
    result_df = result_df.set_index(datetime_col)
    
    # Create a new index with regular intervals
    min_time = result_df.index.min()
    max_time = result_df.index.max()
    
    # Create new index with regular frequency
    new_index = pd.date_range(
        start=min_time,
        end=max_time,
        freq=f"{target_interval_minutes}min"
    )
    
    # Reindex and interpolate
    resampled_df = result_df.reindex(
        pd.DatetimeIndex(new_index).union(result_df.index).sort_values()
    )
    
    # Interpolate missing values
    VALID_INTERPOLATION_METHODS = ('linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline')
    if interpolation_method in VALID_INTERPOLATION_METHODS:
        resampled_df = resampled_df.interpolate(method=interpolation_method)
    else:
        raise ValueError(f"Invalid interpolation method: {interpolation_method}. Must be one of {VALID_INTERPOLATION_METHODS}")
    
    # Keep only the points at the target frequency
    resampled_df = resampled_df.loc[new_index]
    
    # Reset index to make datetime a column again
    resampled_df = resampled_df.reset_index().rename(columns={"index": datetime_col})
    
    return resampled_df

#TODO: Verify that this function can replaced the above function and if it is an improvement.


def resample_to_frequency_ainewdontuse(
    df: pd.DataFrame,
    freq: str = "5min",
    datetime_col: str = "datetime",
    value_cols: Optional[list] = None,
    method: Literal['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'] = "linear"
) -> pd.DataFrame:
    """
    Resample time series data to a specified frequency.
    
    Args:
        df: DataFrame containing time series data
        freq: Target frequency as a string (e.g., '5min', '1H', '1D')
        datetime_col: Name of column containing datetime values
        value_cols: List of column names to resample (if None, all numeric columns)
        method: Interpolation method for filling gaps
        
    Returns:
        DataFrame resampled to the specified frequency
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Ensure datetime column is proper datetime
    result_df[datetime_col] = pd.to_datetime(result_df[datetime_col])
    
    # Set datetime as index
    result_df = result_df.set_index(datetime_col)
    
    # If value_cols not provided, use all numeric columns
    if value_cols is None:
        value_cols = result_df.select_dtypes(include=np.number).columns.tolist()
    
    # Resample the data
    resampled_df = result_df[value_cols].resample(freq).asfreq()
    
    # Interpolate missing values
    resampled_df = resampled_df.interpolate(method=method)
    
    # Add back any non-numeric columns with forward fill
    non_numeric_cols = [col for col in result_df.columns if col not in value_cols]
    if non_numeric_cols:
        for col in non_numeric_cols:
            resampled_df[col] = result_df[col].resample(freq).ffill()
    
    # Reset index to make datetime a column again
    resampled_df = resampled_df.reset_index().rename(columns={"index": datetime_col})
    
    return resampled_df