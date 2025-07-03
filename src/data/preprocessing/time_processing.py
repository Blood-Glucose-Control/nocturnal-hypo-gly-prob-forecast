"""
Time-based preprocessing utilities for continuous glucose monitoring data.

This module provides functions for handling, normalizing, and processing time-series data
specifically tailored for diabetes monitoring applications. It includes functionality for:

1. Creating datetime indices from time strings
2. Computing time differences between sequential measurements
3. Detecting the most common sampling interval in a dataset
4. Ensuring regular time intervals by filling missing timestamps

These utilities help prepare time-series data for analysis by establishing consistent
temporal representations across different patient datasets, handling time gaps, and
normalizing sampling frequencies.

Functions:
    create_datetime_index: Create a proper datetime index from time strings
    _create_time_diff_cols: Add time difference columns between consecutive measurements
    get_most_common_time_interval: Determine the most frequent sampling interval
    ensure_regular_time_intervals: Normalize data to have consistent time intervals
"""

import pandas as pd


def create_datetime_index(
    df: pd.DataFrame, start_date: str = "2025-01-01"
) -> pd.DataFrame:
    """
    Creates a datetime index for the dataframe.
    """
    if "time_diff" not in df.columns:
        df = _create_time_diff_cols(df)

    # Create a str datetime column
    df["datetime"] = pd.to_datetime(
        start_date + " " + df["time"], format="%Y-%m-%d %H:%M:%S"
    )

    # Convert time_diff to a timedelta object
    # time_diff is the difference in time between the current row and the next row
    df["time_diff"] = pd.to_timedelta(df["time_diff"])

    # Create a cumulative sum of the absolute time differences
    df["cumulative_diff"] = (
        df.groupby("p_num")["time_diff"].cumsum().fillna(pd.Timedelta(0))
    )

    # Create a fixed start timestamp for each patient
    start_times = df.groupby("p_num")["datetime"].transform("first")

    # Add cumulative time difference to the fixed start timestamp
    df["datetime"] = start_times + df["cumulative_diff"]

    # Drop the intermediate columns
    df.drop(columns=["time_diff", "cumulative_diff"], inplace=True)

    return df


def _create_time_diff_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'time_diff' column to the DataFrame that represents the time difference between
    consecutive rows for each patient, grouped by patient. The time difference is in minutes.
    Also adds a 'datetime' column to the DataFrame.
    """
    df["temp_datetime"] = pd.to_datetime(df["time"], format="%H:%M:%S")

    # First create raw time differences
    df["time_diff_raw"] = df.groupby("p_num")["temp_datetime"].diff()

    # time_diff_raw is negative across day boundaries e.g. when the current row is at 23:59:59 and the next is at 00:00:00
    # Create a new column with the absolute value of time_diff
    # Have to assume that two entries are not separated by more than 24 hours
    # Since we were not given date, only time, so not considering days in this calculation
    df["time_diff"] = df["time_diff_raw"].apply(
        lambda x: (
            pd.Timedelta(
                hours=x.components.hours,
                minutes=x.components.minutes,
                seconds=x.components.seconds,
            )
            if pd.notna(x)
            else pd.Timedelta(0)
        )
    )

    # Clean up intermediate columns
    df.drop(columns=["temp_datetime", "time_diff_raw"], inplace=True)
    return df


def get_most_common_time_interval(df: pd.DataFrame) -> int:
    """
    Determines the most common time interval between consecutive records in minutes.

    This function calculates the time differences between consecutive rows in the
    DataFrame's 'datetime' column and identifies the most frequently occurring interval.

    Args:
        df: DataFrame containing a 'datetime' column with timestamp data

    Returns:
        int: The most common time interval in minutes between consecutive records
    """
    df_copy = df.copy()
    df_copy["datetime"] = pd.to_datetime(df_copy["datetime"])

    # Calculate time differences in minutes directly
    df_copy["time_diff"] = (
        (df_copy["datetime"].diff().dt.total_seconds() / 60).fillna(0).astype(int)
    )

    # Get the most common value
    time_diff_counts = df_copy["time_diff"].value_counts()
    if len(time_diff_counts) > 0:
        return int(
            time_diff_counts.idxmax()
        )  # idxmax() returns the index with maximum count
    else:
        return 0  # Default value if there are no valid time differences
