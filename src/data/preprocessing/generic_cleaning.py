# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Generic Data Cleaning Module

This module provides functions for processing and cleaning generic diabetes data.
It handles tasks such as standardizing timestamps, handling missing values,
processing meal data, and transforming data into a consistent format.

Key functionality includes:
- Converting timestamps to regular intervals
- Identifying and filtering days with excessive missing data
- Processing meal announcements and handling overlaps
- Selecting top carbohydrate meals per day
- Standardizing column names and formats
"""

import pandas as pd
from src.data.models import ColumnNames


def erase_consecutive_nan_values(
    patient_df: pd.DataFrame, max_consecutive_nan_values_per_day: int
):
    """
    1. If there are more than max_consecutive_nan_values_per_day consecutive NaN values in a given day, then delete that day from the dataframe.
    Parameters:
        patient_df: pd.DataFrame
            The input DataFrame with a datetime index.
        max_consecutive_nan_values_per_day: int
            The maximum number of consecutive NaN values allowed in a given day. If more than this number of consecutive NaN values are found in a day, then delete that day from the dataframe. Otherwise, delete the NaN values from that day.
    Returns:
        pd.DataFrame
            The processed DataFrame with consecutive NaN values handled.
    """
    # Create a copy to avoid modifying original
    df = patient_df.copy()

    # Add day column for grouping
    df["day"] = df.index.date

    # Process each day
    days_to_keep = []
    for day, day_data in df.groupby("day"):
        # Get boolean mask of NaN values
        nan_mask = day_data[ColumnNames.BG.value].isnull()

        # Count consecutive NaNs
        consecutive_nans = 0
        max_consecutive = 0
        for is_nan in nan_mask:
            if is_nan:
                consecutive_nans += 1
                max_consecutive = max(max_consecutive, consecutive_nans)
            else:
                consecutive_nans = 0

        # Keep day if max consecutive NaNs is within limit
        if max_consecutive <= max_consecutive_nan_values_per_day:
            days_to_keep.append(day)

    # Filter to keep only valid days
    result_df = df[df["day"].isin(days_to_keep)].copy()

    # Drop the temporary day column
    result_df.drop("day", axis=1, inplace=True)

    return result_df


def reduce_fp_precision(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the precision of the floating point columns to 2 decimal places to save space.
    """
    columns_to_reduce_precision = [
        ColumnNames.BG.value,
        ColumnNames.FOOD_G.value,
        ColumnNames.DOSE_UNITS.value,
        ColumnNames.IOB.value,
        ColumnNames.COB.value,
        ColumnNames.CARB_AVAILABILITY.value,
        ColumnNames.INSULIN_AVAILABILITY.value,
    ]
    df = df.copy()
    for col in columns_to_reduce_precision:
        if col in df.columns:
            df[col] = df[col].round(2)
    return df
