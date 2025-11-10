# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Time-based preprocessing utilities for continuous glucose monitoring data.

This module provides functions for handling, normalizing, and processing time-series data
specifically tailored for diabetes monitoring applications. It includes functionality for:

1. Creating datetime indices from time strings
2. Computing time differences between sequential measurements
3. Detecting the most common sampling interval in a dataset
4. Ensuring regular time intervals by filling missing timestamps
5. Splitting patient data into daily segments based on configurable day transitions
6. Creating robust train-validation splits for time-series forecasting models

These utilities help prepare time-series data for analysis by establishing consistent
temporal representations across different patient datasets, handling time gaps, and
normalizing sampling frequencies. Day-based splitting allows for analyzing patterns
within clinically meaningful 24-hour periods with flexible day start times. The enhanced
train-validation splitting functionality supports machine learning model development with
proper temporal separation, comprehensive input validation, flexible patient column
specification, and detailed split metadata reporting.

Functions:
    create_datetime_index: Create a proper datetime index from time strings
    _create_time_diff_cols: Add time difference columns between consecutive measurements
    get_most_common_time_interval: Determine the most frequent sampling interval
    ensure_regular_time_intervals: Normalize data to have consistent time intervals
    split_patient_data_by_day: Split patient data into daily segments based on 6am transitions
    get_train_validation_split: Create robust train and validation sets for a single patient with DatetimeIndex requirement for optimal performance
"""

from typing import Generator, cast
from typing_extensions import deprecated
import pandas as pd

from src.data.models import ColumnNames


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
    DataFrame's datetime index and identifies the most frequently occurring interval.

    Args:
        df: DataFrame with datetime index containing timestamp data

    Returns:
        int: The most common time interval in minutes between consecutive records
    """
    # Check if datetime is the index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")

    # Calculate time differences in minutes directly from index
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60

    # Get the most common value (excluding NaN)
    time_diff_counts = time_diffs.dropna().value_counts()
    if len(time_diff_counts) > 0:
        return int(
            time_diff_counts.idxmax()
        )  # idxmax() returns the index with maximum count
    else:
        return 0  # Default value if there are no valid time differences


def split_patient_data_by_day(patients_dfs: pd.DataFrame, patient_id: str) -> dict:
    """
    Split patient data into daily dataframes based on 6am transitions.

    Args:
        df: Input DataFrame containing patient data
        patient: Patient ID (e.g. 'p01')

    Returns:
        Dictionary of DataFrames, with keys as '{patient_id}_{day}' and values as daily DataFrames
    """
    # Convert time column to datetime and extract time and hour components
    patient_copy = patients_dfs[patients_dfs["p_num"] == patient_id].copy()

    patient_copy["time"] = pd.to_datetime(patient_copy["time"]).dt.strftime("%H:%M:%S")
    patient_copy["hour"] = pd.to_datetime(patient_copy["time"]).dt.hour

    # Initialize variables
    day_count = 0
    prev_hour = None
    patient_copy["day"] = 0

    # Traverse rows one by one
    for idx, row in patient_copy.iterrows():
        # Check if we transitioned from 5:xx to 6:xx
        if prev_hour == 5 and row["hour"] == 6:
            day_count += 1

        patient_copy.at[idx, "day"] = day_count
        prev_hour = row["hour"]

    patient_copy.drop(columns=["hour"], inplace=True)

    # Split data into dictionary of daily dataframes
    daily_dfs = {}
    for day in range(patient_copy["day"].max() + 1):
        daily_df = patient_copy[patient_copy["day"] == day].copy()
        # Drop id and p_num columns since they're in the key
        daily_df = daily_df.drop(columns=["id", "p_num", "day"])
        key = f"{patient_id}_{day}"
        daily_dfs[key] = daily_df

    return daily_dfs


@deprecated(
    "Use get_train_validation_split_by_percentage instead. Datasets will have different spans so it no longer makes sense to use a fixed number of days. This function will only be used internally.",
    category=DeprecationWarning,
)
def get_train_validation_split(
    df: pd.DataFrame,
    num_validation_days: int = 20,
    day_start_hour: int = 6,
    min_data_days: int = 1,
    include_partial_days: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Split a single patient's data into train and validation sets based on complete days.

    This function separates a single patient's data into training and validation sets,
    where the validation set contains the specified number of complete days. A day is
    defined as starting at the specified hour (default: 6am) and ending just before
    the same hour the next day.

    IMPORTANT: This function requires the input DataFrame to have a DatetimeIndex for
    optimal performance and semantic correctness of time-series operations.

    Args:
        df (pd.DataFrame): Input dataframe for a SINGLE patient with DatetimeIndex (REQUIRED)
        num_validation_days (int): Number of complete days to use for validation
        day_start_hour (int): Hour that defines the start of a day (default: 6 for 6am)
        min_data_days (int): Minimum number of days of data required (default: 1)
        include_partial_days (bool): Whether to include partial days at the end of data (default: False)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]:
            (train_data, validation_data, split_info)
            where train_data and validation_data are DataFrames for the single patient

    Raises:
        ValueError: If DatetimeIndex is not found or insufficient data for requested validation period
        TypeError: If the DataFrame index is not a DatetimeIndex
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Check for DatetimeIndex (REQUIRED for optimal performance)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame must have a DatetimeIndex for optimal time-series operations. "
            "Please set your datetime column as the index using: "
            "df.set_index('datetime_column_name', inplace=True)"
        )

    if num_validation_days <= 0:
        raise ValueError("num_validation_days must be positive")

    if not 0 <= day_start_hour <= 23:
        raise ValueError("day_start_hour must be between 0 and 23")

    if min_data_days <= 0:
        raise ValueError("min_data_days must be positive")

    # Work with a copy to avoid modifying original
    df = df.copy()

    # Initialize split info dictionary
    split_info = {
        "data_start": df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
        "data_end": df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(df),
        "validation_days_requested": num_validation_days,
        "validation_days_actual": 0,
        "day_start_hour": day_start_hour,
        "include_partial_days": include_partial_days,
        "dropped_partial_day_records": 0,
        "dropped_partial_day_start": None,
        "dropped_partial_day_end": None,
    }

    # Type assertion to help the type checker
    datetime_index = cast(pd.DatetimeIndex, df.index)

    # Get timestamps where hour matches day_start_hour (using efficient index operations)
    day_boundary_mask = datetime_index.hour == day_start_hour
    day_boundary_times = datetime_index[day_boundary_mask]

    # Check if patient has enough data
    if len(day_boundary_times) == 0:
        raise ValueError(
            f"No {day_start_hour}:00 timestamps found in the data. "
            f"Cannot determine day boundaries for splitting."
        )

    # Calculate total days of data available (using efficient index operations)
    total_data_span = (df.index.max() - df.index.min()).days

    if total_data_span < min_data_days:
        raise ValueError(
            f"Insufficient data: {total_data_span} days < {min_data_days} required"
        )

    # Get the last day boundary timestamp
    if include_partial_days:
        # Use the latest data point
        last_boundary = df.index.max()
    else:
        # Use the last day_start_hour timestamp
        last_boundary = day_boundary_times.max()

        # Calculate dropped partial day information when excluding partial days
        original_end = df.index.max()
        if original_end > last_boundary:
            dropped_partial_data = df.loc[last_boundary:].iloc[1:]
            split_info.update(
                {
                    "dropped_partial_day_records": len(dropped_partial_data),
                    "dropped_partial_day_start": last_boundary.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "dropped_partial_day_end": original_end.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )

    # Calculate the start of validation period
    validation_start = last_boundary - pd.Timedelta(days=num_validation_days)

    # Ensure we don't go before the start of available data
    data_start = df.index.min()
    if validation_start < data_start:
        available_days = (last_boundary - data_start).days
        raise ValueError(
            f"Insufficient data for {num_validation_days} validation days: "
            f"only {available_days} days available"
        )

    # Split the data using efficient index-based slicing
    if include_partial_days:
        # Include all data from validation_start onwards
        validation_data = df.loc[validation_start:].copy()
        train_data = df.loc[:validation_start].iloc[:-1].copy()
    else:
        # Use precise time range slicing
        validation_data = df.loc[validation_start:last_boundary].copy()
        train_data = df.loc[:validation_start].iloc[:-1].copy()

    # Ensure we have meaningful data
    if len(validation_data) == 0:
        raise ValueError("Validation set is empty after split")

    if len(train_data) == 0:
        raise ValueError("Training set is empty after split")

    # Calculate actual validation days
    actual_validation_days = (
        validation_data.index.max() - validation_data.index.min()
    ).days

    # Update split info with results
    split_info.update(
        {
            "validation_days_actual": actual_validation_days,
            "train_records": len(train_data),
            "validation_records": len(validation_data),
            "split_ratio": len(validation_data) / len(df),
            "train_data_start": train_data.index.min().strftime("%Y-%m-%d %H:%M:%S"),
            "train_data_end": train_data.index.max().strftime("%Y-%m-%d %H:%M:%S"),
            "validation_data_start": validation_data.index.min().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "validation_data_end": validation_data.index.max().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
    )

    return train_data, validation_data, split_info


def get_train_validation_split_by_percentage(
    df: pd.DataFrame,
    train_percentage: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Split a single patient's data into train and validation sets based on a percentage of the total days.

    This function separates a single patient's data into training and validation sets,
    where the validation set contains the specified percentage of the total days.

    IMPORTANT: This function requires the input DataFrame to have a DatetimeIndex for
    optimal performance and semantic correctness of time-series operations.

    Args:
        df (pd.DataFrame): Input dataframe for a SINGLE patient with DatetimeIndex (REQUIRED)
        train_percentage (float): Percentage of the total days to use for training. If 1, return the entire dataframe as train and None as validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]:
            (train_data, validation_data, split_info)
            where train_data and validation_data are DataFrames for the single patient

    Raises:
        ValueError: If DatetimeIndex is not found or insufficient data for requested validation period
        TypeError: If the DataFrame index is not a DatetimeIndex
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame must have a DatetimeIndex. "
            "Set your datetime column as index: df.set_index('datetime', inplace=True)"
        )
    if not (0 < train_percentage <= 1):
        raise ValueError(
            "train_percentage must be between 0 (exclusive) and 1 (inclusive)"
        )

    total_days = (df.index.max() - df.index.min()).days
    if total_days < 2:
        raise ValueError(
            f"Insufficient data span for percentage split: {total_days} day(s) for pid {df[ColumnNames.P_NUM.value].iloc[0]}"
        )

    train_days = int(total_days * train_percentage)
    # Ensure at least 1 day for both train and validation
    train_days = max(1, min(train_days, total_days - 1))
    num_validation_days = total_days - train_days

    if train_percentage == 1:
        train_df = df
        val_df = None
        info = {
            "train_days_actual": total_days,
            "validation_days_actual": 0,
        }
    else:
        # get_train_validation_split is only for internal use
        train_df, val_df, info = get_train_validation_split(
            df,
            num_validation_days=num_validation_days,
            day_start_hour=6,
            min_data_days=1,
            include_partial_days=False,
        )

    info.update(
        {
            "split_method": "percentage",
            "train_percentage": train_percentage,
            "total_days_estimated": total_days,
            "train_days_estimated": train_days,
            "validation_days_requested": num_validation_days,
        }
    )
    return train_df, val_df, info


def iter_patient_context_forecast_splits(
    patients_dict: dict,
    patient_ids: list | None,
    context_period: tuple[int, int] = (6, 24),
    forecast_horizon: tuple[int, int] = (0, 6),
) -> Generator[tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Get context and forecast periods for multi-patient dictionary.
    This function iterates over a dictionary of patient dataframes and yields context
    and forecast periods for each patient.
    Main purpose is to facilitate time-series forecasting tasks where we wish to
    forecast the same time period every day (e.g. nocturnal hypoglycemia prediction).

    Args:
        patients_dict (dict): Dictionary containing validation data for all patients
        patient_ids (list | None): List of patient IDs to get forecast periods for.
        context_period (tuple[int, int]): Start and end hours for input period (default: (6, 24))
        forecast_horizon (tuple[int, int]): Start and end hours for forecast period (default: (0, 6))

    Yields:
        tuple: (patient_id, context_period, forecast_horizon) where:
            - patient_id is the ID of the patient
            - context_period is the data from context_period[0] to context_period[1] fed into the model
            - forecast_horizon is the data from forecast_horizon[0] to forecast_horizon[1] of next day to predict

    Raises:
        ValueError: If patients_dict is None or patient not found.
    """
    if patients_dict is None:
        raise ValueError("patients_dict data is not available.")

    if not isinstance(patients_dict, dict):
        raise TypeError(f"Expected dict for patients_dict, got {type(patients_dict)}")

    if patient_ids is None:
        patient_ids = list(patients_dict.keys())

    for patient_id in patient_ids:
        if patient_id not in patients_dict:
            raise ValueError(
                f"Patient {patient_id} not found in patients_dict. Available patients: {list(patients_dict.keys())}"
            )

        patient_data = patients_dict[patient_id]
        for y_input_ts_period, y_test_period in iter_daily_context_forecast_splits(
            patient_data,
            context_period=context_period,
            forecast_horizon=forecast_horizon,
        ):
            yield patient_id, y_input_ts_period, y_test_period


def iter_daily_context_forecast_splits(
    patient_data: pd.DataFrame,
    context_period: tuple[int, int] = (6, 24),
    forecast_horizon: tuple[int, int] = (0, 6),
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Split each day's data into input period and forecast horizon based on configurable time periods.

    This function splits the data for a single patient by day into:
    - Input period: Historical context data fed into the model (default: 6am-12am)
    - Forecast horizon: Target period to predict (default: 12am-6am next day)

    Args:
        patient_data (pd.DataFrame): Data for a single patient with datetime index
        context_period (tuple[int, int]): Start and end hours for input period (default: (6, 24))
        forecast_horizon (tuple[int, int]): Start and end hours for forecast period (default: (0, 6))

    Yields:
        tuple[pd.DataFrame, pd.DataFrame]: (input_period, forecast_horizon) where:
            - input_period is the historical data from context_period fed into the model
            - forecast_horizon is the target data from forecast_horizon to predict

    Note:
        Only yields splits where both periods have data available.
        Hours are in 24-hour format (0-23). Use 24 for midnight of next day.
    """
    # Ensure data has datetime index
    if not isinstance(patient_data.index, pd.DatetimeIndex):
        if "datetime" in patient_data.columns:
            patient_data = patient_data.set_index("datetime")
        else:
            raise ValueError("Patient data must have datetime index or datetime column")

    # Ensure data is sorted by datetime index
    patient_data = patient_data.sort_index()

    # Get datetime index for date operations
    datetime_index = pd.to_datetime(patient_data.index)

    # Extract parameters
    context_start, context_end = context_period
    forecast_start, forecast_end = forecast_horizon

    # Validate parameters
    if not (0 <= context_start <= 24 and 0 <= context_end <= 24):
        raise ValueError("context_period hours must be between 0 and 24")
    if not (0 <= forecast_start <= 24 and 0 <= forecast_end <= 24):
        raise ValueError("forecast_horizon hours must be between 0 and 24")

    # Group by date
    for date_key, day_data in patient_data.groupby(datetime_index.date):
        # Convert date key to proper date object
        current_date = pd.to_datetime(date_key)
        # Get current day's input period data
        day_datetime_index = pd.to_datetime(day_data.index)

        # Handle context period logic
        if context_start < context_end and context_end < 24:
            # Context period is within the same day (e.g., 6-22)
            current_day_mask = (day_datetime_index.hour >= context_start) & (
                day_datetime_index.hour < context_end
            )
            current_day_data = day_data[current_day_mask]
        elif context_end == 24:
            # Context period goes to midnight (e.g., 6-24)
            current_day_mask = day_datetime_index.hour >= context_start
            current_day_data = day_data[current_day_mask]
        else:
            # Context period spans to next day (e.g., 22-6)
            # Get current day part (from context_start to end of day)
            current_day_mask = day_datetime_index.hour >= context_start
            current_day_part = day_data[current_day_mask]

            # Get next day part (from start of day to context_end)
            next_date = current_date + pd.Timedelta(days=1)
            next_day_mask = (datetime_index.date == next_date.date()) & (
                datetime_index.hour < context_end
            )
            next_day_part = patient_data[next_day_mask]

            # Combine both parts
            current_day_data = pd.concat([current_day_part, next_day_part])

        # Get forecast period data
        if forecast_start == 0 and forecast_end <= 24:
            # Forecast period starts at midnight of next day
            next_date = current_date + pd.Timedelta(days=1)

            # Create time range for forecast period on next day
            forecast_start_time = next_date.replace(
                hour=forecast_start, minute=0, second=0
            )
            forecast_end_time = next_date.replace(hour=forecast_end, minute=0, second=0)

            # Filter data within the time range
            forecast_data = patient_data[
                (patient_data.index >= forecast_start_time)
                & (patient_data.index < forecast_end_time)
            ]

        else:
            # Forecast period is within the same day or custom configuration
            if forecast_start < forecast_end:
                # Same day forecast
                forecast_mask = (day_datetime_index.hour >= forecast_start) & (
                    day_datetime_index.hour < forecast_end
                )
                forecast_data = day_data[forecast_mask]
            else:
                # Forecast spans to next day
                next_date = current_date + pd.Timedelta(days=1)

                # Current day part (from forecast_start to end of day)
                current_forecast_mask = day_datetime_index.hour >= forecast_start
                current_forecast_data = day_data[current_forecast_mask]

                # Next day part (from start of day to forecast_end)
                next_day_mask = (datetime_index.date == next_date.date()) & (
                    datetime_index.hour < forecast_end
                )
                next_forecast_data = patient_data[next_day_mask]

                # Combine both parts
                forecast_data = pd.concat([current_forecast_data, next_forecast_data])

        # Only yield if both periods have data
        if len(forecast_data) > 0 and len(current_day_data) > 0:
            yield current_day_data, forecast_data
