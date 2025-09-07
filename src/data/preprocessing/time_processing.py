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
    get_train_validation_split: Create robust train and validation sets as patient-separated dictionaries with DatetimeIndex requirement for optimal performance
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


def get_train_validation_split(
    df: pd.DataFrame,
    num_validation_days: int = 20,
    day_start_hour: int = 6,
    patient_col: str = "p_num",
    min_data_days: int = 1,
    include_partial_days: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    """
    Split the data into train and validation sets based on complete days.

    This function separates data into training and validation sets, where the validation set
    contains the specified number of complete days. A day is defined as starting at the
    specified hour (default: 6am) and ending just before the same hour the next day.

    IMPORTANT: This function requires the input DataFrame to have a DatetimeIndex for
    optimal performance and semantic correctness of time-series operations.

    Args:
        df (pd.DataFrame): Input dataframe with DatetimeIndex (REQUIRED)
        num_validation_days (int): Number of complete days to use for validation
        day_start_hour (int): Hour that defines the start of a day (default: 6 for 6am)
        patient_col (str): Column name containing patient identifiers (default: "p_num")
        min_data_days (int): Minimum number of days of data required per patient (default: 1)
        include_partial_days (bool): Whether to include partial days at the end of data (default: False)

    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
            (train_data_dict, validation_data_dict, split_info)
            where train_data_dict and validation_data_dict are dictionaries with patient IDs as keys
            and individual patient DataFrames as values. This prevents accidental cross-patient
            operations and makes patient-specific processing more explicit.

    Raises:
        ValueError: If DatetimeIndex is not found, patient_col doesn't exist, or
                   insufficient data for requested validation period
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
    
    if patient_col not in df.columns:
        raise ValueError(f"Patient column '{patient_col}' not found in DataFrame")

    # Work with a copy to avoid modifying original
    df = df.copy()

    # Initialize split info dictionary
    split_info = {
        "total_patients": 0,
        "patients_included": 0,
        "patients_excluded": [],
        "exclusion_reasons": {},
        "validation_days_actual": {},
        "train_data_range": {},
        "validation_data_range": {},
    }

    # For each patient:
    # 1. Find day boundary timestamps based on day_start_hour
    # 2. Go back num_validation_days to get the start of validation
    # 3. Optionally include partial days
    train_data_dict = {}
    validation_data_dict = {}

    for patient_id, patient_df in df.groupby(patient_col):
        split_info["total_patients"] += 1

        # Get timestamps where hour matches day_start_hour (using efficient index operations)
        # We've already validated this is a DatetimeIndex in input validation
        day_boundary_mask = patient_df.index.hour == day_start_hour  # type: ignore
        day_boundary_times = patient_df.index[day_boundary_mask]

        # Check if patient has enough data
        if len(day_boundary_times) == 0:
            split_info["patients_excluded"].append(patient_id)
            split_info["exclusion_reasons"][patient_id] = (
                f"No {day_start_hour}:00 timestamps found"
            )
            continue

        # Calculate total days of data available (using efficient index operations)
        total_data_span = (patient_df.index.max() - patient_df.index.min()).days

        if total_data_span < min_data_days:
            split_info["patients_excluded"].append(patient_id)
            split_info["exclusion_reasons"][patient_id] = (
                f"Insufficient data: {total_data_span} days < {min_data_days} required"
            )
            continue

        # Get the last day boundary timestamp
        if include_partial_days:
            # Use the latest data point
            last_boundary = patient_df.index.max()
        else:
            # Use the last day_start_hour timestamp
            last_boundary = day_boundary_times.max()

        # Calculate the start of validation period
        validation_start = last_boundary - pd.Timedelta(days=num_validation_days)

        # Ensure we don't go before the start of available data
        data_start = patient_df.index.min()
        if validation_start < data_start:
            available_days = (last_boundary - data_start).days
            split_info["patients_excluded"].append(patient_id)
            split_info["exclusion_reasons"][patient_id] = (
                f"Insufficient data for {num_validation_days} validation days: only {available_days} days available"
            )
            continue

        # Split the patient's data using efficient index-based slicing
        if include_partial_days:
            # Include all data from validation_start onwards
            patient_validation = patient_df.loc[validation_start:]
            patient_train = patient_df.loc[:validation_start]
        else:
            # Use precise time range slicing
            patient_validation = patient_df.loc[validation_start:last_boundary]
            patient_train = patient_df.loc[:validation_start]

        # Only include if we have meaningful data
        if len(patient_validation) > 0 and len(patient_train) > 0:
            # Store each patient's data separately in dictionaries (make copies to avoid warnings)
            validation_data_dict[patient_id] = patient_validation.copy()
            train_data_dict[patient_id] = patient_train.copy()

            split_info["patients_included"] += 1
            actual_validation_days = (
                patient_validation.index.max() - patient_validation.index.min()
            ).days
            split_info["validation_days_actual"][patient_id] = actual_validation_days
            split_info["train_data_range"][patient_id] = (
                patient_train.index.min().strftime("%Y-%m-%d %H:%M:%S"),
                patient_train.index.max().strftime("%Y-%m-%d %H:%M:%S"),
            )
            split_info["validation_data_range"][patient_id] = (
                patient_validation.index.min().strftime("%Y-%m-%d %H:%M:%S"),
                patient_validation.index.max().strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            split_info["patients_excluded"].append(patient_id)
            split_info["exclusion_reasons"][patient_id] = (
                "Empty train or validation set after split"
            )

    # Check if we have any data to return
    if not validation_data_dict or not train_data_dict:
        raise ValueError(
            f"No patients met the criteria for train/validation split. "
            f"Excluded {split_info['total_patients']} patients. "
            f"Reasons: {split_info['exclusion_reasons']}"
        )

    # Since we're using DatetimeIndex, no need for datetime column conversion
    # The index is already properly typed as DatetimeIndex

    # Add summary statistics to split_info
    total_train_records = sum(len(df) for df in train_data_dict.values())
    total_validation_records = sum(len(df) for df in validation_data_dict.values())

    split_info["train_records"] = total_train_records
    split_info["validation_records"] = total_validation_records
    split_info["split_ratio"] = total_validation_records / (
        total_train_records + total_validation_records
    )

    return train_data_dict, validation_data_dict, split_info
