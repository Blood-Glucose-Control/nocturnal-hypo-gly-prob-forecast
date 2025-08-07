"""
Time-based preprocessing utilities for continuous glucose monitoring data.

This module provides functions for handling, normalizing, and processing time-series data
specifically tailored for diabetes monitoring applications. It includes functionality for:

1. Creating datetime indices from time strings
2. Computing time differences between sequential measurements
3. Detecting the most common sampling interval in a dataset
4. Ensuring regular time intervals by filling missing timestamps
5. Splitting patient data into daily segments based on 6am transitions
6. Creating train-validation splits for time-series forecasting models

These utilities help prepare time-series data for analysis by establishing consistent
temporal representations across different patient datasets, handling time gaps, and
normalizing sampling frequencies. Day-based splitting allows for analyzing patterns
within clinically meaningful 24-hour periods (6am to 6am). The train-validation splitting
functionality supports machine learning model development with proper temporal separation
of training and validation data.

Functions:
    create_datetime_index: Create a proper datetime index from time strings
    _create_time_diff_cols: Add time difference columns between consecutive measurements
    get_most_common_time_interval: Determine the most frequent sampling interval
    ensure_regular_time_intervals: Normalize data to have consistent time intervals
    split_patient_data_by_day: Split patient data into daily segments based on 6am transitions
    get_train_validation_split: Create train and validation sets for time-series forecasting
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


# TODO: Remove the dependency of p_num. Kaggle data is the very few dataset where there are multiple patients in the same file.
# TODO: Remove hardcoded 6am.
def get_train_validation_split(
    df: pd.DataFrame,
    num_validation_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and validation sets based on complete days.

    This function separates data into training and validation sets, where the validation set
    contains the specified number of complete days. A day is defined as starting at the
    specified hour (default: 6am) and ending just before the same hour the next day.

    Args:
        df (pd.DataFrame): Input dataframe with datetime column
        num_validation_days (int): Number of complete days to use for validation
        day_start_hour (int): Hour that defines the start of a day (default: 6 for 6am)
        patient_col (str): Column name containing patient identifiers (default: "p_num")

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_data, validation_data) where validation
            contains exactly num_validation_days of complete days
    """
    # Check if datetime is the index or a column
    if "datetime" not in df.columns:
        if df.index.name == "datetime" or isinstance(df.index, pd.DatetimeIndex):
            # datetime is the index, reset it to a column
            df = df.reset_index()
        else:
            raise ValueError(
                "datetime column not found in data. Please run create_datetime_index first."
            )

    df["datetime"] = pd.to_datetime(df["datetime"])

    # For each patient:
    # 1. Find the last 6am timestamp
    # 2. Go back num_validation_days to get the start of validation
    # 3. Trim any data after the last 6am
    validation_data_list = []
    train_data_list = []

    for _, patient_df in df.groupby("p_num"):
        # Get timestamps where hour is 6 (6am)
        six_am_times = patient_df[patient_df["datetime"].dt.hour == 6]["datetime"]

        if len(six_am_times) == 0:
            continue

        # Get the last 6am timestamp
        last_six_am = six_am_times.max()

        # Calculate the start of validation period (num_validation_days before last_six_am)
        validation_start = last_six_am - pd.Timedelta(days=num_validation_days)

        # Split the patient's data
        patient_validation = patient_df[
            (patient_df["datetime"] >= validation_start)
            & (patient_df["datetime"] <= last_six_am)
        ]
        patient_train = patient_df[patient_df["datetime"] < validation_start]

        validation_data_list.append(patient_validation)
        train_data_list.append(patient_train)

    # Combine all patients' data
    validation_data = pd.concat(validation_data_list)
    train_data = pd.concat(train_data_list)

    # Check if conversion is needed (concatenation may lose dtype)
    if not pd.api.types.is_datetime64_any_dtype(validation_data["datetime"]):
        validation_data["datetime"] = pd.to_datetime(validation_data["datetime"])
    if not pd.api.types.is_datetime64_any_dtype(train_data["datetime"]):
        train_data["datetime"] = pd.to_datetime(train_data["datetime"])

    return train_data, validation_data


def ensure_datetime_index(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensures DataFrame has a datetime index.

    Args:
        data (pd.DataFrame): Input DataFrame that either has a datetime index or a 'date' column
        that can be converted to datetime.

    Returns:
        pd.DataFrame: DataFrame with sorted datetime index.
    """
    df = data.copy()

    # Check if the index is already a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # If not, set 'date' column as index and convert to DatetimeIndex
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            raise KeyError(
                "DataFrame must have either a 'date' column or a DatetimeIndex."
            )

    df.index = pd.DatetimeIndex(df.index)

    return df