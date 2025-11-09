# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Data cleaning utilities for the Kaggle Bristol Type 1 Diabetes dataset.

This module provides functions to clean and preprocess the Bristol T1D dataset by
handling time-series data, removing historical data points, and restructuring the
dataframes into more usable formats for analysis and modeling.
"""

import logging
from collections import defaultdict
from pathlib import Path

import os
import pandas as pd

from src.data.cache_manager import get_cache_manager
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)
from src.data.preprocessing.pipeline import preprocessing_pipeline
from src.data.preprocessing.sampling import (
    ensure_regular_time_intervals,
)
from src.utils.kaggle_util import create_time_variable_lists

logger = logging.getLogger(__name__)


def clean_brist1d_test_data(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Clean and transform test data from the Bristol T1D dataset (from offset to regular time intervals).

    This function processes the test dataset by:
    1. Melting the time-series data into a more structured format
    2. Extracting measurement types (bg, insulin, carbs, etc.)
    3. Converting time fields and adjusting timestamps
    4. Organizing data by patient ID and row ID

    Args:
        df: Raw test dataframe containing Bristol T1D data

    Returns:
        A nested dictionary with structure {patient_id: {row_id: dataframe}}
        where each dataframe contains the processed measurements for a specific
        patient and time point.
    """
    patient_ids = df["p_num"].unique().tolist()
    all_value_var_lists = create_time_variable_lists()
    patient_dfs = defaultdict(dict)
    for patient_id in patient_ids:
        for _, row in df[df["p_num"] == patient_id].iterrows():
            row_df = pd.DataFrame([row])  # Convert single row to DataFrame
            df_list = []
            for val_var in all_value_var_lists:
                temp_df = pd.melt(
                    row_df, id_vars=["id", "p_num", "time"], value_vars=val_var
                )
                temp_df = temp_df.rename(
                    columns={
                        "variable": val_var[0][:-4] + "time",
                        "value": val_var[0][:-4] + "0:00",
                    }
                )
                df_list.append(temp_df)

            bg_df = df_list[0]
            insulin_df = df_list[1]
            carbs_df = df_list[2]
            hr_df = df_list[3]
            steps_df = df_list[4]
            cals_df = df_list[5]
            activity_df = df_list[6]

            new_df = pd.concat(
                [
                    bg_df,
                    insulin_df.iloc[:, -1:],
                    carbs_df.iloc[:, -1:],
                    hr_df.iloc[:, -1:],
                    steps_df.iloc[:, -1:],
                    cals_df.iloc[:, -1:],
                    activity_df.iloc[:, -1:],
                ],
                axis=1,
            )

            # Convert time to datetime
            new_df["time"] = pd.to_datetime(new_df["time"])

            # Extract hours and minutes separately
            time_parts = new_df["bg-time"].str.extract(r"bg-(\d+):(\d+)")

            hours = pd.to_timedelta(time_parts[0].astype(int), unit="h")
            minutes = pd.to_timedelta(time_parts[1].astype(int), unit="m")
            total_hours = hours.add(minutes)  # Using .add() method instead of +

            # Subtract offset from time and format to HH:MM:SS
            new_df["time"] = (new_df["time"] - total_hours).dt.strftime("%H:%M:%S")

            row_id = new_df["id"].iloc[0]
            # Drop the bg-time column
            new_df = new_df.drop("bg-time", axis=1)
            new_df = new_df.drop("p_num", axis=1)
            new_df = new_df.drop("id", axis=1)

            new_df["p_num"] = patient_id

            new_df = new_df.rename(
                columns={
                    "bg-0:00": "bg_mM",
                    "insulin-0:00": "dose_units",
                    "carbs-0:00": "food_g",
                    "hr-0:00": "hr_bpm",
                    "steps-0:00": "steps",
                    "cals-0:00": "cals",
                    "activity-0:00": "activity",
                    "time": "datetime",
                }
            )

            patient_dfs[patient_id][row_id] = new_df

    return patient_dfs


def clean_brist1d_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Bristol T1D training dataset by removing historical data columns.

    This function processes the training dataset by:
    1. Identifying columns containing historical measurements
       (e.g., bg-5:55, activity-3:30)
    2. Keeping only the current measurements (those with -0:00 suffix)
    3. Preserving all non-measurement columns

    Args:
        df: Raw training dataframe from the Bristol T1D dataset

    Returns:
        A cleaned DataFrame containing only current measurements and
        metadata columns, with all historical measurements removed.
    """
    prefixes_to_check = ["activity", "bg", "cals", "insulin", "steps", "carbs", "hr"]

    # Create a copy to avoid modifying the original
    data = df.copy()

    # Create the list of columns to drop
    # Only keep the current measurements (those with -0:00 suffix)
    columns_to_drop = [
        col
        for col in data.columns
        if any(prefix in col for prefix in prefixes_to_check)
        and not col.endswith("-0:00")
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    # datetime
    # id
    # p_num
    # time
    # bg-0:00 (already in mmol/L)
    # insulin-0:00
    # carbs-0:00
    # hr-0:00
    # steps-0:00
    # cals-0:00
    # activity-0:00
    # Rename the columns to the correct format
    data = data.rename(
        columns={
            "bg-0:00": "bg_mM",
            "insulin-0:00": "dose_units",
            "carbs-0:00": "food_g",
            "hr-0:00": "hr_bpm",
            "steps-0:00": "steps",
            "cals-0:00": "cals",
            "activity-0:00": "activity",
            "time": "datetime",
        }
    )

    # Add addition columns: [hr_bpm, steps, message_type]
    # Set default message type to "bg"
    # For rows with food_g values, add ANNOUNCE_MEAL to msg_type
    data.loc[data["food_g"].notna() & (data["food_g"] != 0), "msg_type"] = (
        "ANNOUNCE_MEAL"
    )

    # TODO:TONY - Remove this
    cache_manager = get_cache_manager()
    cache_manager.get_cleaning_step_data_path("kaggle_brisT1D")
    dir_path = (
        cache_manager.get_cleaning_step_data_path("kaggle_brisT1D")
        / "data_clean/data_after_cleaning.csv"
    )
    os.makedirs(dir_path.parent, exist_ok=True)
    data.to_csv(
        dir_path,
        index=False,
    )
    return data


def create_datetime_with_rollover_detection(
    time_series: pd.Series, patient_start_date: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    """
    Simple version that detects day rollovers and increments date accordingly.

    Args:
        time_series (pd.Series): Series of times as strings ("%H:%M:%S")
        patient_start_date (pd.Timestamp): The date to start from

    Returns:
        tuple[pd.Series, pd.Series]: (date_series, time_series) where:
            - date_series: Series of dates (datetime64[ns] with date info)
            - time_series: Series of times (time objects)
    """
    if len(time_series) == 0:
        return pd.Series([], dtype="datetime64[ns]"), pd.Series([], dtype="time")

    # Clean and convert to time objects
    time_series = time_series.str.strip()
    times = pd.to_datetime(time_series, format="%H:%M:%S", errors="coerce").dt.time
    result_dates = []
    current_date = patient_start_date.date()  # Get just the date part!
    one_day = pd.Timedelta(days=1)
    for i, current_time in enumerate(times):
        # Check for day rollover (current hour:minute < previous hour:minute)
        if i > 0:  # and not pd.isna(times.iloc[i - 1]):
            prev_time = times.iloc[i - 1]

            # Simple comparison: if current time < previous time, we rolled over
            current_minutes = current_time.hour * 60 + current_time.minute
            prev_minutes = prev_time.hour * 60 + prev_time.minute

            if current_minutes < prev_minutes:
                current_date += one_day

        # Combine current date with current time
        result_dates.append(current_date)

    result_dates = pd.Series(result_dates, dtype="datetime64[ns]")
    logger.info("create_datetime_with_rollover_detection(): ")
    logger.info(f"\tcreate_dt_col() - Result start date: {result_dates.iloc[0]}")
    logger.info(f"\tcreate_dt_col() - Result end date: {result_dates.iloc[-1]}")
    logger.info(f"\tcreate_dt_col() - Result start time: {times.iloc[0]}")
    logger.info(f"\tcreate_dt_col() - Result end time: {times.iloc[-1]}")
    logger.info(
        f"\tcreate_dt_col() - Any NaT values in dates? {result_dates.isna().sum()}"
    )
    logger.info(f"\tcreate_dt_col() - Any NaT values in times? {times.isna().sum()}")

    return (result_dates, times)


def process_single_patient_data(
    patient_data_tuple: tuple, store_in_between_data=False
) -> tuple:
    """
    Process a single patient's data including datetime creation and preprocessing.

    Note: Standalone functions were created to support multiprocessing, as corruption issues
    occur when using class methods with ProcessPoolExecutor.

    Args:
        patient_data_tuple (tuple): Tuple containing (p_num, data, generic_patient_start_date)
            where p_num is patient ID, data is DataFrame, and generic_patient_start_date
            is the Timestamp to use as the starting date.
        store_in_between_data (bool, optional): Whether to save intermediate data to cache.
            Defaults to False.

    Returns:
        tuple: Tuple containing (p_num, processed_data) where p_num is the patient ID
            and processed_data is the DataFrame after preprocessing pipeline.
    """
    p_num, data, generic_patient_start_date = patient_data_tuple
    logger.info(
        f"Running process_single_patient_data(), \n\t\t\tProcessing patient {p_num} data...\n\t\t\tPatient start date: {generic_patient_start_date.date()}"
    )
    logger.info(f"\tInputed patient start time: {data['datetime'].iloc[0]}")
    # Create a copy to avoid modifying the original
    data_copy = data.copy()

    # Create datetime column
    patient_dates, patient_times = create_datetime_with_rollover_detection(
        data_copy["datetime"], generic_patient_start_date
    )
    logger.info(
        f"\tCreated columns for patient {p_num} data...\n\t\t\tPatient start date: {patient_dates.iloc[0]}"
    )
    logger.info(f"\tResult patient start time: {patient_times.iloc[0]}")
    logger.info(f"\tLength of dates: {len(patient_dates)}")
    logger.info(f"\tLength of times: {len(patient_times)}")

    data_copy["datetime"] = date_time_zipper(patient_dates, patient_times)

    # Convert datetime column to index
    data_copy = data_copy.set_index("datetime", drop=True)
    if store_in_between_data:
        cache_manager = get_cache_manager()
        cache_manager.get_cleaning_step_data_path("kaggle_brisT1D")
        dir_path = (
            cache_manager.get_cleaning_step_data_path("kaggle_brisT1D")
            / "datetime_index"
        )
        os.makedirs(dir_path, exist_ok=True)
        data_copy.to_csv(
            dir_path / f"{p_num}.csv",
            index=True,
        )
    # Run preprocessing pipeline
    processed_data = preprocessing_pipeline(p_num, data_copy)

    return p_num, processed_data


def process_patient_prediction_instances(
    patient_item,
    base_cache_path: Path,
    generic_patient_start_date=pd.Timestamp("2024-01-01"),
    save_individual_files=False,
):
    """
    Standalone function to process test data for a single patient in parallel.

    Applies transformations (datetime index, regular intervals, COB/IOB calculations)
    to each row of patient data. Returns processed data in memory rather than saving
    individual files, as the entire nested structure will be saved as compressed pickle.

    Args:
        patient_item (tuple): (patient_id, patient_data_dict) mapping row_ids to DataFrames
        base_cache_path (Path): Base directory for caching processed data
        generic_patient_start_date (pd.Timestamp, optional): Starting date to use for
            datetime creation. Defaults to pd.Timestamp("2024-01-01").

    Returns:
        tuple: (patient_id, processed_rows_dict) mapping row_ids to processed DataFrames
    """
    pid, patient_data = patient_item
    processed_rows = {}

    if save_individual_files:
        # Make new dir for each patient
        patient_cache_dir: Path = base_cache_path / pid
        patient_cache_dir.mkdir(exist_ok=True)

    for row_id, row_df in patient_data.items():
        logger.info(f"Processing patient {pid}, row {row_id}...")

        # Create a copy to avoid modifying the original
        row_df_copy = row_df.copy()

        # Create datetime column using the same approach as training data
        patient_dates, patient_times = create_datetime_with_rollover_detection(
            row_df_copy["datetime"], generic_patient_start_date
        )

        row_df_copy["datetime"] = date_time_zipper(patient_dates, patient_times)

        # Convert datetime column to index
        row_df_copy = row_df_copy.set_index("datetime", drop=True)

        # Get regular intervals and frequency
        row_df_copy, freq = ensure_regular_time_intervals(row_df_copy)

        # Apply COB and IOB calculations with frequency
        row_df_copy = row_df_copy.pipe(
            create_cob_and_carb_availability_cols, freq
        ).pipe(create_iob_and_ins_availability_cols, freq)

        if save_individual_files:
            # Cache processed data - ensure we're working with Path objects
            cache_file = Path(patient_cache_dir) / f"{row_id}.csv"
            row_df_copy.to_csv(str(cache_file), index=True)

        processed_rows[row_id] = row_df_copy

    return pid, processed_rows


def date_time_zipper(patient_dates, patient_times):
    """
    Combine separate date and time series into a single datetime series.
    Args:
        patient_dates (pd.Series): Series of dates (datetime64[ns] with date info)
        patient_times (pd.Series): Series of times (time objects)
    Returns:
        zipped_date_time_list: List of combined datetime (datetime64[ns])
    """
    zipped_date_time_list = [
        pd.Timestamp.combine(date, time)
        if pd.notna(date) and pd.notna(time)
        else pd.NaT
        for date, time in zip(patient_dates, patient_times)
    ]

    return zipped_date_time_list
