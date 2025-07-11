"""
Data cleaning utilities for the Kaggle Bristol Type 1 Diabetes dataset.

This module provides functions to clean and preprocess the Bristol T1D dataset by
handling time-series data, removing historical data points, and restructuring the
dataframes into more usable formats for analysis and modeling.
"""

import pandas as pd
from src.utils.kaggle_util import create_time_variable_lists
from collections import defaultdict


def clean_brist1d_test_data(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Clean and transform test data from the Bristol T1D dataset.

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
    columns_to_drop = [
        col
        for col in data.columns
        if any(prefix in col for prefix in prefixes_to_check)
        and not col.endswith("-0:00")
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    return data
