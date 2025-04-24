"""Data loading functions for the various datasets."""

import pandas as pd
from src.data.kaggle_brisT1D.data_loader import BrisT1DDataLoader
from src.data.gluroo.gluroo import Gluroo


def load_data(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    file_path: str = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    Get the data loader for the given data source name.

    Parameters:
        data_source_name (str): The name of the data source. Default is 'kaggle_brisT1D'.
        dataset_type (str): The type of the dataset, e.g., 'train' or 'test'. Default is 'train'.
        keep_columns (list): A list of column names to keep. If None, all columns are loaded.
            Default is None.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if data_source_name == "kaggle_brisT1D":
        return BrisT1DDataLoader(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            file_path=file_path,
            use_cached=use_cached,
            dataset_type=dataset_type,
        )
    elif data_source_name == "gluroo":
        return Gluroo(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            dataset_type=dataset_type,
            file_path=file_path,
            config=config,
        )
    else:
        raise ValueError("Invalid dataset_name or dataset_type")


# TODO: Remove the dependency of p_num. Kaggle data is the very few dataset where there are multiple patients in the same file.
def get_train_validation_split(
    df: pd.DataFrame,
    num_validation_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and validation sets based on complete days,
    where a day is defined as 6am-6am.

    Args:
        df (pd.DataFrame): Input dataframe with datetime column
        num_validation_days (int): Number of complete 6am-6am days to use for validation

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_data, validation_data) where validation
            contains exactly num_validation_days of complete 6am-6am days
    """
    if "datetime" not in df.columns:
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

    for patient_id, patient_df in df.groupby("p_num"):
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

    return train_data, validation_data
