"""Data loading functions for the various datasets."""

import os
import pandas as pd


def load_data(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list = None,
    use_cached: bool = False,
) -> pd.DataFrame:
    """
    Load RAW or CACHED data from a specified dataset and type, optionally selecting specific columns.
    Cached data is already processed and cleaned.

    Parameters:
    data_source_name (str): The name of the data source. Default is 'kaggle_brisT1D'.
    dataset_type (str): The type of the dataset, e.g., 'train' or 'test'. Default is 'train'.
    columns_to_keep (list): A list of column names to keep. If None, all columns are loaded.
        Default is None.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
    ValueError: If the specified file does not exist.
    """
    local_path = os.path.dirname(__file__)
    file_path = None
    if data_source_name == "kaggle_brisT1D":
        default_path = os.path.join(local_path, f"kaggle_brisT1D/{dataset_type}.csv")
        cached_path = os.path.join(local_path, "kaggle_brisT1D/train_cached.csv")
        file_path = (
            cached_path
            if use_cached and dataset_type == "train" and os.path.exists(cached_path)
            else default_path
        )
        if use_cached and dataset_type == "train" and not os.path.exists(cached_path):
            raise ValueError(
                f"Unable to find train_cached.csv at "
                f"\n {cached_path}. \n"
                f"Verify that the file exists if you want to use the cached version."
            )
    elif data_source_name == "gluroo":
        # Can't cache gluroo data because it's generated
        pass
    else:
        raise ValueError("Invalid dataset_name or dataset_type")

    if not os.path.exists(file_path):
        raise ValueError("Invalid dataset_name or dataset_type")

    return pd.read_csv(file_path, usecols=keep_columns, low_memory=False)


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


# TODO: Add a function that returns the loader class for a given data source name
