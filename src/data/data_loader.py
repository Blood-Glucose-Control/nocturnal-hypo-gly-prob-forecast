"""Data loading functions for the various datasets."""

import os
import pandas as pd
from src.data.data_cleaner import clean_data
from src.data.data_transforms import (
    create_datetime_index,
    create_cob_and_carb_availability_cols,
    create_iob_and_ins_availability_cols,
    ensure_regular_time_intervals
)


def load_data(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list = None,
    use_cached: bool = False,
) -> pd.DataFrame:
    """
    Load data from a specified dataset and type, optionally selecting specific columns.

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


class BrisT1DDataLoader:
    def __init__(
        self,
        keep_columns: list = None,
        use_cached: bool = True,
        num_validation_days: int = 20,
    ):
        self.keep_columns = keep_columns
        self.use_cached = use_cached
        self.raw_data = load_data(
            data_source_name="kaggle_brisT1D",
            dataset_type="train",
            keep_columns=keep_columns,
            use_cached=use_cached,
        )
        self.processed_data = self._process_raw_data()
        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=num_validation_days
        )

    def get_validation_day_splits(self, patient_id: str):
        """
        Get day splits for validation data for a single patient.

        Yields:
            tuple: (patient_id, train_period, test_period)
        """
        patient_data = self.validation_data[self.validation_data["p_num"] == patient_id]
        for train_period, test_period in self._get_day_splits(patient_data):
            yield patient_id, train_period, test_period

    def _process_raw_data(self) -> pd.DataFrame:
        if self.use_cached:
            self.processed_data = self.raw_data
            return self.processed_data
        # Not cached, process the raw data
        self.processed_data = clean_data(self.raw_data)
        self.processed_data = create_datetime_index(self.processed_data)
        self.processed_data = ensure_regular_time_intervals(self.processed_data)
        self.processed_data = create_cob_and_carb_availability_cols(self.processed_data)
        self.processed_data = create_iob_and_ins_availability_cols(self.processed_data)
        return self.processed_data

    def _get_day_splits(self, patient_data: pd.DataFrame):
        """
        Split each day's data into training period (6am-12am) and test period (12am-6am next day).

        Args:
            patient_data (pd.DataFrame): Data for a single patient

        Yields:
            tuple: (train_period, test_period) where:
                - train_period is the data from 6am to 12am of a day
                - test_period is the data from 12am to 6am of the next day
        """

        patient_data["datetime"] = pd.to_datetime(patient_data["datetime"])

        # Ensure data is sorted by datetime
        patient_data = patient_data.sort_values("datetime")

        # Group by date
        for date, day_data in patient_data.groupby(patient_data["datetime"].dt.date):
            # Get next day's early morning data (12am-6am)
            next_date = date + pd.Timedelta(days=1)
            next_day_data = patient_data[
                (patient_data["datetime"].dt.date == next_date)
                & (patient_data["datetime"].dt.hour < 6)
            ]

            # Get current day's data (6am-12am)
            current_day_data = day_data[day_data["datetime"].dt.hour >= 6]

            if len(next_day_data) > 0 and len(current_day_data) > 0:
                yield current_day_data, next_day_data
