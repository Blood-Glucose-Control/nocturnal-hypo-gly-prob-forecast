"""
Kaggle Bristol T1D Dataset Loader.

This module provides functionality to load and process the Bristol Type 1 Diabetes
dataset from Kaggle. It handles both train and test datasets, with options for caching
processed data to avoid redundant computations.

The module supports data preprocessing including time interval regularization,
carbohydrate on board (COB) calculation, insulin on board (IOB) calculation,
and train/validation splitting.
"""

import pandas as pd
from src.data.preprocessing.time_processing import (
    create_datetime_index,
)
from src.data.preprocessing.sampling import (
    ensure_regular_time_intervals,
)
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)

from src.data.preprocessing.time_processing import get_train_validation_split
from src.data.datasets.dataset_base import DatasetBase
import os

from src.data.datasets.kaggle_bris_t1d.data_cleaner import (
    clean_brist1d_train_data,
    clean_brist1d_test_data,
)
from collections import defaultdict


class BrisT1DDataLoader(DatasetBase):
    """
    Data loader for the Bristol T1D diabetes dataset from Kaggle.

    This class handles loading, processing, and caching of the Bristol T1D dataset.
    It supports both train and test datasets with different processing pipelines
    for each. The train data is stored as a DataFrame, while test data is organized
    as a nested dictionary by patient ID and row ID.

    Attributes:
        keep_columns (list[str] | None): Specific columns to load from the dataset
        num_validation_days (int): Number of days to use for validation
        default_path (str): Default path to the raw dataset file
        cached_path (str): Path to the cached processed dataset
        use_cached (bool): Whether to use cached data if available
        dataset_type (str): Type of dataset ('train' or 'test')
        file_path (str): Path to the file being used (raw or cached)
        processed_data (pd.DataFrame | dict): The processed dataset
        train_data (pd.DataFrame): Training subset (when dataset_type is 'train')
        validation_data (pd.DataFrame): Validation subset (when dataset_type is 'train')
    """

    def __init__(
        self,
        keep_columns: list[str] | None = None,
        num_validation_days: int = 20,
        file_path: str | None = None,
        use_cached: bool = True,
        dataset_type: str = "train",
    ):
        """
        Initialize the Bristol T1D data loader.

        Args:
            keep_columns (list[str] | None, optional): Specific columns to load from the dataset.
                Defaults to None, which loads all columns.
            num_validation_days (int, optional): Number of days to use for validation.
                Defaults to 20.
            file_path (str | None, optional): Custom file path for the dataset.
                Defaults to None, which uses the default path.
            use_cached (bool, optional): Whether to use cached processed data if available.
                Defaults to True.
            dataset_type (str, optional): Type of dataset to load ('train' or 'test').
                Defaults to "train".
        """
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        # TODO: Raw train.csv is quite large. Need to download from kaggle
        # Set this to the cached path for now as it is already processed
        # self.default_path = os.path.join(
        #     os.path.dirname(__file__), f"{dataset_type}.csv"
        self.default_path = os.path.join(
            os.path.dirname(__file__), f"raw/{dataset_type}.csv"
        )
        self.cached_path = os.path.join(
            os.path.dirname(__file__),
            f"processed/{dataset_type}_cached{'.csv' if dataset_type == 'train' else ''}",
        )
        self.use_cached = use_cached
        self.dataset_type = dataset_type
        self.file_path = self.default_path
        if file_path is not None:
            self.file_path = file_path
        elif use_cached and os.path.exists(self.cached_path):
            self.file_path = self.cached_path

        # Preload data
        self.load_data()

    @property
    def dataset_name(self):
        """
        Return the name of the dataset.

        Returns:
            str: Name of the dataset
        """
        return "kaggle_brisT1D"

    def load_raw(self):
        """
        Load the raw dataset from CSV file.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.
        """
        # Return all columns
        return pd.read_csv(self.file_path, low_memory=False)

    def load_data(self):
        """
        Load and process the dataset, with caching support.

        This method handles loading the data either from cache or by processing
        the raw dataset. For train data, it returns a DataFrame and splits it into
        training and validation sets. For test data, it organizes the data into a
        nested dictionary by patient ID and row ID.

        If using cached data, it reads from the cache location. Otherwise, it processes
        the raw data and saves it to cache for future use.

        Returns:
            pd.DataFrame or dict[str, dict[str, pd.DataFrame]]: The loaded and processed data.
            For train data, returns a DataFrame. For test data, returns a nested dict.
        """
        if self.use_cached and os.path.exists(self.cached_path):
            if self.dataset_type == "train":
                self.processed_data = pd.read_csv(
                    self.cached_path, usecols=self.keep_columns
                )
            elif self.dataset_type == "test":
                self.processed_data = defaultdict(dict)

                if not os.path.isdir(self.cached_path):
                    raise NotADirectoryError(
                        f"Cache path '{self.cached_path}' is not a directory."
                    )

                for pid in os.listdir(self.cached_path):
                    patient_dir = os.path.join(self.cached_path, pid)
                    if not os.path.isdir(patient_dir):
                        raise NotADirectoryError(
                            f"Expected a directory for patient '{pid}', but got something else: {patient_dir}"
                        )

                    for filename in os.listdir(patient_dir):
                        if filename.endswith(".csv"):
                            file_path = os.path.join(patient_dir, filename)
                            df = pd.read_csv(file_path)

                            row_id = filename.replace(".csv", "")
                            self.processed_data[pid][row_id] = df
        else:
            # Not using cache or cache does not exist, process raw data and save
            self.raw_data = self.load_raw()
            self.processed_data = self._process_raw_data()

        if self.dataset_type == "train":
            # Make sure processed_data is a DataFrame before splitting
            if isinstance(self.processed_data, pd.DataFrame):
                # Split data into train and validation
                self.train_data, self.validation_data = get_train_validation_split(
                    self.processed_data, num_validation_days=self.num_validation_days
                )
            else:
                raise TypeError(
                    f"Expected processed_data to be a DataFrame for train dataset_type, but got {type(self.processed_data)}"
                )

    def _process_raw_data(self) -> pd.DataFrame | dict[str, dict[str, pd.DataFrame]]:
        """
        Process the raw data according to dataset type.

        For training data, this applies a series of preprocessing steps to the entire dataset.
        For test data, it processes each patient and row separately, organizing the result
        in a nested dictionary structure.

        The preprocessing steps include:
        - Cleaning the data
        - Creating datetime index
        - Ensuring regular time intervals
        - Calculating carbohydrates on board and availability
        - Calculating insulin on board and availability

        Returns:
            pd.DataFrame | dict[str, dict[str, pd.DataFrame]]: The processed data.
            For train data, returns a DataFrame. For test data, returns a nested dict.

        Raises:
            ValueError: If dataset_type is not 'train' or 'test'.
        """
        # Not cached, process the raw data
        if self.dataset_type == "train":
            data = clean_brist1d_train_data(self.raw_data)
            data = create_datetime_index(data)
            data = ensure_regular_time_intervals(data)
            data = create_cob_and_carb_availability_cols(data)
            data = create_iob_and_ins_availability_cols(data)

            # Save processed data to cache
            data.to_csv(self.cached_path, index=True)

            return data
        elif self.dataset_type == "test":  # test data: {pid : {rowid: df }}
            data = clean_brist1d_test_data(self.raw_data)
            processed_data = defaultdict(dict)

            # Setup cache dir
            os.makedirs(self.cached_path, exist_ok=True)

            for pid in data:
                # Make new dir for each patient
                patient_cache_dir = os.path.join(self.cached_path, pid)
                os.makedirs(patient_cache_dir, exist_ok=True)

                for row_id in data[pid]:
                    row_df = data[pid][row_id]
                    row_df = create_datetime_index(row_df)
                    row_df = ensure_regular_time_intervals(row_df)
                    row_df = create_cob_and_carb_availability_cols(row_df)
                    row_df = create_iob_and_ins_availability_cols(row_df)

                    cache_file = os.path.join(patient_cache_dir, f"{row_id}.csv")
                    row_df.to_csv(cache_file, index=True)

                    processed_data[pid][row_id] = row_df

            return processed_data
        else:
            raise ValueError(
                f"Unknown dataset_type: {self.dataset_type}. Must be 'train' or 'test'."
            )

    # TODO: MOVE THIS TO THE time_processing.py improve the name to be more clear what this function does
    # This function is used to split the validation data by day for each patient
    # It yields tuples of (patient_id, y_input_ts_period, y_test_period) for each day
    # where:
    # - y_input_ts_period is the data from 6am to 12am of a day
    # - y_test_period is the data from 12am to 6am of the next day
    def get_validation_day_splits(self, patient_id: str):
        """
        Get day splits for validation data for a single patient.

        This function splits the validation data by day for the specified patient,
        yielding each split as a tuple.

        Args:
            patient_id (str): ID of the patient to get validation splits for

        Yields:
            tuple: (patient_id, y_input_ts_period, y_test_period) where:
                - patient_id is the ID of the patient
                - y_input_ts_period is the data from 6am to 12am of a day, fed into the model
                - y_test_period is the data from 12am to 6am of the next day, ground truth to compare y_pred to.
        """
        patient_data = self.validation_data[self.validation_data["p_num"] == patient_id]
        for y_input_ts_period, y_test_period in self._get_day_splits(patient_data):
            yield patient_id, y_input_ts_period, y_test_period

    # TODO: MOVE THIS TO THE preprocessing.time_processing.py instead of splitter.py (fns were migrated)
    # TODO: change this function to be more general, so it can be used for both train and validation data
    # TODO: Change this function so that you cna specify the training and test periods lengths e.g. 6 hours, 8 hours, etc.
    def _get_day_splits(self, patient_data: pd.DataFrame):
        """
        Split each day's data into training period (6am-12am) and test period (12am-6am next day).

        This function splits the data for a single day into:
        - Training period: 6am to midnight of the current day
        - Test period: midnight to 6am of the next day

        Args:
            patient_data (pd.DataFrame): Data for a single patient

        Yields:
            tuple: (train_period, test_period) where:
                - train_period is the data from 6am to 12am of a day
                - test_period is the data from 12am to 6am of the next day
        """
        patient_data.loc[:, "datetime"] = pd.to_datetime(patient_data["datetime"])

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
