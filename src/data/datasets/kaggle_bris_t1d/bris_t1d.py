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
import functools
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

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
        train_dt_col_type (type): Data type of the datetime column in training data
        val_dt_col_type (type): Data type of the datetime column in validation data
        num_train_days (int): Number of unique days in the training dataset
        raw_data (pd.DataFrame): The original unprocessed dataset loaded from file
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
        self.raw_data_path = os.path.join(
            os.path.dirname(__file__), f"raw/{dataset_type}.csv"
        )
        self.cached_path = os.path.join(
            os.path.dirname(__file__),
            f"processed/{dataset_type}_cached{'.csv' if dataset_type == 'train' else ''}",
        )
        self.use_cached = use_cached
        self.dataset_type = dataset_type
        self.file_path = self.raw_data_path
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
        The raw dataset path is determined by the dataset type.

        This method reads the raw CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.
        """
        # Return all columns
        return pd.read_csv(self.file_path, low_memory=False)

    def load_data_old(self):
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
                self.keep_columns = self.processed_data.columns.tolist()
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

                self.train_dt_col_type = None
                self.val_dt_col_type = None
                self.num_train_days = None
                # self.num_validation_days = None
        else:
            # Not using cache or cache does not exist, process raw data and save
            self.raw_data = self.load_raw()
            self.processed_data = self._process_raw_data()

            if self.dataset_type == "train":
                # Make sure processed_data is a DataFrame before splitting
                if isinstance(self.processed_data, pd.DataFrame):
                    # Split data into train and validation
                    self.train_data, self.validation_data = get_train_validation_split(
                        self.processed_data,
                        num_validation_days=self.num_validation_days,
                    )
                    self.train_dt_col_type = self.train_data["datetime"].dtype
                    self.val_dt_col_type = self.validation_data["datetime"].dtype

                    self.num_train_days = len(
                        self.train_data["datetime"].dt.date.unique()
                    )
                else:
                    raise TypeError(
                        f"Expected processed_data to be a DataFrame for train dataset_type, but got {type(self.processed_data)}"
                    )

    def load_data(self):
        """
        Load and process the dataset, with caching support.

        This method handles loading the data either from cache or by processing
        the raw dataset. For train data, it returns a DataFrame and splits it into
        training and validation sets. For test data, it organizes the data into a
        nested dictionary by patient ID and row ID.

        If using cached data, it reads from the cache location. Otherwise, it processes
        the raw data and saves it to cache for future use.
        """
        if self.use_cached and os.path.exists(self.cached_path):
            self._load_from_cache()
        else:
            self._process_and_cache_data()

        # For train data, split into train and validation sets
        if self.dataset_type == "train" and isinstance(
            self.processed_data, pd.DataFrame
        ):
            self._split_train_validation()

    def _load_from_cache(self):
        """
        Load processed data from cache based on dataset type.

        This method handles different loading strategies depending on the dataset type:
        - For train data: Loads the cached CSV file and sets appropriate columns
        - For test data: Delegates to _load_test_data_from_cache() for handling the
        nested directory structure, then initializes timing metadata as None/zero
        since test data doesn't have these concepts

        The method respects the keep_columns attribute to load only specified columns
        when applicable.
        """
        if self.dataset_type == "train":
            self.processed_data = pd.read_csv(
                self.cached_path, usecols=self.keep_columns
            )
            self.keep_columns = self.processed_data.columns.tolist()
        elif self.dataset_type == "test":
            self._load_test_data_from_cache()

            # Initialize properties as None for cached test data
            self.train_dt_col_type = None
            self.val_dt_col_type = None
            self.num_train_days = 0
            self.num_validation_days = 0

    def _load_test_data_from_cache(self):
        """
        Load cached test data from directory structure.

        Test data is stored in a nested directory structure where:
        - The top level contains patient ID directories
        - Each patient directory contains CSV files named by row_id

        This method traverses this structure and loads each CSV into a nested
        dictionary organized as {patient_id: {row_id: DataFrame}}. This preserves
        the hierarchical organization of test data for later processing.

        Raises:
            NotADirectoryError: If the cache path does not exist or is not a directory,
                            or if a patient directory is not actually a directory
        """
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

    def _process_and_cache_data(self):
        """
        Process raw data and save to cache.

        This method orchestrates the workflow for data that isn't already cached:
        1. Load the raw data from the source file
        2. Process the raw data using the appropriate pipeline
        3. Store the processed data in self.processed_data

        The actual processing is delegated to _process_raw_data(), which handles
        different processing pipelines for train and test data.

        Note:
            The processed data is saved to cache within _process_raw_data(), not here.
        """
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()

    def _split_train_validation(self):
        """
        Split processed data into training and validation sets.

        This method is specific to train data and divides the processed dataset
        into training and validation subsets based on the specified number of
        validation days. After splitting, it calculates and stores metadata about
        the resulting datasets:

        - Data types of datetime columns
        - Number of unique days in the training dataset

        These metadata values are useful for later processing and analysis.

        Raises:
            TypeError: If processed_data is not a DataFrame, which is required for
                    the train/validation split operation.
        """
        if not isinstance(self.processed_data, pd.DataFrame):
            raise TypeError(
                f"Cannot split train/validation data: processed_data must be a DataFrame, but got {type(self.processed_data)}"
            )

        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=self.num_validation_days
        )
        self.train_dt_col_type = self.train_data["datetime"].dtype
        self.val_dt_col_type = self.validation_data["datetime"].dtype
        self.num_train_days = len(self.train_data["datetime"].dt.date.unique())

    def _process_raw_data_old(
        self,
    ) -> pd.DataFrame | dict[str, dict[str, pd.DataFrame]]:
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

            # TODO: I think this process should be vectorized in some way to improve performance.

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

        elif self.dataset_type == "test":
            data = clean_brist1d_test_data(self.raw_data)
            processed_data = defaultdict(dict)

            # Setup cache dir
            os.makedirs(self.cached_path, exist_ok=True)

            # Process each patient in parallel
            with ProcessPoolExecutor() as executor:
                # Create a partial function with common arguments
                process_patient_fn = functools.partial(
                    self._process_patient_data, base_cache_path=self.cached_path
                )

                # Map function to all patients and collect results
                results = executor.map(process_patient_fn, data.items())

                # Merge results into processed_data
                for pid, patient_data in results:
                    processed_data[pid] = patient_data

                return processed_data
        else:
            raise ValueError(
                f"Unknown dataset_type: {self.dataset_type}. Must be 'train' or 'test'."
            )

    def _process_patient_data(self, patient_item, base_cache_path):
        """Process data for a single patient."""
        pid, patient_data = patient_item
        processed_rows = {}

        # Make new dir for each patient
        patient_cache_dir = os.path.join(base_cache_path, pid)
        os.makedirs(patient_cache_dir, exist_ok=True)

        for row_id, row_df in patient_data.items():
            # Apply all transformations at once
            row_df = (
                row_df.pipe(create_datetime_index)
                .pipe(ensure_regular_time_intervals)
                .pipe(create_cob_and_carb_availability_cols)
                .pipe(create_iob_and_ins_availability_cols)
            )

            # Cache processed data
            cache_file = os.path.join(patient_cache_dir, f"{row_id}.csv")
            row_df.to_csv(cache_file, index=True)

            processed_rows[row_id] = row_df

        return pid, processed_rows

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
