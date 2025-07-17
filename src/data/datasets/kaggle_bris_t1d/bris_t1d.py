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
from src.data.cache_manager import get_cache_manager
from src.data.dataset_configs import get_dataset_config

from src.data.datasets.kaggle_bris_t1d.data_cleaner import (
    clean_brist1d_train_data,
    clean_brist1d_test_data,
)

import logging

logger = logging.getLogger(__name__)


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
        use_cached (bool): Whether to use cached "processed" data if available
        dataset_type (str): Type of dataset ('train' or 'test')
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
            use_cached (bool, optional): Whether to use cached processed data if available.
                Defaults to True.
            dataset_type (str, optional): Type of dataset to load ('train' or 'test').
                Defaults to "train".
        """
        # Ensure 'datetime' is included in keep_columns if specified
        if keep_columns is not None:
            if "datetime" not in keep_columns:
                keep_columns = keep_columns + ["datetime"]
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.use_cached = use_cached
        self.dataset_type = dataset_type

        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        self.dataset_config = get_dataset_config(self.dataset_name)

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
        Load the raw dataset from cache or fetch from source.
        The raw dataset is automatically fetched from Kaggle if not available.

        This method ensures raw data is available in the cache, fetching it
        from Kaggle if necessary, then reads the appropriate CSV file.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.
        """
        # Ensure raw data is available
        raw_data_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )

        # Load the appropriate file based on dataset type
        if self.dataset_type == "train":
            file_path = raw_data_path / "train.csv"
        elif self.dataset_type == "test":
            file_path = raw_data_path / "test.csv"
        else:
            raise ValueError(
                f"Unknown dataset_type: {self.dataset_type}. Must be 'train' or 'test'."
            )

        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

        # Return all columns
        return pd.read_csv(file_path, low_memory=False)

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
        need_to_process_data = True
        if self.use_cached:
            cached_data = self.cache_manager.load_processed_data(
                self.dataset_name, self.dataset_type
            )
            if cached_data is not None:
                # This sets the data given the dataset type (train or test)
                self._load_from_cache(cached_data)
                need_to_process_data = False

        if self.dataset_type == "train" and isinstance(
            self.processed_data, pd.DataFrame
        ):
            self._split_train_validation()

        if need_to_process_data:
            logger.info(
                "Processed cache not found or not used, processing raw data and saving to cache..."
            )
            # This will attempt to load the raw data (and fetch from kaggle if not available), process it and save it to cache
            self._process_and_cache_data()

    def _load_from_cache(self, cached_data):
        """
        Load processed data from cache to self.processed_data.

        Args:
            cached_data: The cached data loaded from the cache manager
        """
        if self.dataset_type == "train":
            self.processed_data = cached_data
            if self.keep_columns:
                # Check if datetime is in keep_columns but is actually the index
                if (
                    "datetime" in self.keep_columns
                    and "datetime" not in self.processed_data.columns
                ):
                    # datetime is the index, so we need to reset it to a column first
                    self.processed_data = self.processed_data.reset_index()

                # Now filter by keep_columns
                self.processed_data = self.processed_data[self.keep_columns]
                self.keep_columns = self.processed_data.columns.tolist()
        elif self.dataset_type == "test":
            # For test data, we need to load the nested structure from cache
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

        processed_path = self.cache_manager.get_processed_data_path_for_type(
            self.dataset_name, self.dataset_type
        )

        if not processed_path.exists():
            raise FileNotFoundError(
                f"Processed test data not found at: {processed_path}"
            )

        for pid in processed_path.iterdir():
            if not pid.is_dir():
                continue

            for csv_file in pid.glob("*.csv"):
                df = pd.read_csv(csv_file)
                row_id = csv_file.stem
                self.processed_data[pid.name][row_id] = df

    def _process_and_cache_data(self):
        """
        Process raw data and save to cache.

        This method orchestrates the workflow for data that isn't already cached:
        1. Load the raw data from the source file (prompt users to fetch from kaggle if not available. Cache is just another level of cache for the raw data)
        2. Process the raw data using the appropriate pipeline
        3. Store the processed data in self.processed_data
        4. Save the processed data to cache

        The actual processing is delegated to _process_raw_data(), which handles
        different processing pipelines for train and test data.
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

    def _process_raw_data(self) -> pd.DataFrame | dict[str, dict[str, pd.DataFrame]]:
        """
        Process the raw data according to dataset type with performance optimizations.

        For training data, this applies a series of preprocessing steps to the entire dataset.
        For test data, it processes each patient in parallel using a process pool,
        organizing the result in a nested dictionary structure.

        The preprocessing steps include:
        - Cleaning the data
        - Creating datetime index
        - Ensuring regular time intervals
        - Calculating carbohydrates on board and availability
        - Calculating insulin on board and availability

        Returns:
            pd.DataFrame | dict[str, dict[str, pd.DataFrame]]: The processed data.
            For train data, returns a DataFrame. For test data, returns a nested dict
            organized as {patient_id: {row_id: DataFrame}}.

        Raises:
            ValueError: If dataset_type is not 'train' or 'test'.
        """
        # Not cached, process the raw data
        if self.dataset_type == "train":
            logger.info("Processing train data. This may take a while...")
            data = clean_brist1d_train_data(self.raw_data)
            data = create_datetime_index(data)
            data = ensure_regular_time_intervals(data)
            data = create_cob_and_carb_availability_cols(data)
            data = create_iob_and_ins_availability_cols(data)

            # Save processed data to cache
            self.cache_manager.save_processed_data(
                self.dataset_name, self.dataset_type, data
            )

            return data

        elif self.dataset_type == "test":
            logger.info("Processing test data. This may take a while...")
            data = clean_brist1d_test_data(self.raw_data)
            processed_data = defaultdict(dict)

            processed_path = self.cache_manager.get_processed_data_path_for_type(
                self.dataset_name, self.dataset_type
            )
            processed_path.mkdir(parents=True, exist_ok=True)

            # Process each patient in parallel
            with ProcessPoolExecutor() as executor:
                # Create a partial function with common arguments
                process_patient_fn = functools.partial(
                    self._process_patient_data, base_cache_path=processed_path
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
        """
        Process data for a single patient in parallel.

        This function is designed to be called by ProcessPoolExecutor to enable
        parallel processing of patient data. It applies a sequence of transformations
        to each row of patient data using method chaining for efficiency.

        For each row:
        1. Creates datetime index
        2. Ensures regular time intervals
        3. Calculates COB and carb availability
        4. Calculates IOB and insulin availability
        5. Saves the processed row to cache

        Args:
            patient_item (tuple): A tuple of (patient_id, patient_data_dict) where
                patient_data_dict is a dictionary mapping row_ids to DataFrames
            base_cache_path (Path): Base directory path for caching processed data

        Returns:
            tuple: (patient_id, processed_rows_dict) where processed_rows_dict is a
                dictionary mapping row_ids to processed DataFrames
        """
        pid, patient_data = patient_item
        processed_rows = {}

        # Make new dir for each patient
        patient_cache_dir = base_cache_path / pid
        patient_cache_dir.mkdir(exist_ok=True)

        for row_id, row_df in patient_data.items():
            # Apply all transformations at once
            row_df = (
                row_df.pipe(create_datetime_index)
                .pipe(ensure_regular_time_intervals)
                .pipe(create_cob_and_carb_availability_cols)
                .pipe(create_iob_and_ins_availability_cols)
            )

            # Cache processed data
            cache_file = patient_cache_dir / f"{row_id}.csv"
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
