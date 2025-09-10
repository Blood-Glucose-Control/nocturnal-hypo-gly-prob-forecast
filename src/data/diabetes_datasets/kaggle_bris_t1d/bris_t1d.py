"""
Kaggle Bristol T1D Dataset Loader.

This module provides functionality to load and process the Bristol Type 1 Diabetes
dataset from Kaggle. It handles both train and test datasets, with options for caching
processed data to avoid redundant computations.

The module supports data preprocessing including time interval regularization,
carbohydrate on board (COB) calculation, insulin on board (IOB) calculation,
and train/validation splitting.
"""

import functools
import gzip
import logging
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from src.data.cache_manager import get_cache_manager
from src.data.dataset_configs import get_dataset_config
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.diabetes_datasets.kaggle_bris_t1d.data_cleaner import (
    clean_brist1d_test_data,
    clean_brist1d_train_data,
    process_patient_prediction_instances,
    process_single_patient_data,
)
from src.data.preprocessing.data_splitting import split_multipatient_dataframe
from src.data.preprocessing.time_processing import (
    get_train_validation_split,
)

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
        parallel: bool = True,
        generic_patient_start_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
        max_workers: int = 3,
    ):
        """
        Initialize the Bristol T1D data loader.

        Args:
            keep_columns (list[str] | None, optional): Specific columns to load from the dataset.
                If provided, 'datetime' column is automatically included if not already present.
                Defaults to None, which loads all columns.
            num_validation_days (int, optional): Number of days to use for validation.
                Only applies when dataset_type is 'train'. Defaults to 20.
            use_cached (bool, optional): Whether to use cached processed data if available.
                Defaults to True.
            dataset_type (str, optional): Type of dataset to load ('train' or 'test').
                Defaults to "train".
            parallel (bool, optional): Whether to use parallel processing.
                Defaults to True.
            generic_patient_start_date (pd.Timestamp, optional): Starting date to use for
                all patients when creating datetime columns. Defaults to "2024-01-01".
            max_workers (int, optional): Maximum number of workers for parallel processing.
                Defaults to 3.
        """
        # Ensure 'datetime' is included in keep_columns if specified
        if keep_columns is not None:
            if "datetime" not in keep_columns:
                keep_columns = keep_columns + ["datetime"]
        self.generic_patient_start_date = generic_patient_start_date
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.use_cached = use_cached
        self.dataset_type = dataset_type
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        self.dataset_config = get_dataset_config(self.dataset_name)

        # Data
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None

        # Metadata
        self.train_dt_col_type = None
        self.val_dt_col_type = None
        self.num_train_days = None

        # Preload data
        self.load_data()

    @property
    def dataset_name(self):
        return "kaggle_brisT1D"

    def load_data(self):
        """
        Load processed data from cache or process raw data and save to cache.
        Then split train/validation data.
        """
        logger.info("============================================================")
        logger.info("Beginning data loading process with the following parmeters:")
        logger.info(f"\tDataset: {self.dataset_name} - {self.dataset_type}")
        logger.info(f"\tColumns: {self.keep_columns}")
        logger.info(f"\tGeneric patient start date: {self.generic_patient_start_date}")
        if self.dataset_type != "test":
            logger.info(f"\tNumber of validation days: {self.num_validation_days}")
        if self.parallel:
            logger.info(f"\tIn parallel with up to {self.max_workers} workers.\n")
        else:
            logger.info("\tNot using parallel processing.\n")

        need_to_process_data = True
        if self.use_cached:
            if self.dataset_type == "test":
                # Check for nested test data cache
                if self.cache_manager.nested_test_data_exists(
                    self.dataset_name, self.dataset_type
                ):
                    self._load_nested_test_data_from_cache()
                    need_to_process_data = False
                    logger.info("Loaded nested test data from compressed cache")
            else:
                # Regular cache loading for training data
                cached_data = self.cache_manager.load_processed_data(
                    self.dataset_name, self.dataset_type
                )
                logger.info(
                    f"cache_manager.load_processed_data() returned dfs for:\n {[list(cached_data.keys())] if cached_data is not None else 'None'}"
                )
                # Processed data exists
                if cached_data is not None:
                    # This sets the data given the dataset type (train or test)
                    self._load_from_cache(cached_data)
                    need_to_process_data = False

        # Either processed data DNE or use_cached is False
        if need_to_process_data:
            logger.info(
                "Processed cache not found or not used, processing raw data and saving to cache..."
            )
            self._process_and_cache_data()

        # Split train/validation data
        if self.dataset_type == "train" and isinstance(self.processed_data, dict):
            self._split_train_validation()
        else:
            logger.info(
                f"Skipping train/validation split for test data or invalid processed data type. \nDataset: {self.dataset_type}\nProcessed data type: {type(self.processed_data)}"
            )

    def _load_from_cache(self, cached_data):
        """
        Load processed data from cache to self.processed_data.

        Args:
            cached_data: The cached data loaded from the cache manager.
                For train data: dict[str, pd.DataFrame] mapping patient_id -> DataFrame
                For test data: Handled by _load_nested_test_data_from_cache()
        """
        logger.info("Loading processed data from cache...")
        if self.dataset_type == "train":
            # cached_data is now dict[str, pd.DataFrame] for train data
            if not isinstance(cached_data, dict):
                raise TypeError(
                    f"Expected dict for cached train data, got {type(cached_data)}"
                )

            self.processed_data = cached_data

            # Apply column filtering to each patient's DataFrame if needed
            if self.keep_columns:
                filtered_data = {}
                for patient_id, patient_df in cached_data.items():
                    # Filter columns, but handle datetime specially since it should be the index
                    columns_to_keep = [
                        col for col in self.keep_columns if col != "datetime"
                    ]

                    # Check if all required columns are available
                    if all(col in patient_df.columns for col in columns_to_keep):
                        filtered_data[patient_id] = patient_df[columns_to_keep]
                    else:
                        missing_cols = [
                            col
                            for col in columns_to_keep
                            if col not in patient_df.columns
                        ]
                        logger.warning(
                            f"Index name: {patient_df.index.name}. "
                            f"Patient {patient_id}: Missing columns {missing_cols}. "
                            f"Available columns: {list(patient_df.columns)}"
                        )
                        # Keep all available columns from keep_columns (excluding datetime)
                        available_cols = [
                            col for col in columns_to_keep if col in patient_df.columns
                        ]
                        filtered_data[patient_id] = patient_df[available_cols]

                    # Ensure datetime index is preserved
                    if patient_df.index.name != "datetime" and not isinstance(
                        patient_df.index, pd.DatetimeIndex
                    ):
                        logger.warning(
                            f"Patient {patient_id}: Expected datetime index, but got {type(patient_df.index)}"
                        )
                        if "datetime" in patient_df.columns:
                            filtered_data[patient_id] = filtered_data[
                                patient_id
                            ].set_index(patient_df["datetime"])
                            filtered_data[patient_id].index.name = "datetime"

                self.processed_data = filtered_data
                # Update keep_columns to reflect what was actually available (excluding datetime)
                if filtered_data:
                    # Use the columns from the first patient as representative
                    first_patient_df = next(iter(filtered_data.values()))
                    self.keep_columns = ["datetime"] + first_patient_df.columns.tolist()

        elif self.dataset_type == "test":
            # For test data, we need to load the nested structure from cache
            self._load_nested_test_data_from_cache()

            # Initialize properties as None for cached test data
            self.train_dt_col_type = None
            self.val_dt_col_type = None
            self.num_train_days = 0
            self.num_validation_days = 0

    def _process_and_cache_data(self):
        """
        Try to load the raw data, process it and save it to cache.
        If the raw data is not available, fetch it from Kaggle.
        """
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()
        if self.dataset_type == "test":
            self.test_data = self.processed_data

    def load_raw(self):
        """
        Load the raw dataset from cache or fetch from source.
        The raw dataset is automatically fetched from Kaggle if not available.
        Then reads the appropriate CSV file.

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

    def _process_raw_data(
        self,
    ) -> dict[str, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]]:
        """
        Process raw test or train data with appropriate preprocessing pipelines.

        For train data: processes the entire dataset and returns a dictionary mapping
        patient IDs to their processed DataFrames.
        For test data: processes each row as a separate prediction instance and returns
        a nested dictionary mapping patient IDs to row IDs to DataFrames.

        The processed data is automatically saved to cache for future use.

        Returns:
            dict[str, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]]:
                - For train: dict mapping patient_id -> DataFrame
                - For test: dict mapping patient_id -> dict mapping row_id -> DataFrame

        Raises:
            ValueError: If raw data is not loaded or dataset_type is invalid.
        """

        # Ensure raw data is loaded
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")

        # Not cached, process the raw data
        if self.dataset_type == "train":
            return self._process_raw_train_data()
        # TODO:TONY = Test the test set's caching functions
        elif self.dataset_type == "test":
            return self._process_raw_test_data()

        else:
            raise ValueError(
                f"Unknown dataset_type: {self.dataset_type}. Must be 'train' or 'test'."
            )

    def _process_raw_train_data(
        self, store_in_between_data=False
    ) -> dict[str, pd.DataFrame]:
        """
        Process raw training data with full preprocessing pipeline.

        Splits the multi-patient training data into individual patient DataFrames,
        applies datetime processing and preprocessing pipeline to each patient
        (either in parallel or sequentially), and saves all processed data to cache.

        Args:
            store_in_between_data (bool, optional): Whether to save intermediate data
                (pre-cleaning patient data) to cache for debugging. Defaults to False.

        Returns:
            dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their processed DataFrames.

        Raises:
            ValueError: If raw data is not loaded.
            TypeError: If translated raw data is not a DataFrame (unexpected for train data).
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")

        logger.info(
            "_process_raw_train_data: Processing train data. This may take a while..."
        )
        pre_processed_data = clean_brist1d_train_data(self.raw_data)

        # Type narrowing: ensure train data returns a DataFrame
        if not isinstance(pre_processed_data, pd.DataFrame):
            raise TypeError(
                f"Expected DataFrame for train data, got {type(pre_processed_data)}"
            )

        multipatient_data_dict = split_multipatient_dataframe(
            pre_processed_data, "p_num"
        )
        logger.info(f"Processing {len(multipatient_data_dict)} patients:")
        # Store pre-cleaning dfs
        pre_cleaning_dfs_path = self.cache_manager.get_cleaning_step_data_path(
            self.dataset_name
        )
        if store_in_between_data:
            for p_num, df in multipatient_data_dict.items():
                df.to_csv(
                    pre_cleaning_dfs_path / f"patient_{p_num}_pre_cleaning.csv",
                    index=False,
                )
                logger.info(f"\tStored pre-cleaning DataFrame for patient {p_num}")
        if self.parallel:
            processed_results = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Prepare data tuples for parallel processing
                patient_data_tuples = [
                    (p_num, patient_df, self.generic_patient_start_date)
                    for p_num, patient_df in multipatient_data_dict.items()
                ]

                # Submit all tasks
                future_to_patient = {
                    executor.submit(
                        process_single_patient_data, patient_tuple
                    ): patient_tuple[0]
                    for patient_tuple in patient_data_tuples
                }

                # Collect results
                for future in as_completed(future_to_patient):
                    p_num = future_to_patient[future]
                    try:
                        patient_id, result = future.result()
                        processed_results[patient_id] = result
                        logger.info(
                            f"processed_results: Successfully processed patient {patient_id}"
                        )
                    except Exception as exc:
                        logger.error(
                            f"processed_results: Patient {p_num} generated an exception: {exc}"
                        )
        else:
            processed_results = {}
            for p_num, patient_df in multipatient_data_dict.items():
                logger.info(
                    f"\n\n========================\n Processing patient {p_num} data...\n========================\n "
                )
                patient_data_tuple = (
                    p_num,
                    patient_df,
                    self.generic_patient_start_date,
                )
                p_num, processed_patient_df = process_single_patient_data(
                    patient_data_tuple
                )
                processed_results[p_num] = processed_patient_df

        logger.info("Done processing train data. Saving processed data to cache...")
        # Save processed data to cache
        for p_num, patient_df in processed_results.items():
            self.cache_manager.save_processed_data(
                self.dataset_name, self.dataset_type, p_num, patient_df
            )

        logger.info(
            f"Successfully processed and cached data for {len(processed_results)} patients"
        )
        return processed_results

    def _process_raw_test_data(self) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Process raw test data with preprocessing transformations.

        Test data consists of individual prediction instances (rows) for each patient.
        Each row represents a separate prediction scenario and is processed independently.
        The processed data is organized in a nested dictionary structure and saved to cache.

        Returns:
            dict[str, dict[str, pd.DataFrame]]: Nested dictionary mapping:
                - First level: patient_id -> patient's data
                - Second level: row_id -> processed DataFrame for that prediction instance

        Raises:
            ValueError: If raw data is not loaded.

        Note:
            Currently uses basic transformations (datetime index, regular intervals, COB/IOB).
        """
        # Ensure raw data is loaded
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")
        logger.info("Processing test data. This may take a while...")
        data = clean_brist1d_test_data(self.raw_data)  # TODO: Very slow bottleneck
        processed_data = defaultdict(dict)

        processed_path = self.cache_manager.get_processed_data_path_for_type(
            self.dataset_name, self.dataset_type
        )
        processed_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processed test data will be cached at: {processed_path}")

        if self.parallel:
            logger.info(
                f"Processing test data in parallel with up to {self.max_workers} workers..."
            )
            # TODO:TONY - Add data processing pipeline to the test set too
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a partial function with common arguments
                process_patient_fn = functools.partial(
                    process_patient_prediction_instances,
                    base_cache_path=processed_path,
                    generic_patient_start_date=self.generic_patient_start_date,
                )

                # Map function to all patients and collect results
                results = executor.map(process_patient_fn, data.items())

                # Merge results into processed_data
                for pid, patient_data in results:
                    processed_data[pid] = patient_data

        else:
            logger.info("Processing test data sequentially...")
            # Process each patient sequentially
            for pid, patient_data in data.items():
                pid_result, patient_processed_data = (
                    process_patient_prediction_instances(
                        (pid, patient_data),
                        processed_path,
                        self.generic_patient_start_date,
                    )
                )
                processed_data[pid_result] = patient_processed_data

        logger.info(
            f"Successfully processed and cached test data for {len(processed_data)} patients"
        )

        # Save the entire nested dictionary as a compressed pickle file
        processed_path = self.cache_manager.get_processed_data_path_for_type(
            self.dataset_name, self.dataset_type
        )
        processed_path.mkdir(parents=True, exist_ok=True)

        # Save as compressed pickle
        nested_data_file = processed_path / "nested_test_data.pkl.gz"
        with gzip.open(nested_data_file, "wb") as f:
            pickle.dump(dict(processed_data), f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved nested test data to: {nested_data_file}")
        return dict(processed_data)

    def _load_nested_test_data_from_cache(self):
        """
        Load cached test data from compressed pickle file using cache manager.

        Uses the cache manager to load nested test data structure:
        {patient_id: {row_id: DataFrame}}

        Raises:
            FileNotFoundError: If the compressed cache file does not exist
        """
        self.processed_data = self.cache_manager.load_nested_test_data(
            self.dataset_name, self.dataset_type
        )

        if self.processed_data is None:
            raise FileNotFoundError(
                f"Nested test data not found for {self.dataset_name} {self.dataset_type}"
            )

        logger.info("Loaded nested test data from cache")
        logger.info(f"Loaded data for {len(self.processed_data)} patients")

    def _split_train_validation(self):
        """
        Split processed data into training and validation sets based on num_validation_days.
        Maintains dictionary structure where each patient's data is split individually.
        Calculates metadata (datetime column types, number of training days) for later use.

        Raises:
            TypeError: If processed_data is not a dictionary of patient DataFrames.
        """
        if not isinstance(self.processed_data, dict):
            raise TypeError(
                f"Cannot split train/validation data: processed_data must be a dict[str, pd.DataFrame], but got {type(self.processed_data)}"
            )
        logger.info(
            f"Splitting train/validation data with {self.num_validation_days} validation days..."
        )
        # Split each patient's data individually
        train_data_dict = {}
        validation_data_dict = {}

        for patient_id, patient_df in self.processed_data.items():
            # Ensure patient_df is a DataFrame
            if not isinstance(patient_df, pd.DataFrame):
                raise TypeError(
                    f"Expected DataFrame for patient {patient_id}, got {type(patient_df)}"
                )

            # Ensure datetime index exists - it should already be the index
            patient_data = patient_df.copy()
            if (
                not isinstance(patient_data.index, pd.DatetimeIndex)
                and patient_data.index.name != "datetime"
            ):
                if "datetime" in patient_data.columns:
                    # If datetime is a column, set it as index
                    patient_data = patient_data.set_index("datetime")
                else:
                    raise ValueError(
                        f"No datetime index found for patient {patient_id}"
                    )

            # Ensure p_num column exists for compatibility with get_train_validation_split
            if "p_num" not in patient_data.columns:
                patient_data["p_num"] = patient_id

            patient_train, patient_validation, _ = get_train_validation_split(
                patient_data, num_validation_days=self.num_validation_days
            )
            train_data_dict[patient_id] = patient_train
            validation_data_dict[patient_id] = patient_validation

        # Store as dictionaries
        self.train_data = train_data_dict
        self.validation_data = validation_data_dict

        # TODO: Fixed storage of train_data and validation data using cache manager

        # Calculate metadata from the first available patient for compatibility
        if validation_data_dict:
            first_validation_df = next(iter(validation_data_dict.values()))
            # For datetime index, get dtype of the index
            self.val_dt_col_type = first_validation_df.index.dtype

        if train_data_dict:
            first_train_df = next(iter(train_data_dict.values()))
            # For datetime index, get dtype of the index
            self.train_dt_col_type = first_train_df.index.dtype

            # Calculate total unique training days across all patients
            all_train_dates = set()
            for patient_train_df in train_data_dict.values():
                # Use index instead of datetime column
                patient_dates = patient_train_df.index.date
                all_train_dates.update(patient_dates)
            self.num_train_days = len(all_train_dates)

    # TODO: MOVE THIS TO THE time_processing.py improve the name to be more clear what this function does
    # This function is used to split the validation data by day for each patient
    # It yields tuples of (patient_id, y_input_ts_period, y_test_period) for each day
    # where:
    # - y_input_ts_period is the data from 6am to 12am of a day
    # - y_test_period is the data from 12am to 6am of the next day
    def iter_validation_day_splits(self, patient_id: str):
        """
        Get day splits for validation data for a single patient.

        This function splits the validation data by day for the specified patient,
        yielding each split as a tuple. Only works when dataset_type is 'train' and
        validation data has been created through train/validation splitting.

        Args:
            patient_id (str): ID of the patient to get validation splits for

        Yields:
            tuple: (patient_id, y_input_ts_period, y_test_period) where:
                - patient_id is the ID of the patient
                - y_input_ts_period is the data from 6am to 12am of a day, fed into the model
                - y_test_period is the data from 12am to 6am of the next day, ground truth to compare y_pred to.

        Raises:
            ValueError: If validation_data is None or patient not found.
        """
        if self.validation_data is None:
            raise ValueError(
                "Validation data is not available. This method only works with 'train' dataset_type "
                "after train/validation splitting has been performed."
            )

        # validation_data is now a dictionary, get the specific patient's data
        if not isinstance(self.validation_data, dict):
            raise TypeError(
                f"Expected dict for validation_data, got {type(self.validation_data)}"
            )

        if patient_id not in self.validation_data:
            raise ValueError(
                f"Patient {patient_id} not found in validation data. Available patients: {list(self.validation_data.keys())}"
            )

        patient_data = self.validation_data[patient_id]
        for y_input_ts_period, y_test_period in self._iter_day_splits(patient_data):
            yield patient_id, y_input_ts_period, y_test_period

    # TODO: MOVE THIS TO THE preprocessing.time_processing.py instead of splitter.py (fns were migrated)
    # TODO: change this function to be more general, so it can be used for both train and validation data
    # TODO: Change this function so that you can specify the training and test periods lengths e.g. 6 hours, 8 hours, etc.
    def _iter_day_splits(self, patient_data: pd.DataFrame):
        """
        Split each day's data into training period (6am-12am) and test period (12am-6am next day).

        This function splits the data for a single patient by day into:
        - Training period: 6am to midnight of the current day
        - Test period: midnight to 6am of the next day

        Args:
            patient_data (pd.DataFrame): Data for a single patient with datetime index

        Yields:
            tuple[pd.DataFrame, pd.DataFrame]: (train_period, test_period) where:
                - train_period is the data from 6am to 12am of a day
                - test_period is the data from 12am to 6am of the next day

        Note:
            Only yields splits where both periods have data available.
        """
        # Ensure data has datetime index
        if not isinstance(patient_data.index, pd.DatetimeIndex):
            if "datetime" in patient_data.columns:
                patient_data = patient_data.set_index("datetime")
            else:
                raise ValueError(
                    "Patient data must have datetime index or datetime column"
                )

        # Ensure data is sorted by datetime index
        patient_data = patient_data.sort_index()

        # Get datetime index for date operations
        datetime_index = pd.to_datetime(patient_data.index)

        # Group by date
        for date_key, day_data in patient_data.groupby(datetime_index.date):
            # Convert date key to proper date object
            current_date = pd.to_datetime(date_key)

            # Get next day's early morning data (12am-6am)
            next_date = current_date + pd.Timedelta(days=1)
            next_day_mask = (datetime_index.date == next_date.date) & (
                datetime_index.hour < 6
            )
            next_day_data = patient_data[next_day_mask]

            # Get current day's data (6am-12am) - use datetime index from day_data
            day_datetime_index = pd.to_datetime(day_data.index)
            current_day_mask = day_datetime_index.hour >= 6
            current_day_data = day_data[current_day_mask]

            if len(next_day_data) > 0 and len(current_day_data) > 0:
                yield current_day_data, next_day_data

    def _create_datetime_column(
        self, time_series: pd.Series, patient_start_date: pd.Timestamp
    ) -> pd.Series:
        """
        Create datetime column by inferring day rollovers from time intervals.

        DEPRECATED: This method uses interval-based day rollover detection which can be
        inaccurate. Use create_datetime_column_standalone() instead for better rollover detection.

        Given a Series of times (format "%H:%M:%S") and a start date,
        create a datetime column where the date increments based on inferred
        data collection intervals (assumes regular intervals throughout).

        Args:
            time_series (pd.Series): Series of times as strings ("%H:%M:%S")
            patient_start_date (pd.Timestamp): The date to start from (e.g., pd.Timestamp("2024-01-01"))

        Returns:
            pd.Series: Series of full datetime objects with inferred dates

        Note:
            This method assumes regular time intervals and may not handle irregular
            data collection periods correctly.
        """
        # Remove whitespace
        time_series = time_series.str.strip()
        # Convert to time objects
        times = pd.to_datetime(time_series, format="%H:%M:%S", errors="coerce").dt.time

        # Infer interval in minutes
        if len(times) > 1:
            dummy_dates = [pd.Timestamp.combine(patient_start_date, t) for t in times]
            diffs = pd.Series(dummy_dates).diff().dropna()
            interval_minutes = int(diffs.mode()[0].total_seconds() / 60)
        else:
            interval_minutes = 15  # Default if only one row

        # Calculate rows per day
        rows_per_day = int(24 * 60 / interval_minutes)
        day_offsets = pd.Series(range(len(times))) // rows_per_day

        # Build full datetime for each row
        datetime_series = [
            pd.Timestamp.combine(
                patient_start_date + pd.Timedelta(days=int(day_offset)), t
            )
            for day_offset, t in zip(day_offsets, times)
        ]
        return pd.Series(datetime_series)
