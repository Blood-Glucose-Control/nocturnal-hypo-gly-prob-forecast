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
    for each. The train data is stored as a dictionary mapping patient IDs to
    DataFrames, while test data is organized as a nested dictionary by patient ID
    and row ID.

    The loader supports intelligent caching at multiple levels:
    - Raw data caching to avoid re-downloading
    - Processed data caching to avoid re-processing
    - Train/validation split caching for consistent splits
    for each. The train data is stored as a dictionary mapping patient IDs to
    DataFrames, while test data is organized as a nested dictionary by patient ID
    and row ID.

    The loader supports intelligent caching at multiple levels:
    - Raw data caching to avoid re-downloading
    - Processed data caching to avoid re-processing
    - Train/validation split caching for consistent splits

    Attributes:
        keep_columns (list[str] | None): Specific columns to load from the dataset
        num_validation_days (int): Number of days to use for validation
        use_cached (bool): Whether to use cached processed data if available
        use_cached (bool): Whether to use cached processed data if available
        dataset_type (str): Type of dataset ('train' or 'test')
        processed_data (dict[str, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]]):
            The processed dataset - dict for train, nested dict for test
        train_data (dict[str, pd.DataFrame] | None): Training subset (when dataset_type is 'train')
        validation_data (dict[str, pd.DataFrame] | None): Validation subset (when dataset_type is 'train')
        test_data (dict[str, dict[str, pd.DataFrame]] | None): Test data (when dataset_type is 'test')
        train_dt_col_type (type): Data type of the datetime index in training data
        val_dt_col_type (type): Data type of the datetime index in validation data
        num_train_days (int): Number of unique days across all training data
        processed_data (dict[str, pd.DataFrame] | dict[str, dict[str, pd.DataFrame]]):
            The processed dataset - dict for train, nested dict for test
        train_data (dict[str, pd.DataFrame] | None): Training subset (when dataset_type is 'train')
        validation_data (dict[str, pd.DataFrame] | None): Validation subset (when dataset_type is 'train')
        test_data (dict[str, dict[str, pd.DataFrame]] | None): Test data (when dataset_type is 'test')
        train_dt_col_type (type): Data type of the datetime index in training data
        val_dt_col_type (type): Data type of the datetime index in validation data
        num_train_days (int): Number of unique days across all training data
        raw_data (pd.DataFrame): The original unprocessed dataset loaded from file

    Properties:
        dataset_name (str): Returns "kaggle_brisT1D"
        num_patients (int): Number of patients in the dataset
        patient_ids (list[str]): List of patient IDs
        data_shape_summary (dict[str, tuple[int, int]]): Shape summary for each patient

    Properties:
        dataset_name (str): Returns "kaggle_brisT1D"
        num_patients (int): Number of patients in the dataset
        patient_ids (list[str]): List of patient IDs
        data_shape_summary (dict[str, tuple[int, int]]): Shape summary for each patient
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

    @property
    def num_patients(self) -> int:
        """Get the number of patients in the dataset."""
        if self.processed_data is None:
            return 0
        return len(self.processed_data)

    @property
    def patient_ids(self) -> list[str]:
        """Get list of patient IDs in the dataset."""
        if self.processed_data is None:
            return []
        return list(self.processed_data.keys())

    @property
    def train_data_shape_summary(self) -> dict[str, tuple[int, int]]:
        """
        Get shape summary for each patient's train/validation data.
        Returns a dict mapping patient_id to shape tuple.
        Returns empty dict if dataset_type is not 'train'.
        """
        if self.dataset_type != "train" or not isinstance(self.processed_data, dict):
            return {}

        shape_summary = {}
        for patient_id, patient_df in self.processed_data.items():
            if isinstance(patient_df, pd.DataFrame):
                shape_summary[patient_id] = patient_df.shape
        return shape_summary

    @property
    def test_data_shape_summary(self) -> dict[tuple[str, str], tuple[int, int]]:
        """
        Get shape summary for each patient's test data (nested by sub_id).
        Returns a dict mapping (patient_id, sub_id) to shape tuple.
        Returns empty dict if dataset_type is not 'test'.
        """
        if self.dataset_type != "test" or not isinstance(self.processed_data, dict):
            return {}

        shape_summary = {}
        for patient_id, patient_data in self.processed_data.items():
            if isinstance(patient_data, dict):
                for sub_id, sub_df in patient_data.items():
                    if isinstance(sub_df, pd.DataFrame):
                        shape_summary[(patient_id, sub_id)] = sub_df.shape
        return shape_summary

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

        # Cache the split data for future use
        for patient_id, patient_train_df in train_data_dict.items():
            try:
                self.cache_manager.save_processed_data(
                    self.dataset_name, "train", patient_id, patient_train_df
                )
            except Exception as e:
                logger.error(
                    f"Failed to cache train data for patient {patient_id}: {e}",
                    exc_info=True,
                )
        for patient_id, patient_val_df in validation_data_dict.items():
            try:
                self.cache_manager.save_processed_data(
                    self.dataset_name, "validation", patient_id, patient_val_df
                )
            except Exception as e:
                logger.error(
                    f"Failed to cache validation data for patient {patient_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Cached train/validation split data for {len(train_data_dict)} patients"
        )
        # Cache the split data for future use
        for patient_id, patient_train_df in train_data_dict.items():
            try:
                self.cache_manager.save_processed_data(
                    self.dataset_name, "train", patient_id, patient_train_df
                )
            except Exception as e:
                logger.error(
                    f"Failed to cache train data for patient {patient_id}: {e}",
                    exc_info=True,
                )
        for patient_id, patient_val_df in validation_data_dict.items():
            try:
                self.cache_manager.save_processed_data(
                    self.dataset_name, "validation", patient_id, patient_val_df
                )
            except Exception as e:
                logger.error(
                    f"Failed to cache validation data for patient {patient_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Cached train/validation split data for {len(train_data_dict)} patients"
        )

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
