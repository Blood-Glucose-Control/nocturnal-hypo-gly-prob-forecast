# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Lynch 2022 Dataset Loader.

This module provides functionality to load and process the Lynch 2022 dataset,
mirroring the structure of the Kaggle Bristol T1D loader. It handles both train
and test datasets with caching and preprocessing pipelines.
"""

import logging
import functools
import gzip
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.data.cache_manager import get_cache_manager
from src.data.dataset_configs import get_dataset_config
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.diabetes_datasets.lynch_2022.data_cleaner import (
    clean_lynch2022_test_data,
    clean_lynch2022_train_data,
    load_lynch2022_raw_dataset,
    process_patient_prediction_instances,
    process_single_patient_data,
)
from src.data.preprocessing.data_splitting import split_multipatient_dataframe
from src.data.preprocessing.time_processing import get_train_validation_split

logger = logging.getLogger(__name__)


class Lynch2022DataLoader(DatasetBase):
    """Data loader for the Lynch 2022 IOBP2 RCT dataset."""

    def __init__(
        self,
        keep_columns: list[str] | None = None,
        use_cached: bool = True,
        num_validation_days: int = 20,
        train_percentage: float = 0.9,
        dataset_type: str = "train",
        config: dict | None = None,
        parallel: bool = True,
        generic_patient_start_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
        max_workers: int = 3,
    ):
        """Initialize the Lynch 2022 data loader."""
        # Ensure 'datetime' is included in keep_columns if specified
        if keep_columns is not None and "datetime" not in keep_columns:
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

        # Data attributes
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
        logger.info("Initializing Lynch 2022 Data Loader...")
        logger.info(
            f"Use of train_percentage parameter is currently not implemented: {train_percentage}"
        )
        logger.info(
            f"Use of configuration parameter is currently not implemented: {config}"
        )
        self.load_data()

    @property
    def dataset_name(self):
        return "lynch_2022"

    @property
    def description(self):
        """Return a description of the dataset.

        Returns:
            str: Human-readable description of study.
        """
        return """
                Objective: 'To evaluate a transition from standard-of-care (SC) management of type 1 diabetes
                    (any insulin delivery method including hybrid closed-loop systems plus real-time continuous
                    glucose monitoring [CGM]) to use of the insulin-only configuration of the iLet® bionic
                    pancreas (BP) in 90 adults and children (age 6–71 years).'
                Title: 'The Insulin-Only Bionic Pancreas Pivotal Trial Extension Study: A Multi-Center Single-Arm
                    Evaluation of the Insulin-Only Configuration of the Bionic Pancreas in Adults and Youth with
                    Type 1 Diabetes'
                n = 440 participants using either insulin aspart, insulin lispro, or fast-acting insulin aspart
                Duration: 13 weeks
                Paper: https://journals.sagepub.com/doi/full/10.1089/dia.2022.0341
            """

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
    def data_shape_summary(self) -> dict[str | tuple[str, str], tuple[int, int]]:
        """Get shape summary for each patient's data.
        Returns a dict mapping patient_id or (patient_id, sub_id) to shape tuple.
        """
        if not self.processed_data:
            return {}
        return {
            patient_id: df.shape
            for patient_id, df in self.processed_data.items()
            if isinstance(df, pd.DataFrame)
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten processed_data into a single DataFrame.

        - For train-like data (patient_id -> DataFrame) returns rows with p_num and datetime.
        - For nested test-like data (patient_id -> {sub_id -> DataFrame}) returns rows with p_num, sub_id and datetime.

        The resulting DataFrame will have a multi-index:
          - train: (p_num, datetime)
          - test: (p_num, sub_id, datetime)

        Returns an empty DataFrame if processed_data is None or empty.
        """

        if not self.processed_data:
            return pd.DataFrame()

        parts = []

        # Case 1: processed_data is patient_id -> DataFrame
        if all(isinstance(v, pd.DataFrame) for v in self.processed_data.values()):
            for pid, df in self.processed_data.items():
                assert isinstance(df, pd.DataFrame)  # Type guard for Pylance
                tmp = df.copy()
                # move datetime index into column if present as index
                if (
                    isinstance(tmp.index, pd.DatetimeIndex)
                    or tmp.index.name == "datetime"
                ):
                    tmp = tmp.reset_index()
                if "p_num" not in tmp.columns:
                    tmp["p_num"] = pid
                parts.append(tmp)
            if not parts:
                return pd.DataFrame()
            out = pd.concat(parts, ignore_index=True)
            if "datetime" in out.columns:
                out = out.set_index(["p_num", "datetime"])
            else:
                out = out.set_index("p_num")
            return out

        # Case 2: processed_data is nested dict patient_id -> {sub_id -> DataFrame}
        for pid, inner in self.processed_data.items():
            if isinstance(inner, dict):
                for sub_id, df in inner.items():
                    tmp = df.copy()
                    if (
                        isinstance(tmp.index, pd.DatetimeIndex)
                        or tmp.index.name == "datetime"
                    ):
                        tmp = tmp.reset_index()
                    tmp["p_num"] = pid
                    tmp["sub_id"] = sub_id
                    parts.append(tmp)
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        if "datetime" in out.columns:
            out = out.set_index(["p_num", "sub_id", "datetime"])
        else:
            out = out.set_index(["p_num", "sub_id"])
        return out

    def load_data(self) -> None:
        """
        Load processed data from cache or process raw data and save to cache.
        Then split train/validation data.

        Side Effects:
            Sets self.processed_data, self.train_data, and self.validation_data.
        """
        logger.info("============================================================")
        logger.info("Beginning data loading process with the following parameters:")
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
                # For train data, try to load full processed data first
                cached_full_data = self.cache_manager.load_full_processed_data(
                    self.dataset_name
                )
                if cached_full_data is not None:
                    self.processed_data = cached_full_data
                    logger.info(
                        f"Loaded full processed data from cache for {len(cached_full_data)} patients"
                    )
                    need_to_process_data = False
                else:
                    # Fallback to old method for backwards compatibility
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
        If the raw data is not available, fetch it from source.
        """
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()
        if self.dataset_type == "test":
            self.test_data = self.processed_data

    def load_raw(self):
        """Load the raw Lynch dataset directly from the SAS tables."""
        raw_data_path = self.cache_manager.get_raw_data_path(self.dataset_name)
        sas_base = raw_data_path / "IOBP2 RCT Public Dataset" / "Data Tables in SAS"

        if not sas_base.exists():
            raise FileNotFoundError(
                f"Expected SAS tables at {sas_base} but directory does not exist."
            )

        logger.info("Loading Lynch 2022 raw data from %s", sas_base)
        raw_df = load_lynch2022_raw_dataset(sas_base)
        logger.info("Loaded Lynch 2022 raw data with shape %s", raw_df.shape)
        return raw_df

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

        # Process the raw data
        if self.dataset_type == "train":
            return self._process_raw_train_data()
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
        Process raw Lynch 2022 training data through cleaning and preprocessing pipeline.

        Returns:
            Dictionary mapping patient IDs to their processed DataFrames
        """
        logger.info("Cleaning Lynch 2022 train data...")
        assert self.raw_data is not None, "Raw data not loaded. Call load_raw() first."
        pre_processed_data = clean_lynch2022_train_data(self.raw_data)

        logger.info("Running preprocessing pipeline on Lynch 2022 train data...")

        # Split data into per-patient dictionary FIRST (matches Kaggle pattern)
        multipatient_data_dict = split_multipatient_dataframe(
            pre_processed_data, "p_num"
        )

        # Create tuples from the dictionary (matches Kaggle pattern exactly)
        patient_data_tuples = [
            (p_num, patient_df, self.generic_patient_start_date)
            for p_num, patient_df in multipatient_data_dict.items()
        ]

        if self.parallel:
            logger.info(
                f"Processing {len(patient_data_tuples)} Lynch patients in parallel with {self.max_workers} workers..."
            )
            processed_dict = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_patient_data,
                        patient_tuple,
                        store_in_between_data,
                    ): patient_tuple[0]
                    for patient_tuple in patient_data_tuples
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing Lynch patients",
                    unit="patient",
                ):
                    p_num = futures[future]
                    try:
                        patient_id, processed_data = future.result()
                        processed_dict[patient_id] = processed_data
                    except Exception as exc:
                        logger.error(
                            f"Lynch patient {p_num} generated an exception: {exc}"
                        )
                        raise exc
        else:
            logger.info(
                f"Processing {len(patient_data_tuples)} Lynch patients sequentially..."
            )
            processed_dict = {}
            for patient_tuple in tqdm(
                patient_data_tuples, desc="Processing Lynch patients", unit="patient"
            ):
                patient_id, processed_data = process_single_patient_data(
                    patient_tuple, store_in_between_data
                )
                processed_dict[patient_id] = processed_data

        logger.info(f"Processed {len(processed_dict)} Lynch patients successfully")

        # Save full processed data (before split) to cache - MATCHES KAGGLE
        logger.info("Saving full processed data to cache...")
        self.cache_manager.save_full_processed_data(self.dataset_name, processed_dict)
        logger.info(
            f"Successfully processed and cached full data for {len(processed_dict)} patients"
        )

        return processed_dict

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
            Uses Lynch 2022 specific transformations for test data processing.
        """
        # Ensure raw data is loaded
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")

        logger.info("Processing test data. This may take a while...")
        data = clean_lynch2022_test_data(self.raw_data)
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
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Create a partial function with common arguments
                process_patient_fn = functools.partial(
                    process_patient_prediction_instances,
                    base_cache_path=processed_path,
                    generic_patient_start_date=self.generic_patient_start_date,
                )

                # Map function to all patients and collect results with progress bar
                data_items = list(data.items())
                results = list(
                    tqdm(
                        executor.map(process_patient_fn, data_items),
                        total=len(data_items),
                        desc="Processing test patients",
                        unit="patient",
                    )
                )

                # Merge results into processed_data
                for pid, patient_data in results:
                    processed_data[pid] = patient_data

        else:
            logger.info("Processing test data sequentially...")
            # Process each patient sequentially with progress bar
            for pid, patient_data in tqdm(
                data.items(), desc="Processing test patients", unit="patient"
            ):
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
        Uses serialized caching to avoid re-splitting data with the same parameters.
        Maintains dictionary structure where each patient's data is split individually.
        Calculates metadata (datetime column types, number of training days) for later use.

        Raises:
            TypeError: If processed_data is not a dictionary of patient DataFrames.
        """
        if not isinstance(self.processed_data, dict):
            raise TypeError(
                f"Cannot split train/validation data: processed_data must be a dict[str, pd.DataFrame], but got {type(self.processed_data)}"
            )

        # Define split parameters - MATCHES KAGGLE
        split_params = {
            "num_validation_days": self.num_validation_days,
            "split_method": "get_train_validation_split",
            "dataset_type": self.dataset_type,
        }

        # Try to load existing split data - MATCHES KAGGLE
        cached_split_data = self.cache_manager.load_split_data(
            self.dataset_name, split_params
        )

        if cached_split_data is not None:
            train_data_dict, validation_data_dict = cached_split_data
            logger.info(
                f"Loaded existing train/validation split from cache for {len(train_data_dict)} patients"
            )
        else:
            logger.info(
                f"No cached split found, splitting train/validation data with {self.num_validation_days} validation days..."
            )
            # Split each patient's data individually
            train_data_dict = {}
            validation_data_dict = {}
            skipped_patients = []
            min_required_days = (
                self.num_validation_days + 1
            )  # At least 1 day for training

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

                # Calculate number of unique days for this patient
                assert isinstance(
                    patient_data.index, pd.DatetimeIndex
                ), "Index must be a DatetimeIndex"
                unique_days = patient_data.index.normalize().nunique()

                # Skip patients with insufficient data
                if unique_days < min_required_days:
                    logger.warning(
                        f"Skipping patient {patient_id}: Only {unique_days} unique days "
                        f"(need at least {min_required_days} days for {self.num_validation_days} validation days)"
                    )
                    skipped_patients.append(patient_id)
                    continue

                # Ensure p_num column exists for compatibility with get_train_validation_split
                if "p_num" not in patient_data.columns:
                    patient_data["p_num"] = patient_id

                try:
                    patient_train, patient_validation, _ = get_train_validation_split(
                        patient_data, num_validation_days=self.num_validation_days
                    )
                    train_data_dict[patient_id] = patient_train
                    validation_data_dict[patient_id] = patient_validation
                except ValueError as e:
                    logger.warning(f"Failed to split patient {patient_id}: {e}")
                    skipped_patients.append(patient_id)
                    continue

            # Log summary of skipped patients
            if skipped_patients:
                logger.warning(
                    f"Skipped {len(skipped_patients)} patients with insufficient data: "
                    f"{skipped_patients[:10]}{'...' if len(skipped_patients) > 10 else ''}"
                )

            if not train_data_dict:
                raise ValueError(
                    f"No patients have sufficient data for train/validation split. "
                    f"Required: at least {min_required_days} days per patient."
                )

            # Save split data to cache using serialized format - MATCHES KAGGLE
            self.cache_manager.save_split_data(
                self.dataset_name, train_data_dict, validation_data_dict, split_params
            )
            logger.info(
                f"Cached new train/validation split data for {len(train_data_dict)} patients"
            )

        self.train_data = train_data_dict
        self.validation_data = validation_data_dict

        # Store as dictionaries
        # self.train_data = train_data_dict
        # self.validation_data = validation_data_dict

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
                assert isinstance(
                    patient_train_df.index, pd.DatetimeIndex
                ), "Index must be a DatetimeIndex"
                patient_dates = patient_train_df.index.date
                all_train_dates.update(patient_dates)
            self.num_train_days = len(all_train_dates)
