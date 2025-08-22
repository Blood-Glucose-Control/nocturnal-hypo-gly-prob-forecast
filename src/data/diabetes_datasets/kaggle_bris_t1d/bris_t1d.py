"""
Kaggle Bristol T1D Dataset Loader.

This module provides functionality to load and process the Bristol Type 1 Diabetes
dataset from Kaggle. It handles both train and test datasets, with options for caching
processed data to avoid redundant computations.

The module supports data preprocessing including time interval regularization,
carbohydrate on board (COB) calculation, insulin on board (IOB) calculation,
and train/validation splitting.
"""

import logging
import pandas as pd
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from src.data.preprocessing.pipeline import preprocessing_pipeline
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
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.cache_manager import get_cache_manager
from src.data.dataset_configs import get_dataset_config

from src.data.diabetes_datasets.kaggle_bris_t1d.data_cleaner import (
    clean_brist1d_train_data,
    clean_brist1d_test_data,
)


logger = logging.getLogger(__name__)


def create_datetime_column_standalone(
    time_series: pd.Series, patient_start_date: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    """
    Simple version that detects day rollovers and increments date accordingly.

    Args:
        time_series (pd.Series): Series of times as strings ("%H:%M:%S")
        patient_start_date (pd.Timestamp): The date to start from

    Returns:
        tuple[pd.Series, pd.Series]: (date_series, time_series) where:
            - date_series: Series of dates (datetime64[ns] with date info)
            - time_series: Series of times (time objects)
    """
    if len(time_series) == 0:
        return pd.Series([], dtype="datetime64[ns]"), pd.Series([], dtype="time")

    # Clean and convert to time objects
    time_series = time_series.str.strip()
    times = pd.to_datetime(time_series, format="%H:%M:%S", errors="coerce").dt.time
    result_dates = []
    current_date = patient_start_date.date()  # Get just the date part!
    for i, current_time in enumerate(times):
        if pd.isna(current_time):
            result_dates.append(pd.NaT)
            continue

        # Check for day rollover (current hour:minute < previous hour:minute)
        if i > 0 and not pd.isna(times.iloc[i - 1]):
            prev_time = times.iloc[i - 1]

            # Simple comparison: if current time < previous time, we rolled over
            current_minutes = current_time.hour * 60 + current_time.minute
            prev_minutes = prev_time.hour * 60 + prev_time.minute

            if current_minutes < prev_minutes:
                current_date = current_date + pd.Timedelta(days=1)

        # Combine current date with current time
        result_dates.append(current_date)

    result_dates = pd.Series(result_dates, dtype="datetime64[ns]")
    logger.info(f"\tcreate_dt_col() - Result start date: {result_dates.iloc[0]}")
    logger.info(f"\tcreate_dt_col() - Result end date: {result_dates.iloc[-1]}")
    logger.info(f"\tcreate_dt_col() - Result start time: {times.iloc[0]}")
    logger.info(f"\tcreate_dt_col() - Result end time: {times.iloc[-1]}")
    logger.info(
        f"\tcreate_dt_col() - Any NaT values in dates? {result_dates.isna().sum()}"
    )
    logger.info(f"\tcreate_dt_col() - Any NaT values in times? {times.isna().sum()}")

    return (result_dates, times)


def process_patient_data_standalone(patient_data_tuple: tuple) -> tuple:
    """
    Process a single patient's data including datetime creation and preprocessing.

    Note: Standalones were created to support multiprocessing, corruption issues occur otherwise.

    Args:
        patient_data_tuple: Tuple containing (p_num, data, generic_patient_start_date)

    Returns:
        Tuple containing (p_num, processed_data)
    """
    p_num, data, generic_patient_start_date = patient_data_tuple
    logger.info(
        f"Running process_patient_data_standalone(), \n\t\t\tProcessing patient {p_num} data...\n\t\t\tPatient start date: {generic_patient_start_date.date()}"
    )
    logger.info(f"\tInputed patient start time: {data['datetime'].iloc[0]}")
    # Create a copy to avoid modifying the original
    data_copy = data.copy()

    # Create datetime column
    patient_dates, patient_times = create_datetime_column_standalone(
        data_copy["datetime"], generic_patient_start_date
    )
    logger.info(
        f"\tCreated columns for patient {p_num} data...\n\t\t\tPatient start date: {patient_dates.iloc[0]}"
    )
    logger.info(f"\tResult patient start time: {patient_times.iloc[0]}")
    logger.info(f"\tLength of dates: {len(patient_dates)}")
    logger.info(f"\tLength of times: {len(patient_times)}")

    data_copy["datetime"] = [
        pd.Timestamp.combine(date, time)
        if pd.notna(date) and pd.notna(time)
        else "BAD: " + str(date) + " " + str(time)
        for date, time in zip(patient_dates, patient_times)
    ]

    # Convert datetime column to index
    data_copy = data_copy.set_index("datetime", drop=True)
    # TODO:TONY - Remove this
    cache_manager = get_cache_manager()
    cache_manager.get_cleaning_step_data_path("kaggle_brisT1D")
    data_copy.to_csv(
        cache_manager.get_cleaning_step_data_path("kaggle_brisT1D")
        / f"datetime_index/{p_num}.csv",
        index=True,
    )
    # Run preprocessing pipeline
    processed_data = preprocessing_pipeline(p_num, data_copy)

    return p_num, processed_data


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
            max_workers (int, optional): Maximum number of workers for parallel processing.
                Defaults to 9.
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
        self.train_data = None
        self.processed_data = None
        self.raw_data = None
        self.validation_data = None
        self.train_dt_col_type = None
        self.val_dt_col_type = None
        self.num_train_days = None

        # Preload data
        self.load_data()

    @property
    def dataset_name(self):
        return "kaggle_brisT1D"

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
        logger.info(f"\tNumber of validation days: {self.num_validation_days}")
        if self.parallel:
            logger.info(f"\tIn parallel with up to {self.max_workers} workers.\n")
        else:
            logger.info("\tNot using parallel processing.\n")

        need_to_process_data = True
        if self.use_cached:
            cached_data = self.cache_manager.load_processed_data(
                self.dataset_name, self.dataset_type
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
        if self.dataset_type == "train" and isinstance(
            self.processed_data, pd.DataFrame
        ):
            self._split_train_validation()

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
        Try to load the raw data, process it and save it to cache.
        If the raw data is not available, fetch it from Kaggle.
        """
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()

    def _split_train_validation(self):
        """
        Split processed data into training and validation sets based on num_validation_days.
        Calculates metadata (datetime column types, number of training days) for later use.

        Raises:
            TypeError: If processed_data is not a DataFrame.
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

    def _translate_raw_data(
        self, raw_data: pd.DataFrame
    ) -> pd.DataFrame | dict[str, dict[str, pd.DataFrame]]:
        """
        Translate the raw data to the correct format for the preprocessing pipeline.

        For train data, returns a single DataFrame with cleaned and standardized columns.
        For test data, returns a nested dictionary structure where each patient's data
        is organized by row IDs, since test data represents individual prediction instances
        rather than continuous time series.

        Args:
            raw_data (pd.DataFrame): The raw data loaded from CSV file.

        Returns:
            pd.DataFrame | dict[str, dict[str, pd.DataFrame]]:
                - For train: Single DataFrame with processed data
                - For test: Nested dict as {patient_id: {row_id: DataFrame}}
        """
        if self.dataset_type == "train":
            result = clean_brist1d_train_data(raw_data)
            # Ensure type safety - this should always be a DataFrame for train data
            assert isinstance(
                result, pd.DataFrame
            ), "Train data cleaning should return DataFrame"
            return result
        elif self.dataset_type == "test":
            return clean_brist1d_test_data(raw_data)
        else:
            raise ValueError(
                f"Unknown dataset_type: {self.dataset_type}. Must be 'train' or 'test'."
            )

    def _process_raw_data(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """
        Process raw test or train data.
        For train data, process the entire dataset (which is just one single df)
        For test data, a row is a df, so we need to process each row in parallel.
        Result: Save processed data to cache.
        // TODO:TONY - Test the test set's caching functions
        """
        # Ensure raw data is loaded
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")

        # Not cached, process the raw data
        if self.dataset_type == "train":
            return self._process_raw_train_data()

        elif self.dataset_type == "test":
            return self._process_raw_test_data()

        else:
            raise ValueError(
                f"Unknown dataset_type: {self.dataset_type}. Must be 'train' or 'test'."
            )

    def _process_raw_train_data(self) -> dict[str, pd.DataFrame]:
        """
        Process raw training data.

        Returns
            dict[str, pd.DataFrame]: Processed training data.
        """
        BOLD = "\033[1m"
        RESET = "\033[0m"
        # Ensure raw data is loaded
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")

        logger.info(
            "_process_raw_train_data: Processing train data. This may take a while..."
        )
        pre_processed_data = self._translate_raw_data(self.raw_data)

        # Type narrowing: ensure train data returns a DataFrame
        if not isinstance(pre_processed_data, pd.DataFrame):
            raise TypeError(
                f"Expected DataFrame for train data, got {type(pre_processed_data)}"
            )

        multipatient_data_dict = self.multipatient_to_dict_of_single_patient(
            pre_processed_data
        )
        logger.info(f"Processing {len(multipatient_data_dict)} patients:")
        # Store pre-cleaning dfs
        pre_cleaning_dfs_path = self.cache_manager.get_cleaning_step_data_path(
            self.dataset_name
        )
        for p_num, df in multipatient_data_dict.items():
            df.to_csv(
                pre_cleaning_dfs_path / f"patient_{p_num}_pre_cleaning.csv", index=False
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
                        process_patient_data_standalone, patient_tuple
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
                    f"\n\n{BOLD}========================\n Processing patient {p_num} data...\n========================{RESET}\n "
                )
                patient_data_tuple = (
                    p_num,
                    patient_df,
                    self.generic_patient_start_date,
                )
                p_num, processed_patient_df = process_patient_data_standalone(
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
        # Ensure raw data is loaded
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")
        logger.info("Processing test data. This may take a while...")
        data = self._translate_raw_data(self.raw_data)
        processed_data = defaultdict(dict)

        processed_path = self.cache_manager.get_processed_data_path_for_type(
            self.dataset_name, self.dataset_type
        )
        processed_path.mkdir(parents=True, exist_ok=True)

        # Process each patient in parallel
        # TODO:TONY - Add data processing pipeline to the test set too
        with ProcessPoolExecutor() as executor:
            # Create a partial function with common arguments
            process_patient_fn = functools.partial(
                self._process_patient_test_data, base_cache_path=processed_path
            )

            # Map function to all patients and collect results
            results = executor.map(process_patient_fn, data.items())

            # Merge results into processed_data
            for pid, patient_data in results:
                processed_data[pid] = patient_data

            return processed_data

    def _process_patient_test_data(self, patient_item, base_cache_path):
        """
        Process data for a single patient in parallel.

        Applies transformations (datetime index, regular intervals, COB/IOB calculations)
        to each row of patient data and saves to cache.

        Args:
            patient_item (tuple): (patient_id, patient_data_dict) mapping row_ids to DataFrames
            base_cache_path (Path): Base directory for caching processed data

        Returns:
            tuple: (patient_id, processed_rows_dict) mapping row_ids to processed DataFrames
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

    def multipatient_to_dict_of_single_patient(
        self, data: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Convert multi-patient data to single patient data.

        Args:
            data (pd.DataFrame): Multi-patient data, with p_num column.

        Returns:
            dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their data.
        """
        unique_patient_ids = data["p_num"].unique()
        patient_data = {
            patient_id: data[data["p_num"] == patient_id].copy()
            for patient_id in unique_patient_ids
        }
        return patient_data

    def _create_datetime_column(
        self, time_series: pd.Series, patient_start_date: pd.Timestamp
    ) -> pd.Series:
        """
        Given a Series of times (format "%H:%M:%S") and a start date,
        create a datetime column where the time stays the same and the date rolls forward
        for each full day of data.

        Args:
            time_series (pd.Series): Series of times as strings ("%H:%M:%S")
            patient_start_date (pd.Timestamp): The date to start from (e.g., pd.Timestamp("2024-01-01"))

        Returns:
            pd.Series: Series of full datetime objects with correct dates
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
