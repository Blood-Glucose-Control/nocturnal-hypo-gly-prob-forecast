"""
Gluroo dataset loader module for diabetes data processing.

This module provides functionality for loading, processing, and preparing Gluroo
diabetes data for machine learning applications. The GlurooDataLoader class handles:
- Loading raw data from CSV files
- Cleaning and preprocessing the data
- Creating physiological features (insulin-on-board, carbs-on-board)
- Splitting data into training and validation sets
- Providing day-by-day data splits for time-based predictions

The loader implements the DatasetBase interface for standardized access across
different data sources.
"""

import pandas as pd
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.diabetes_datasets.gluroo.data_cleaner import clean_gluroo_data
from src.data.preprocessing.time_processing import get_train_validation_split
from src.data.preprocessing.sampling import ensure_regular_time_intervals
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)


# TODO: Maybe need to return the test set too.
class GlurooDataLoader(DatasetBase):
    """
    Loader for Gluroo diabetes dataset with preprocessing and feature engineering.

    This class loads Gluroo diabetes data, processes it by cleaning irregularities,
    engineering physiological features, and preparing it for training machine learning
    models. It provides methods to access train/validation splits and day-by-day
    prediction windows.

    Attributes:
        keep_columns (list[str] | None): Columns to keep from the raw data.
        num_validation_days (int): Number of days to use for validation.
        file_path (str | None): Path to the Gluroo data file.
        config (dict | None): Configuration for data processing.
        use_cached (bool): Whether to use previously cached processed data.
        raw_data (pd.DataFrame | None): Raw data loaded from source.
        processed_data (pd.DataFrame | None): Processed data after cleaning and feature engineering.
        train_data (pd.DataFrame | None): Data for model training.
        validation_data (pd.DataFrame | None): Data for model validation.
    """

    def __init__(
        self,
        keep_columns: list[str] | None = None,
        num_validation_days: int = 20,
        file_path: str | None = None,
        config: dict | None = None,
        use_cached: bool = False,
    ):
        """
        Initialize the Gluroo data loader.

        Args:
            keep_columns (list[str] | None): Columns to retain from the raw data.
                If None, all columns are kept.
            num_validation_days (int): Number of days to reserve for validation.
                Default is 20 days.
            file_path (str | None): Path to the CSV file containing Gluroo data.
                Required unless use_cached is True.
            config (dict | None): Configuration dictionary for data processing steps.
                Can include parameters for cleaning, feature engineering, etc.
            use_cached (bool): If True, load previously processed data from cache
                instead of processing raw data again. Default is False.
        """
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.file_path = file_path  # Raw file path
        self.config = config
        self.use_cached = use_cached
        self.raw_data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None
        self.train_data: pd.DataFrame | None = None
        self.validation_data: pd.DataFrame | None = None
        self.load_data()

    @property
    def dataset_name(self):
        """
        Get the name identifier for this dataset.

        Returns:
            str: The dataset name ('gluroo')
        """
        return "gluroo"

    def load_raw(self):
        """
        Load the raw Gluroo dataset from a CSV file.

        Loads data from the file specified in the file_path attribute,
        optionally selecting only the columns specified in keep_columns.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.

        Raises:
            ValueError: If file_path is None.
        """
        if self.file_path is None:
            raise ValueError("File path is required")
        return pd.read_csv(self.file_path, usecols=self.keep_columns)

    def load_data(self):
        """
        Load and process the raw data, setting up train/validation splits.

        If use_cached is True, loads previously processed data from cache.
        Otherwise, loads raw data, processes it, and saves it to cache.
        Then splits the processed data into training and validation sets.
        """
        if self.use_cached:
            cached_data = pd.read_csv("gluroo_cached.csv")
            self.processed_data = cached_data
        else:
            self.raw_data = self.load_raw()
            self.processed_data = self._process_raw_data()
            self.processed_data.to_csv("/gluroo_cached.csv", index=False, mode="w")

        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=self.num_validation_days
        )

    def _process_raw_data(self):
        """
        Process the raw data using Gluroo-specific cleaning and feature engineering.

        This method performs the following steps:
        1. Selects columns according to keep_columns
        2. Cleans the data using Gluroo-specific cleaning functions
        3. Ensures regular time intervals in the data
        4. Adds carbs-on-board and carb availability columns
        5. Adds insulin-on-board and insulin availability columns

        Returns:
            pd.DataFrame: The fully processed data with engineered features.

        Raises:
            ValueError: If raw_data is None.
        """
        if self.raw_data is None:
            raise ValueError("Raw data is required")

        # Get the subset of columns
        if self.keep_columns is None:
            raw = self.raw_data.copy()
        else:
            # Get the subset of columns - ensure it's a DataFrame not a Series
            if len(self.keep_columns) == 1:
                # If only one column is selected, pandas might return a Series
                # Convert it back to DataFrame
                column_name = self.keep_columns[0]
                raw = self.raw_data[
                    [column_name]
                ].copy()  # Using [[col]] ensures a DataFrame
            else:
                raw = self.raw_data[self.keep_columns].copy()

        cleaned_df = clean_gluroo_data(raw, self.config)
        processed_df_regular = ensure_regular_time_intervals(cleaned_df)
        processed_df_cob = create_cob_and_carb_availability_cols(processed_df_regular)
        processed_df_iob = create_iob_and_ins_availability_cols(processed_df_cob)
        return processed_df_iob

    def get_validation_day_splits(self, patient_id: str):
        """
        Generate day-by-day training and testing periods for a specific patient.

        For each day in the validation data, yields:
        - Current day's data from 6am-12am (training period)
        - Next day's data from 12am-6am (testing/prediction period)

        This is useful for next-day prediction tasks, where models train on
        daytime data and predict overnight glucose values.

        Args:
            patient_id (str): Identifier for the patient whose data to split.

        Yields:
            tuple: (patient_id, train_period_data, test_period_data) where
                  train_period_data is the current day's data (6am-12am)
                  test_period_data is the next day's early morning data (12am-6am)

        Raises:
            ValueError: If validation_data is None (not loaded).
        """
        if self.validation_data is None:
            raise ValueError(
                "Validation data is not loaded. Please ensure data is loaded before calling this method."
            )
        patient_data = self.validation_data[self.validation_data["p_num"] == patient_id]
        for train_period, test_period in self._get_day_splits(patient_data):
            yield patient_id, train_period, test_period

    def _get_day_splits(self, patient_data: pd.DataFrame):
        """
        Split each day's data into training period (6am-12am) and test period (12am-6am next day).

        This method implements the actual day-by-day splitting logic used by
        get_validation_day_splits. It divides each day's data into a daytime
        portion for training and the following overnight period for testing.

        Args:
            patient_data (pd.DataFrame): Data for a single patient, must include
                                        'datetime' column.

        Yields:
            tuple: (current_day_data, next_day_data) where
                  current_day_data is from 6am-12am of the current day
                  next_day_data is from 12am-6am of the following day
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
