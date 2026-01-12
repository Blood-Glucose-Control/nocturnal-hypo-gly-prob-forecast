# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Gluroo dataset loader module for diabetes data processing.

This module provides functionality for loading, processing, and preparing Gluroo
diabetes data for machine learning applications. The GlurooDataLoader class handles:
- Loading raw data from CSV files
- Translating Gluroo-specific data formats to standardized format
- Applying preprocessing pipeline for cleaning and feature engineering
- Creating physiological features (insulin-on-board, carbs-on-board)
- Splitting data into training and validation sets
- Providing day-by-day data splits for time-based predictions
- Caching processed data for improved performance

The loader implements the DatasetBase interface for standardized access across
different data sources and delegates heavy processing work to the preprocessing
pipeline system.
"""

import pandas as pd
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.diabetes_datasets.gluroo.data_cleaner import data_translation
from src.data.preprocessing.pipeline import preprocessing_pipeline
from src.data.preprocessing.time_processing import (
    get_train_validation_split,
    iter_daily_context_forecast_splits,
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
        parallel: bool = True,
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
                Can include parameters for cleaning, feature engineering, etc. #TODO: Is this config really necessary? Why is it here?
            use_cached (bool): If True, load previously processed data from cache
                instead of processing raw data again. Default is False.
        """
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.file_path = file_path  # Raw file path
        self.config = config  # TODO: Is this config really necessary? Why is it here?
        self.use_cached = use_cached
        self.raw_data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None
        self.train_data: pd.DataFrame | None = None
        self.validation_data: pd.DataFrame | None = None
        self.parallel = parallel  # Whether to use parallel processing
        self.load_data()

        if self.parallel:
            print(
                "Parallel processing is not yet developed for Gluroo data loader. Proceeding with sequential processing."
            )

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

    def load_data(self) -> None:
        """
        Load and process the raw data, setting up train/validation splits.

        If use_cached is True, loads previously processed data from cache.
        Otherwise, loads raw data, processes it, and saves it to cache.
        Then splits the processed data into training and validation sets.
        
        Side Effects:
            Sets self.processed_data, self.train_data, and self.validation_data.
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
        1. Selects columns according to keep_columns if specified
        2. Translates the raw data using Gluroo-specific data translation
        3. Applies the preprocessing pipeline for cleaning and feature engineering

        The preprocessing pipeline handles:
        - Data cleaning and regularization of time intervals
        - Physiological feature engineering (carbs-on-board, insulin-on-board)
        - Addition of availability columns for carbs and insulin

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

        raw = self._clean_and_format_raw_data(raw)
        to_return = preprocessing_pipeline(raw)
        return to_return

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

    def _clean_and_format_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Translate raw Gluroo data to standardized format.

        Converts blood glucose from mg/dL to mmol/L, standardizes datetime column,
        and adds patient identifier. This is a wrapper around data_translation().

        Args:
            raw_data (pd.DataFrame): Raw Gluroo data with original column names.

        Returns:
            pd.DataFrame: Translated data with standardized format.
        """
        return data_translation(raw_data)

    def _get_day_splits(
        self,
        patient_data: pd.DataFrame,
        context_period: tuple[int, int] = (6, 24),
        forecast_horizon: tuple[int, int] = (0, 6),
    ):
        """
        Split each day's data into context period and forecast horizon.

        This method implements the actual day-by-day splitting logic used by
        get_validation_day_splits. By default, it divides each day's data into a daytime
        portion (6am-midnight) for context and the following overnight period (midnight-6am)
        for forecasting.

        Args:
            patient_data (pd.DataFrame): Data for a single patient, must include
                                        'datetime' column or DatetimeIndex.
            context_period (tuple[int, int]): Start and end hours for context period.
                Default: (6, 24) for 6am-midnight.
            forecast_horizon (tuple[int, int]): Start and end hours for forecast period.
                Default: (0, 6) for midnight-6am next day.

        Yields:
            tuple: (context_data, forecast_data) where
                  context_data is from the context period of the current day
                  forecast_data is from the forecast horizon of the following day
        """
        yield from iter_daily_context_forecast_splits(
            patient_data,
            context_period=context_period,
            forecast_horizon=forecast_horizon,
        )
