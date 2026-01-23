# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

from src.data.diabetes_datasets.aleppo.preprocess import create_aleppo_csv
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.cache_manager import get_cache_manager

# from src.data.data_models import Dataset
from src.data.dataset_configs import DatasetConfig, get_dataset_config
from src.data.preprocessing.time_processing import (
    get_train_validation_split_by_percentage,
)
from .data_cleaner import clean_all_patients
import pandas as pd
import logging


logger = logging.getLogger(__name__)
# There are 226 unique ids in the database but the study mentioned 225 participants only.
# Maybe 81 is not counted as it doesn't have enough data
# This is for the interim folder
PATIENT_COUNT = 226


# TODO: ISF/CR is not dropped in the dataset. We could use this to calculate slope of the glucose curve.
# to give models some hints about trend of the glucose curve.
class AleppoDataLoader(DatasetBase):
    def __init__(
        self,
        keep_columns: list[str] | None = None,
        dataset_type: str = "train",
        num_validation_days: int = 20,
        use_cached: bool = True,
        train_percentage: float = 0.9,
        config: dict | None = None,
        parallel: bool = True,
        max_workers: int = 10,
    ):
        """
        Args:
            keep_columns (list): List of columns to keep from the raw data.
            train_percentage (float): Percentage of the data to use for training.
            use_cached (bool): Whether to use cached data. WARNING: Processing data takes a VERY LONG TIME.
        """
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.train_percentage = train_percentage
        self.dataset_type = dataset_type
        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)
        self.raw_data_path = None
        self.use_cached = use_cached
        self.parallel = parallel
        self.max_workers = max_workers

        logger.info(
            f"Initializing AleppoDataLoader with use_cached={use_cached} and dataset_type={dataset_type}"
        )
        logger.info(
            f"Currently not used: dataset_type: {dataset_type}, config: {config}"
        )
        self.load_data()

    @property
    def dataset_name(self):
        # return Dataset.ALEPPO.value
        return "aleppo"

    @property
    def description(self):
        return """
                The purpose of this study was to determine whether the use of continuous glucose monitoring (CGM) without blood glucose monitoring (BGM) measurements is as safe and effective as using CGM with BGM in adults (25-40) with type 1 diabetes.
                The total sample size was 225 participants. The Dexcom G4 was used to continuously monitor glucose levels for a span of 6 months.
           """

    def load_data(self) -> None:
        """
        Load the raw data, process data and split it into train and validation.
        If the dataset is not cached, the function will process the raw data and save it to the cache.

        Side Effects:
            Sets self.processed_data, self.train_data, and self.validation_data.
        """
        need_to_process_data = True
        if self.use_cached:
            cached_data = self.cache_manager.load_processed_data(
                self.dataset_name, file_format="csv"
            )
            if cached_data is not None:
                self.processed_data = cached_data
                need_to_process_data = False
        if need_to_process_data:
            self._process_and_cache_data()

        self.train_data, self.validation_data = self._split_train_validation()

    def load_raw(self):
        """
        Raw data of this dataset is not loadable (not in csv format). So we only check if the raw data exists.
        If not we throw an error and give instructions to the user on how to download the data and place it in the correct cache directory.
        """
        self.raw_data_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )

    def _process_and_cache_data(self):
        """
        We don't have the processed data cached so we need to load raw data then process it and save it to the cache.
        """
        # This will guarantee the raw data exists or throw an error if it does not.
        self.load_raw()
        self.processed_data = self._process_raw_data()

    # TODO: Maybe we don't need interim folder. Just process from the query to processed data directly?
    def _process_raw_data(self) -> dict[str, pd.DataFrame]:
        """
        1.Transform the raw data from text to csv by patients (saved to interim folder)
        2.Do the processing on the csv files.
        3.Save the processed data to the cache.
        """

        processed_path = self.cache_manager.get_processed_data_path(self.dataset_name)
        processed_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Create parent directory

        interim_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "interim"
        )

        # Raw -> interim ({pid}_full.csv)
        interim_csvs = list(interim_path.glob("*.csv"))
        if len(interim_csvs) != PATIENT_COUNT:
            logger.warning(
                f"Interim folder contains {len(interim_csvs)} CSV files, expected {PATIENT_COUNT}. Recreating interim folder."
            )
            interim_csvs = []
        if not interim_csvs:
            if self.raw_data_path is None:
                raise ValueError(
                    "Raw data path is not set. Please call load_raw() first."
                )
            create_aleppo_csv(self.raw_data_path)

        # interim -> processed ({pid}_full.csv)
        logger.info(
            f"Cleaning all patients from {interim_path} to {processed_path} with parallel={self.parallel} and max_workers={self.max_workers}"
        )
        # clean and save
        return clean_all_patients(
            interim_path,
            processed_path,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )

    def _split_train_validation(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Split the processed data into train and validation dicts per patient using train_percentage.
        If a patient has less than 2 days of data, it will be skipped.
        """
        train_dict: dict[str, pd.DataFrame] = {}
        val_dict: dict[str, pd.DataFrame] = {}

        for patient_id, df in self.processed_data.items():
            records = 0
            span_days = 0
            try:
                patient_df = df.copy()

                # Ensure DatetimeIndex as required by the splitter
                if isinstance(patient_df.index, pd.DatetimeIndex):
                    patient_df = patient_df.sort_index()
                else:
                    if "datetime" in patient_df.columns:
                        patient_df = patient_df.sort_values("datetime").set_index(
                            "datetime"
                        )
                    else:
                        logger.warning(
                            f"Patient {patient_id} skipped: missing 'datetime' column; records={len(patient_df)}"
                        )
                        continue

                # Basic span/records info for logging
                records = len(patient_df)
                span_days = max(
                    0, (patient_df.index.max() - patient_df.index.min()).days
                )

                # Attempt split
                train_df, val_df, _ = get_train_validation_split_by_percentage(
                    patient_df, train_percentage=self.train_percentage
                )

                train_dict[patient_id] = train_df
                val_dict[patient_id] = val_df

            except (ValueError, TypeError) as e:
                # p81 should be the only patient with insufficient data.
                logger.warning(
                    f"Patient {patient_id} skipped due to insufficient/invalid data: "
                    f"{e}; records={records}, span_days={span_days}"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"Patient {patient_id} skipped due to unexpected error: {e}; "
                    f"records={records if 'records' in locals() else 'unknown'}"
                )
                continue

        return train_dict, val_dict
