# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

from src.data.models import DatasetSourceType
from src.data.diabetes_datasets.aleppo_2017.preprocess import create_aleppo_csv
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
class Aleppo2017DataLoader(DatasetBase):
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
        return DatasetSourceType.ALEPPO_2017.value

    @property
    def description(self):
        return """
                Objective: 'To determine whether the use of continuous glucose monitoring (CGM) without confirmatory
                    blood glucose monitoring (BGM) measurements is as safe and effective as using CGM adjunctive to
                    BGM in adults with well-controlled type 1 diabetes (T1D).'
                Title: 'REPLACE-BG: A Randomized Trial Comparing Continuous Glucose Monitoring With and Without
                    Routine Blood Glucose Monitoring in Adults With Well-Controlled Type 1 Diabetes'
                n = 226 participants.
                    - 149 CGM-only
                    - 77 CGM + BGM (control)
                Paper: https://diabetesjournals.org/care/article-abstract/40/4/538/3687/REPLACE-BG-A-Randomized-Trial-Comparing-Continuous?redirectedFrom=fulltext
                Notes: The Dexcom G4 was used to continuously monitor glucose levels for a span of 6 months.
            """

    @property
    def num_patients(self) -> int:
        """Get the number of patients in the dataset.

        Returns:
            int: The count of processed patients, or 0 if no data is loaded.
        """
        return len(self.processed_data) if self.processed_data else 0

    @property
    def patient_ids(self) -> list[str]:
        """Get list of patient IDs in the dataset.

        Returns:
            list[str]: List of patient ID strings, or empty list if no data.
        """
        return list(self.processed_data.keys()) if self.processed_data else []

    @property
    def train_data_shape_summary(self) -> dict[str, tuple[int, int]]:
        """Get shape summary for each patient's training data.

        Returns:
            dict[str, tuple[int, int]]: Dictionary mapping patient IDs to their
                DataFrame shape as (num_rows, num_columns). Returns empty dict
                if train_data is not available.
        """
        if not self.train_data:
            return {}
        return {
            patient_id: df.shape
            for patient_id, df in self.train_data.items()
            if isinstance(df, pd.DataFrame)
        }

    @property
    def dataset_info(self) -> dict[str, object]:
        """Get comprehensive information about the dataset.

        Returns:
            dict[str, object]: Dictionary containing dataset statistics and metadata
                including dataset_name, num_patients, patient_ids, train_percentage,
                parallel, max_workers, and optionally train_shapes, num_train_patients,
                and num_validation_patients.
        """
        info = {
            "dataset_name": self.dataset_name,
            "num_patients": self.num_patients,
            "patient_ids": self.patient_ids,
            "train_percentage": self.train_percentage,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
        }
        if self.train_data:
            info["train_shapes"] = self.train_data_shape_summary
            info["num_train_patients"] = len(self.train_data)
        if self.validation_data:
            info["num_validation_patients"] = len(self.validation_data)
        return info

    # ==================== Public Methods ====================

    def get_patient_data(self, patient_id: str) -> pd.DataFrame | None:
        """Get processed data for a specific patient.

        Args:
            patient_id: Patient identifier string.

        Returns:
            DataFrame for the patient, or None if not found.
        """
        if not self.processed_data:
            return None
        return self.processed_data.get(patient_id)

    def get_combined_data(self, data_type: str = "all") -> pd.DataFrame:
        """Combine all patients' data into a single DataFrame.

        Args:
            data_type: One of 'all', 'train', 'validation'.

        Returns:
            Combined DataFrame with patient data indexed by (patient_id, datetime).

        Raises:
            ValueError: If invalid data_type or no data available.
        """
        if data_type == "all":
            data_dict = self.processed_data
        elif data_type == "train":
            data_dict = self.train_data
        elif data_type == "validation":
            data_dict = self.validation_data
        else:
            raise ValueError(
                f"Invalid data_type: {data_type}. Use 'all', 'train', or 'validation'."
            )

        if not data_dict:
            raise ValueError(f"No {data_type} data available.")

        return pd.concat(
            data_dict.values(), keys=data_dict.keys(), names=["patient_id"]
        )

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
