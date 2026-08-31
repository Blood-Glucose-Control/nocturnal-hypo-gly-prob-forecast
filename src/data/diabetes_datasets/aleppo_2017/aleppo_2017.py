# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
DataLoader for the Aleppo 2017 REPLACE-BG dataset.

Study: REPLACE-BG - CGM with vs without routine blood glucose monitoring
- 226 total participants
- 149 in CGM-only arm
- 77 in CGM + BGM control arm

Data Sources:
- Data Tables/* - source trial tables converted to per-patient CSV files
"""

import logging

import pandas as pd

from ...cache_manager import get_cache_manager
from ...dataset_configs import DatasetConfig, get_dataset_config
from ...models import DatasetSourceType
from ..dataset_base import DatasetBase
from .data_cleaner import clean_dataset_data
from .preprocess import create_aleppo_csv

logger = logging.getLogger(__name__)


class Aleppo2017DataLoader(DatasetBase[dict[str, pd.DataFrame]]):
    """Data loader for the Aleppo 2017 (REPLACE-BG) CGM dataset.

    This class handles loading, processing, and caching of the Aleppo 2017
    dataset, which contains continuous glucose monitoring data from the
    REPLACE-BG randomized trial comparing CGM with and without routine blood
    glucose monitoring in adults with well-controlled Type 1 diabetes.

    The study evaluated whether CGM without confirmatory blood glucose
    monitoring (BGM) is as safe and effective as using CGM adjunctive to BGM.

    Key features of this dataset:
        - n = 226 participants (149 CGM-only, 77 CGM + BGM control)
        - 6-month study duration using Dexcom G4 CGM
        - CGM data from adults with well-controlled T1D
        - Useful for comparing CGM-only vs CGM+BGM treatment approaches

    Attributes:
        keep_columns: Specific columns to load from the dataset.
        use_cached: Whether to use cached processed data if available.
        train_percentage: Percentage of data to use for training.
        parallel: Whether to use parallel processing.
        max_workers: Maximum number of workers for parallel processing.
        config: Optional configuration dictionary.

    Example:
        >>> loader = Aleppo2017DataLoader(use_cached=True)
        >>> pretraining_dict = loader.processed_data
    """

    def __init__(
        self,
        # Data Selection
        keep_columns: list[str] | None = None,
        # Caching
        use_cached: bool = True,
        # Train/Validation Splitting
        train_percentage: float = 0.9,
        # Parallel Processing
        parallel: bool = True,
        max_workers: int = 14,
        # Date Normalization (if applicable)
        # Dataset-Specific Parameters
    ):
        """Initialize the Aleppo 2017 data loader.

        Args:
            keep_columns: Optional list of columns to retain per patient.
            use_cached: Whether to load cached processed data when available.
            train_percentage: Fraction of each patient's timeline used for training.
            parallel: Whether patient processing should run in parallel.
            max_workers: Maximum worker count for parallel processing.

        Side Effects:
            Initializes cache/dataset configuration attributes and immediately
            calls load_data() to populate processed_data, train_data, and
            validation_data.
        """
        super().__init__()
        self.keep_columns = keep_columns
        self.train_percentage = train_percentage
        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)
        self.raw_data_path = None
        self.use_cached = use_cached
        self.parallel = parallel
        self.max_workers = max_workers
        self.processed_data = None
        self.train_data = None
        self.validation_data = None

        logger.info(
            "Initializing %s with use_cached=%s.",
            self.__class__.__name__,
            self.use_cached,
        )
        self.load_data()

    # ==================== Properties ====================
    @property
    def dataset_name(self) -> str:
        return DatasetSourceType.ALEPPO_2017.value

    @property
    def description(self) -> str:
        return """
                Objective: 'To determine whether the use of continuous glucose monitoring (CGM) without confirmatory
                    blood glucose monitoring (BGM) measurements is as safe and effective as using CGM adjunctive to
                    BGM in adults with well-controlled type 1 diabetes (T1D).'
                Title: 'REPLACE-BG: A Randomized Trial Comparing Continuous Glucose Monitoring With and Without
                    Routine Blood Glucose Monitoring in Adults With Well-Controlled Type 1 Diabetes'
                n = 226 participants
                    - 149 CGM-only
                    - 77 CGM + BGM (control)
                Duration: 6 months
                Paper: https://diabetesjournals.org/care/article-abstract/40/4/538/3687/REPLACE-BG-A-Randomized-Trial-Comparing-Continuous?redirectedFrom=fulltext
                Notes: The Dexcom G4 was used to continuously monitor glucose levels for a span of 6 months.
            """

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

    def load_raw(self):
        """
        Raw data of this dataset is not loadable (not in csv format). So we only check if the raw data exists.
        If not we throw an error and give instructions to the user on how to download the data and place it in the correct cache directory.
        """
        self.raw_data_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )

    # ==================== Protected Methods ====================

    def _process_and_cache_data(self) -> dict[str, pd.DataFrame]:
        """
        We don't have the processed data cached so we need to load raw data then process it and save it to the cache.
        """
        # This will guarantee the raw data exists or throw an error if it does not.
        self.load_raw()
        self.processed_data = self._process_raw_data()
        return self.processed_data

    # TODO: Maybe we don't need interim folder. Just process from the query to processed data directly?
    def _process_raw_data(self) -> dict[str, pd.DataFrame]:
        """
        1.Transform the raw data from text to csv by patients (saved to interim folder)
        2.Do the processing on the csv files.
        3.Save the processed data to the cache.
        """

        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Create parent directory

        interim_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "interim"
        )

        # Raw -> interim ({pid}_full.csv)
        interim_csvs = list(interim_path.glob("*.csv"))

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
        return clean_dataset_data(
            interim_path,
            processed_path,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )
