# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
DataLoader for the Brown 2019 DCLP3 dataset.

Study: DCLP3 - Closed-Loop Control vs Sensor-Augmented Pump therapy
- 168 total patients
- 125 have insulin pump data (basal + bolus)
- 43 have CGM only (no pump data)

Data Sources:
- cgm.txt - CGM readings (~9M rows)
- Pump_BasalRateChange.txt - Basal rate changes (~2.6M rows)
- Pump_BolusDelivered.txt - Bolus deliveries (~221K rows)
"""

import logging

import pandas as pd

from src.data.cache_manager import get_cache_manager
from src.data.dataset_configs import DatasetConfig, get_dataset_config
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.models import ColumnNames, DatasetSourceType
from src.data.preprocessing.time_processing import (
    get_train_validation_split_by_percentage,
)

from .data_cleaner import (
    DATA_DIR,
    clean_brown_2019_data,
    load_raw_brown_2019_data,
    process_single_patient,
)

logger = logging.getLogger(__name__)

# Total patients in dataset
TOTAL_PATIENTS = 168
PATIENTS_WITH_PUMP = 125
PATIENTS_WITHOUT_PUMP = 43


class Brown2019DataLoader(DatasetBase):
    """
    DataLoader for Brown 2019 DCLP3 dataset.

    This loader handles the complete pipeline:
    1. Load raw data (CGM, basal, bolus)
    2. Clean and merge into single DataFrame
    3. Optionally run through preprocessing pipeline for IOB
    4. Split into train/validation sets

    Args:
        use_cached: Use cached processed data if available.
        train_percentage: Percentage of data for training (0.0-1.0).
        include_patients_without_pump: Include 43 patients with CGM only.
        run_preprocessing_pipeline: Run preprocessing_pipeline for IOB calculation.
        keep_columns: List of columns to keep (None = keep all).

    Example:
        ```python
        loader = Brown2019DataLoader(use_cached=True)
        train_data = loader.train_data  # dict[patient_id, DataFrame]
        validation_data = loader.validation_data
        ```
    """

    def __init__(
        self,
        use_cached: bool = True,
        train_percentage: float = 0.9,
        include_patients_without_pump: bool = True,
        run_preprocessing_pipeline: bool = True,
        keep_columns: list[str] | None = None,
    ):
        super().__init__()
        self.use_cached = use_cached
        self.train_percentage = train_percentage
        self.include_patients_without_pump = include_patients_without_pump
        self.run_preprocessing_pipeline = run_preprocessing_pipeline
        self.keep_columns = keep_columns

        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)

        # Will be populated by load_data()
        self.train_data: dict[str, pd.DataFrame] = {}
        self.validation_data: dict[str, pd.DataFrame] = {}

        # Load data on init
        self.load_data()

    @property
    def dataset_name(self) -> str:
        return DatasetSourceType.BROWN_2019.value

    @property
    def description(self) -> str:
        return """
        Brown 2019 DCLP3 Study: A randomized trial comparing Closed-Loop Control (Control-IQ)
        vs Sensor-Augmented Pump therapy in adults with Type 1 diabetes.

        - 168 participants total
        - 125 have insulin pump data (basal rate changes + bolus deliveries)
        - 43 have CGM only (no pump data)
        - Data spans ~6 months per patient (Baseline + Post Randomization periods)
        - CGM: Dexcom G6, 5-minute intervals
        """

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw data files (CGM, basal, bolus).

        Returns:
            Tuple of (cgm_df, basal_df, bolus_df).

        Raises:
            FileNotFoundError: If raw data directory doesn't exist.
        """
        # Ensure raw data exists
        self.cache_manager.ensure_raw_data(self.dataset_name, self.dataset_config)

        return load_raw_brown_2019_data(DATA_DIR)

    def load_data(self) -> dict[str, pd.DataFrame]:
        """
        Load and process the dataset.

        If cached data exists and use_cached=True, loads from cache.
        Otherwise, processes raw data and saves to cache.

        Returns:
            Dict mapping patient_id -> DataFrame.
        """
        need_to_process = True

        if self.use_cached:
            cached_data = self.cache_manager.load_processed_data(
                self.dataset_name, file_format="parquet"
            )
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} patients from cache")
                self.processed_data = cached_data
                need_to_process = False

        if need_to_process:
            self._process_and_cache_data()

        # Filter out patients without pump data if requested
        if not self.include_patients_without_pump:
            self._filter_patients_without_pump()

        # Split into train/validation
        self.train_data, self.validation_data = self._split_train_validation()

        return self.processed_data

    def _process_and_cache_data(self):
        """
        Process raw data and save to cache.
        """
        logger.info("Processing Brown 2019 raw data...")

        # Load and clean raw data
        self.processed_data = self._process_raw_data()

        # Save to cache
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        # Save each patient as separate parquet file
        for patient_id, patient_df in self.processed_data.items():
            patient_path = processed_path / f"{patient_id}.parquet"
            patient_df.to_parquet(patient_path)

        logger.info(f"Cached {len(self.processed_data)} patients to {processed_path}")

    def _process_raw_data(self) -> dict[str, pd.DataFrame]:
        """
        Process raw data into cleaned, per-patient DataFrames.

        Returns:
            Dict mapping patient_id -> DataFrame.
        """
        # Load raw data
        cgm_df, basal_df, bolus_df = self.load_raw()

        # Clean and merge
        cleaned_df = clean_brown_2019_data(cgm_df, basal_df, bolus_df)

        # Split into per-patient dict
        patient_dict = {}
        for patient_id, group in cleaned_df.groupby(ColumnNames.P_NUM.value):
            patient_df = group.drop(columns=[ColumnNames.P_NUM.value])

            # Optionally run through preprocessing pipeline for IOB
            if self.run_preprocessing_pipeline:
                try:
                    patient_df = process_single_patient(patient_df, str(patient_id))
                except Exception as e:
                    logger.warning(
                        f"Patient {patient_id} preprocessing failed: {e}. Using cleaned data."
                    )

            # Filter columns if requested
            if self.keep_columns is not None:
                available_cols = [
                    c for c in self.keep_columns if c in patient_df.columns
                ]
                patient_df = patient_df[available_cols]

            patient_dict[str(int(patient_id))] = patient_df

        logger.info(f"Processed {len(patient_dict)} patients")
        return patient_dict

    def _filter_patients_without_pump(self):
        """
        Remove patients who have no pump data (all NaN for basal rate).
        """
        filtered = {}
        removed = 0

        for patient_id, patient_df in self.processed_data.items():
            if ColumnNames.RATE.value in patient_df.columns:
                if patient_df[ColumnNames.RATE.value].notna().any():
                    filtered[patient_id] = patient_df
                else:
                    removed += 1
            else:
                # If rate column doesn't exist, keep the patient
                filtered[patient_id] = patient_df

        logger.info(f"Filtered out {removed} patients without pump data")
        self.processed_data = filtered

    def _split_train_validation(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Split processed data into train and validation sets.

        Returns:
            Tuple of (train_dict, validation_dict).
        """
        train_dict: dict[str, pd.DataFrame] = {}
        val_dict: dict[str, pd.DataFrame] = {}

        for patient_id, patient_df in self.processed_data.items():
            try:
                # Ensure datetime index
                if not isinstance(patient_df.index, pd.DatetimeIndex):
                    if ColumnNames.DATETIME.value in patient_df.columns:
                        patient_df = patient_df.set_index(ColumnNames.DATETIME.value)
                    else:
                        logger.warning(
                            f"Patient {patient_id} skipped: no datetime index"
                        )
                        continue

                patient_df = patient_df.sort_index()

                # Check minimum data requirement
                span_days = (patient_df.index.max() - patient_df.index.min()).days
                if span_days < 2:
                    logger.warning(
                        f"Patient {patient_id} skipped: only {span_days} days of data"
                    )
                    continue

                # Split
                train_df, val_df, _ = get_train_validation_split_by_percentage(
                    patient_df, train_percentage=self.train_percentage
                )

                train_dict[patient_id] = train_df
                val_dict[patient_id] = val_df

            except (ValueError, TypeError) as e:
                logger.warning(f"Patient {patient_id} split failed: {e}")
                continue

        logger.info(
            f"Split complete: {len(train_dict)} train patients, {len(val_dict)} validation patients"
        )
        return train_dict, val_dict

    @property
    def num_patients(self) -> int:
        """Number of patients in processed data."""
        return len(self.processed_data) if self.processed_data else 0

    @property
    def patient_ids(self) -> list[str]:
        """List of patient IDs in processed data."""
        return list(self.processed_data.keys()) if self.processed_data else []


if __name__ == "__main__":
    # Example usage / testing
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Brown2019DataLoader ===\n")

    # Test with fresh processing (no cache)
    loader = Brown2019DataLoader(
        use_cached=False,
        train_percentage=0.9,
        include_patients_without_pump=True,
        run_preprocessing_pipeline=False,  # Skip for faster testing
    )

    print("\n=== Results ===")
    print(f"Total patients: {loader.num_patients}")
    print(f"Train patients: {len(loader.train_data)}")
    print(f"Validation patients: {len(loader.validation_data)}")

    # Sample patient
    if loader.patient_ids:
        sample_id = loader.patient_ids[0]
        sample_df = loader.processed_data[sample_id]
        print(f"\nSample patient {sample_id}:")
        print(f"  Shape: {sample_df.shape}")
        print(f"  Columns: {list(sample_df.columns)}")
        print(f"  Date range: {sample_df.index.min()} to {sample_df.index.max()}")
