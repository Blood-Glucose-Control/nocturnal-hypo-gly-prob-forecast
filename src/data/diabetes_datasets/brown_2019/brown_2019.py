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
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.data.cache_manager import get_cache_manager
from src.data.dataset_configs import DatasetConfig, get_dataset_config
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.models import ColumnNames, DatasetSourceType
from src.data.preprocessing.pipeline import preprocessing_pipeline
from src.data.preprocessing.time_processing import (
    get_train_validation_split_by_percentage,
)

from src.data.diabetes_datasets.brown_2019.data_cleaner import (
    DATA_DIR,
    clean_brown_2019_data,
    load_raw_brown_2019_data,
)

logger = logging.getLogger(__name__)


def _process_single_patient(args: tuple) -> tuple[str, pd.DataFrame]:
    """
    Process a single patient's data through the preprocessing pipeline.

    This is a module-level function (required for pickling in ProcessPoolExecutor).

    Args:
        args: Tuple of (patient_id, patient_df, use_aggregation, basal_delivery_type)

    Returns:
        Tuple of (patient_id_str, processed_df)
    """
    patient_id, patient_df, use_aggregation, basal_delivery_type = args

    # Preserve original bolus before basal is added to dose_units
    patient_df[ColumnNames.BOLUS.value] = patient_df[
        ColumnNames.DOSE_UNITS.value
    ].copy()

    # Run preprocessing pipeline (basal rollover, IOB/COB calculation)
    try:
        patient_df = preprocessing_pipeline(
            str(patient_id),
            patient_df,
            use_aggregation=use_aggregation,
            basal_delivery_type=basal_delivery_type,
        )
    except Exception as e:
        logger.warning(
            f"Patient {patient_id} preprocessing failed: {e}. Using cleaned data."
        )

    return str(int(patient_id)), patient_df


class Brown2019DataLoader(DatasetBase):
    """
    DataLoader for Brown 2019 DCLP3 dataset.

    This loader handles the complete pipeline:
    1. Load raw data (CGM, basal, bolus)
    2. Clean and merge into single DataFrame
    3. Run preprocessing pipeline for IOB calculation
    4. Split into train/validation sets

    Args:
        use_cached: Use cached processed data if available.
        train_percentage: Percentage of data for training (0.0-1.0).
        keep_columns: List of columns to keep (None = keep all).
        parallel: Use parallel processing for patient preprocessing.
        max_workers: Number of parallel workers (default 3).

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
        keep_columns: list[str] | None = None,
        parallel: bool = True,
        max_workers: int = 3,
    ):
        super().__init__()
        self.use_cached = use_cached
        self.train_percentage = train_percentage
        self.keep_columns = keep_columns
        self.parallel = parallel
        self.max_workers = max_workers

        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)

        # Will be populated by load_data()
        self.train_data: dict[str, pd.DataFrame] = {}
        self.validation_data: dict[str, pd.DataFrame] = {}

        # Metadata tracking
        self.data_metrics: dict = {}
        self.train_dt_col_type: str | None = None
        self.val_dt_col_type: str | None = None
        self.num_train_days: int | None = None

        # Load data on init
        self.load_data()

    @property
    def dataset_name(self) -> str:
        """Return the dataset name.

        Returns:
            str: The name identifier for this dataset ('brown_2019').
        """
        return DatasetSourceType.BROWN_2019.value

    @property
    def description(self) -> str:
        """Return a description of the dataset.

        Returns:
            str: Human-readable description of the Brown 2019 DCLP3 study.
        """
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
            cached_data = self.cache_manager.load_full_processed_data(self.dataset_name)
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} patients from cache")
                self.processed_data = cached_data
                need_to_process = False

        if need_to_process:
            self._process_and_cache_data()

        # Split into train/validation
        self.train_data, self.validation_data = self._split_train_validation()

        # Compute validation metrics
        self._validate_dataset()

        return self.processed_data

    def _process_and_cache_data(self):
        """
        Process raw data and save to cache.
        """
        logger.info("Processing Brown 2019 raw data...")

        # Load and clean raw data
        self.processed_data = self._process_raw_data()

        # Save to cache using cache manager's paired save/load methods
        self.cache_manager.save_full_processed_data(
            self.dataset_name, self.processed_data
        )

        logger.info(f"Cached {len(self.processed_data)} patients")

    def _process_raw_data(self) -> dict[str, pd.DataFrame]:
        """
        Process raw data into cleaned, per-patient DataFrames.

        Uses parallel processing if self.parallel=True.

        Returns:
            Dict mapping patient_id -> DataFrame.
        """
        # Load raw data
        cgm_df, basal_df, bolus_df = self.load_raw()

        # Clean and merge
        cleaned_df = clean_brown_2019_data(cgm_df, basal_df, bolus_df)

        # Prepare patient data tuples for processing
        # Brown 2019 uses Control-IQ (automated basal) - rate persists until next change
        # use_aggregation=False because data_cleaner already produces regularized 5-min data
        patient_tuples = [
            (patient_id, group.copy(), False, "automated")
            for patient_id, group in cleaned_df.groupby(ColumnNames.P_NUM.value)
        ]

        patient_dict = {}

        if self.parallel and len(patient_tuples) > 1:
            # Parallel processing
            logger.info(
                f"Processing {len(patient_tuples)} patients in parallel "
                f"(max_workers={self.max_workers})"
            )
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(_process_single_patient, pt): pt[0]
                    for pt in patient_tuples
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing patients",
                ):
                    patient_id = futures[future]
                    try:
                        pid, patient_df = future.result()

                        # Filter columns if requested
                        if self.keep_columns is not None:
                            available_cols = [
                                c for c in self.keep_columns if c in patient_df.columns
                            ]
                            patient_df = patient_df[available_cols]

                        patient_dict[pid] = patient_df
                    except Exception as e:
                        logger.error(f"Patient {patient_id} failed: {e}")
        else:
            # Sequential processing (for debugging or single patient)
            logger.info(f"Processing {len(patient_tuples)} patients sequentially")
            for patient_tuple in tqdm(patient_tuples, desc="Processing patients"):
                try:
                    pid, patient_df = _process_single_patient(patient_tuple)

                    # Filter columns if requested
                    if self.keep_columns is not None:
                        available_cols = [
                            c for c in self.keep_columns if c in patient_df.columns
                        ]
                        patient_df = patient_df[available_cols]

                    patient_dict[pid] = patient_df
                except Exception as e:
                    logger.error(f"Patient {patient_tuple[0]} failed: {e}")

        logger.info(f"Processed {len(patient_dict)} patients")
        return patient_dict

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

        # Track metadata
        if train_dict:
            first_train = next(iter(train_dict.values()))
            self.train_dt_col_type = str(first_train.index.dtype)
            self.num_train_days = sum(
                pd.DatetimeIndex(df.index).normalize().nunique()
                for df in train_dict.values()
            )
        if val_dict:
            first_val = next(iter(val_dict.values()))
            self.val_dt_col_type = str(first_val.index.dtype)

        return train_dict, val_dict

    def _validate_dataset(self) -> None:
        """Compute and store validation metrics for the dataset."""
        if not self.processed_data:
            self.data_metrics = {}
            return

        # Combine all data for statistics
        all_data = pd.concat(self.processed_data.values())

        self.data_metrics = {
            "total_rows": len(all_data),
            "unique_patients": len(self.processed_data),
            "patients_with_insulin": sum(
                1
                for df in self.processed_data.values()
                if ColumnNames.IOB.value in df.columns
                and df[ColumnNames.IOB.value].notna().any()
            ),
            "patients_cgm_only": sum(
                1
                for df in self.processed_data.values()
                if ColumnNames.IOB.value not in df.columns
                or df[ColumnNames.IOB.value].isna().all()
            ),
        }

        # Glucose statistics
        if ColumnNames.BG.value in all_data.columns:
            bg = all_data[ColumnNames.BG.value].dropna()
            self.data_metrics.update(
                {
                    "glucose_mean_mmol": round(bg.mean(), 2),
                    "glucose_std_mmol": round(bg.std(), 2),
                    "glucose_min_mmol": round(bg.min(), 2),
                    "glucose_max_mmol": round(bg.max(), 2),
                }
            )

        logger.info(f"Dataset validation: {self.data_metrics}")

    @property
    def num_patients(self) -> int:
        """Get the number of patients in the dataset.

        Returns:
            int: The count of patients, or 0 if no data is loaded.
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
                num_validation_patients, and metrics.
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
        if self.data_metrics:
            info["metrics"] = self.data_metrics
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


if __name__ == "__main__":
    # Example usage / testing
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Brown2019DataLoader ===\n")

    # Test with fresh processing (no cache)
    loader = Brown2019DataLoader(
        use_cached=False,
        train_percentage=0.9,
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
