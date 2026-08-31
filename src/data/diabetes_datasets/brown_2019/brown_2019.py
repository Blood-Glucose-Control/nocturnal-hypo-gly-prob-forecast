# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

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

from ...cache_manager import get_cache_manager
from ...dataset_configs import DatasetConfig, get_dataset_config
from ...models import ColumnNames, DatasetSourceType
from ..dataset_base import DatasetBase
from .data_cleaner import (
    DATA_DIR,
    clean_dataset_data,
    load_raw_dataset_data,
    process_single_patient_data,
)

logger = logging.getLogger(__name__)


class Brown2019DataLoader(DatasetBase[dict[str, pd.DataFrame]]):
    """Data loader for the Brown 2019 DCLP3 dataset.

    This class handles loading, processing, and caching of the Brown 2019
    dataset from the DCLP3 (Closed-Loop Control vs Sensor-Augmented Pump)
    study comparing closed-loop insulin delivery systems.

    The study evaluated hybrid closed-loop insulin delivery in patients with
    Type 1 diabetes, providing CGM, basal rate, and bolus delivery data.

    Key features of this dataset:
        - n = 168 total patients (125 with pump data, 43 CGM-only)
        - CGM readings with basal and bolus insulin data
        - Suitable for IOB/COB calculation and closed-loop research
        - Multi-source data (CGM, basal rates, bolus deliveries)

    Attributes:
        keep_columns: Specific columns to load from the dataset.
        use_cached: Whether to use cached processed data if available.
        train_percentage: Percentage of data to use for training.
        parallel: Whether to use parallel processing.
        max_workers: Maximum number of workers for parallel processing.

    Example:
        >>> loader = Brown2019DataLoader(use_cached=True)
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
        """Initialize the Brown 2019 data loader.

        Args:
            keep_columns: Optional list of columns to retain per patient.
            use_cached: Whether to load cached processed data when available.
            train_percentage: Fraction of each patient's timeline used for training.
            parallel: Whether patient processing should run in parallel.
            max_workers: Maximum worker count for parallel processing.

        Side Effects:
            Initializes cache/dataset configuration attributes, immediately
            calls load_data() to populate processed_data/train_data/validation_data,
            and computes validation metrics.
        """
        super().__init__()
        self.use_cached = use_cached
        self.train_percentage = train_percentage
        self.keep_columns = keep_columns
        self.parallel = parallel
        self.max_workers = max_workers

        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)
        self.processed_data = None

        # Will be populated by load_data()
        self.train_data: dict[str, pd.DataFrame] = {}
        self.validation_data: dict[str, pd.DataFrame] = {}

        # Metadata tracking
        self.data_metrics: dict = {}
        self.train_dt_col_type: str | None = None
        self.val_dt_col_type: str | None = None
        self.num_train_days: int | None = None

        # Load data on init
        logger.info(
            "Initializing %s with use_cached=%s.",
            self.__class__.__name__,
            self.use_cached,
        )
        self.load_data()
        self._validate_dataset()

    # ==================== Properties ====================
    @property
    def dataset_name(self) -> str:
        return DatasetSourceType.BROWN_2019.value

    @property
    def description(self) -> str:
        return """
                Objective: 'Closed-loop systems that automate insulin
                    delivery may improve glycemic outcomes in patients with type 1 diabetes'
                Title: 'Six-Month Randomized, Multicenter Trial of Closed-Loop Control in Type 1 Diabetes'
                n = 168 participants
                    - 125 have insulin pump data (basal rate changes + bolus deliveries)
                    - 43 have CGM only (no pump data)
                    - CGM: Dexcom G6, 5-minute intervals
                Duration: 6 months (Baseline + Post Randomization periods)
                Paper: https://www.nejm.org/doi/full/10.1056/NEJMoa1907863
                Note  Brown 2019 DCLP3 Study: A randomized trial comparing Closed-Loop Control (Control-IQ)
                    vs Sensor-Augmented Pump therapy in adults with Type 1 diabetes.
            """

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
            info["train_shapes"] = {
                patient_id: patient_df.shape
                for patient_id, patient_df in self.train_data.items()
                if isinstance(patient_df, pd.DataFrame)
            }
            info["num_train_patients"] = len(self.train_data)
        if self.validation_data:
            info["num_validation_patients"] = len(self.validation_data)
        if self.data_metrics:
            info["metrics"] = self.data_metrics
        return info

    # ==================== Public Methods ====================

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

        return load_raw_dataset_data(DATA_DIR)

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

    # ==================== Protected Methods ====================

    def _process_and_cache_data(self) -> dict[str, pd.DataFrame]:
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
        return self.processed_data

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
        cleaned_df = clean_dataset_data(cgm_df, basal_df, bolus_df)

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
                    executor.submit(process_single_patient_data, pt): pt[0]
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
                    pid, patient_df = process_single_patient_data(patient_tuple)

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
