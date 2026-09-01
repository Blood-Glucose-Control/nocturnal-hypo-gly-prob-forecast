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
from ..dataset_base import DatasetBase, ProcessedPatientDataFrames
from .data_cleaner import (
    DATA_DIR,
    clean_dataset_data,
    load_raw_dataset_data,
    process_single_patient_data,
)

logger = logging.getLogger(__name__)


class Brown2019DataLoader(DatasetBase):
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
            parallel: Whether patient processing should run in parallel.
            max_workers: Maximum worker count for parallel processing.

        Side Effects:
            Initializes cache/dataset configuration attributes and immediately
            calls load_data() to populate processed_data.
        """
        super().__init__()
        self.use_cached = use_cached
        self.keep_columns = keep_columns
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)
        self.processed_data = None

        # Data Objects
        self.raw_data = None
        self.processed_data = None

        # Load data on init
        logger.info(
            "Initializing %s with use_cached=%s.",
            self.__class__.__name__,
            self.use_cached,
        )
        self.load_data()

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

    # ==================== Protected Methods ====================

    def _process_and_cache_data(self) -> ProcessedPatientDataFrames:
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

    def _process_raw_data(self) -> ProcessedPatientDataFrames:
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
