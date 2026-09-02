# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
DataLoader for the Lynch 2022 IOBP2 RCT dataset.

Study: IOBP2 RCT - insulin-only bionic pancreas pivotal extension
- 440 total participants
- 13-week study duration
- Uses a standardized single per-patient processing path

Data Sources:
- IOBP2 RCT Public Dataset/Data Tables/*.txt - pipe-separated raw study tables
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from ...cache_manager import get_cache_manager
from ...dataset_configs import get_dataset_config
from ...models import DatasetConfig, DatasetSourceType
from ...preprocessing.data_splitting import split_multipatient_dataframe
from ..dataset_base import DatasetBase, ProcessedPatientDataFrames
from .data_cleaner import (
    clean_dataset_data,
    load_raw_dataset_data,
    process_single_patient_data,
)

logger = logging.getLogger(__name__)


class Lynch2022DataLoader(DatasetBase):
    """Data loader for the Lynch 2022 IOBP2 RCT dataset.

    This class handles loading, processing, and caching of the Lynch 2022
    dataset, which contains continuous glucose monitoring data from the IOBP2
    (Insulin-Only Bionic Pancreas Pivotal Trial Extension Study) randomized
    controlled trial.

    The study evaluated a transition from standard-of-care management of Type 1
    diabetes to use of the insulin-only configuration of the iLet® bionic
    pancreas in adults and children (age 6-71 years).

    Key features of this dataset:
        - n = 440 participants using insulin aspart, lispro, or fast-acting aspart
        - 13-week study duration
        - CGM data from both standard-of-care and bionic pancreas periods
        - Standardized to a single per-patient train-style processing path

    Attributes:
        keep_columns: Specific columns to load from the dataset.
        use_cached: Whether to use cached processed data if available.
        parallel: Whether to use parallel processing.
        max_workers: Maximum number of workers for parallel processing.
        generic_patient_start_date: Starting date for all patients.

    Example:
        >>> loader = Lynch2022DataLoader(use_cached=True)
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
        generic_patient_start_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
        # Dataset-Specific Parameters
    ):
        """Initialize the Lynch 2022 data loader.

        Args:
            keep_columns: Optional list of columns to retain per patient.
            use_cached: Whether to load cached processed data when available.
            parallel: Whether patient processing should run in parallel.
            max_workers: Maximum worker count for parallel processing.
            generic_patient_start_date: Synthetic date used when normalizing timestamps.

        Side Effects:
            Initializes cache/dataset configuration attributes and immediately
            calls load_data() to populate processed_data.
        """
        super().__init__()
        self.use_cached = use_cached
        self.keep_columns = keep_columns
        self.parallel = parallel
        self.max_workers = max_workers

        self.generic_patient_start_date = generic_patient_start_date

        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)

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
        return DatasetSourceType.LYNCH_2022.value

    @property
    def description(self) -> str:
        return """
                Objective: 'To evaluate a transition from standard-of-care (SC) management of type 1 diabetes
                    (any insulin delivery method including hybrid closed-loop systems plus real-time continuous
                    glucose monitoring [CGM]) to use of the insulin-only configuration of the iLet® bionic
                    pancreas (BP) in 90 adults and children (age 6–71 years).'
                Title: 'The Insulin-Only Bionic Pancreas Pivotal Trial Extension Study: A Multi-Center Single-Arm
                    Evaluation of the Insulin-Only Configuration of the Bionic Pancreas in Adults and Youth with
                    Type 1 Diabetes'
                n = 440 participants using either insulin aspart, insulin lispro, or fast-acting insulin aspart
                Duration: 13 weeks
                Paper: https://journals.sagepub.com/doi/full/10.1089/dia.2022.0341
            """

    # ==================== Public Methods ====================
    def load_raw(self):
        """Load the raw Lynch dataset from the pipe-separated txt files."""
        raw_data_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "raw"
        )
        txt_base = raw_data_path / "IOBP2 RCT Public Dataset" / "Data Tables"

        if not txt_base.exists():
            raise FileNotFoundError(
                f"Expected txt data tables at {txt_base} but directory does not exist."
            )

        logger.info("Loading Lynch 2022 raw data from %s", txt_base)
        raw_df = load_raw_dataset_data(txt_base)
        logger.info("Loaded Lynch 2022 raw data with shape %s", raw_df.shape)
        return raw_df

    # ==================== Protected Methods ====================

    def _process_and_cache_data(self) -> ProcessedPatientDataFrames:
        """Load raw data, process per patient, and return processed_data."""
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()
        return self.processed_data

    def _process_raw_data(self) -> ProcessedPatientDataFrames:
        """Process raw Lynch 2022 data through the per-patient preprocessing pipeline."""
        store_in_between_data = False
        logger.info("Cleaning Lynch 2022 train data...")
        assert self.raw_data is not None, "Raw data not loaded. Call load_raw() first."
        pre_processed_data = clean_dataset_data(self.raw_data)

        logger.info("Running preprocessing pipeline on Lynch 2022 train data...")

        multipatient_data_dict = split_multipatient_dataframe(
            pre_processed_data, "patient_id"
        )

        patient_data_tuples = [
            (patient_id, patient_df, self.generic_patient_start_date)
            for patient_id, patient_df in multipatient_data_dict.items()
        ]

        if self.parallel:
            logger.info(
                f"Processing {len(patient_data_tuples)} Lynch patients in parallel with {self.max_workers} workers..."
            )
            processed_dict = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_patient_data,
                        patient_tuple,
                        store_in_between_data,
                    ): patient_tuple[0]
                    for patient_tuple in patient_data_tuples
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing Lynch patients",
                    unit="patient",
                ):
                    patient_id = futures[future]
                    try:
                        patient_id, processed_data = future.result()
                        processed_dict[patient_id] = processed_data
                    except Exception as exc:
                        logger.error(
                            f"Lynch patient {patient_id} generated an exception: {exc}"
                        )
                        raise exc
        else:
            logger.info(
                f"Processing {len(patient_data_tuples)} Lynch patients sequentially..."
            )
            processed_dict = {}
            for patient_tuple in tqdm(
                patient_data_tuples, desc="Processing Lynch patients", unit="patient"
            ):
                patient_id, processed_data = process_single_patient_data(
                    patient_tuple, store_in_between_data
                )
                processed_dict[patient_id] = processed_data

        logger.info(f"Processed {len(processed_dict)} Lynch patients successfully")
        logger.info("Saving full processed data to cache...")
        self.cache_manager.save_full_processed_data(self.dataset_name, processed_dict)
        logger.info(
            f"Successfully processed and cached full data for {len(processed_dict)} patients"
        )

        return processed_dict
