# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
DataLoader for the Tamborlane 2008 DirecNet CGM dataset.

Study: DirecNet/JDRF pediatric CGM randomized trial
- Pediatric Type 1 diabetes cohort (age 8-17 years)
- High-frequency CGM measurements (5-minute intervals)
- Multi-day continuous monitoring per patient

Data Sources:
- DataRTCGM*.csv - raw CGM export files (RecID, PtID, DeviceDate, DeviceTime, GlucoseValue)
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ...cache_manager import get_cache_manager
from ...dataset_configs import get_dataset_config
from ...models import DatasetConfig, DatasetSourceType
from ...preprocessing.data_splitting import split_multipatient_dataframe
from ..dataset_base import DatasetBase, ProcessedPatientDataFrames
from .data_cleaner import (
    clean_dataset_data,
    extract_cgm_features,
    process_single_patient_data,
)

logger = logging.getLogger(__name__)


class Tamborlane2008DataLoader(DatasetBase):
    """Data loader for the Tamborlane 2008 CGM dataset.

    This class handles loading, processing, and caching of the Tamborlane 2008
    dataset, which contains continuous glucose monitoring data from pediatric
    Type 1 diabetes patients collected as part of the DirecNet study.

    The study evaluated the accuracy and safety of CGM in children with T1D,
    providing high-frequency glucose measurements useful for hypoglycemia
    prediction research.

    Key features of this dataset:
        - CGM data from pediatric patients (age 8-17)
        - High-frequency measurements (5-minute intervals)
        - Multi-day continuous monitoring periods
        - Useful for nocturnal hypoglycemia prediction

    Attributes:
        keep_columns: Specific columns to load from the dataset.
        use_cached: Whether to use cached processed data if available.
        parallel: Whether to use parallel processing.
        max_workers: Maximum number of workers for parallel processing.
        generic_patient_start_date: Starting date for all patients.
        extract_features: Whether to extract CGM-specific features.
        raw_data_path: Path to the raw data files.

    Example:
        >>> loader = Tamborlane2008DataLoader(use_cached=True)
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
        generic_patient_start_date: pd.Timestamp = pd.Timestamp("2008-01-01"),
        # Dataset-Specific Parameters
        extract_features: bool = True,
        raw_data_path: str | Path | None = None,
    ):
        """
        Initialize the Tamborlane 2008 data loader.

        Args:
            keep_columns: Optional list of columns to retain per patient.
            use_cached: Whether to load cached processed data when available.
            parallel: Whether patient processing should run in parallel.
            max_workers: Maximum worker count for parallel processing.
            generic_patient_start_date: Synthetic date used when normalizing timestamps.
            extract_features: Whether to compute CGM feature columns.
            raw_data_path: Optional override for the raw data directory.

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
        self.extract_features = extract_features
        self.raw_data_path = Path(raw_data_path) if raw_data_path else None

        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)

        # Data Objects
        self.raw_data = None
        self.processed_data = None

        # Load data on initialization
        logger.info(
            "Initializing %s with use_cached=%s.",
            self.__class__.__name__,
            self.use_cached,
        )
        self.load_data()

    # ==================== Properties ====================
    @property
    def dataset_name(self) -> str:
        return DatasetSourceType.TAMBORLANE_2008.value

    @property
    def description(self) -> str:
        return """
                Objective: 'The value of continuous glucose monitoring in the management of type 1 diabetes mellitus has not been determined. (2008)'
                Title: 'Continuous Glucose Monitoring and Intensive Treatment of Type 1 Diabetes'
                n = 322 participants
                Duration: 6 months
                Paper: https://www.nejm.org/doi/full/10.1056/NEJMoa0805017

            """

    # ==================== Public Methods ====================
    def load_raw(self) -> pd.DataFrame:
        """Load raw CGM data from CSV files.

        Searches for CSV files containing 'DataRTCGM' in the filename within
        the raw data directory and its subdirectories. Handles the CGM data
        format with columns: RecID, PtID, DeviceDate, DeviceTime, GlucoseValue.

        Returns:
            pd.DataFrame: Combined raw data from all found CSV files.

        Raises:
            FileNotFoundError: If the raw data path does not exist or no
                data files are found.
            ValueError: If no data could be loaded from any files.
        """
        # Determine the raw data path
        if self.raw_data_path:
            raw_data_path = self.raw_data_path
        elif self.cache_manager:
            # Use cache manager to get the proper path
            raw_data_path = self.cache_manager.get_absolute_path_by_type(
                self.dataset_name, "raw"
            )
        else:
            # Try default cache location
            raw_data_path = Path("cache/data/tamborlane_2008/raw")
            if not raw_data_path.exists():
                # Try alternative paths
                for alt_path in [
                    Path("./raw"),
                    Path("./data/raw"),
                ]:
                    if alt_path.exists():
                        raw_data_path = alt_path
                        break

        if not raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")

        logger.info(f"Loading raw data from: {raw_data_path}")

        # Search for CSV and Excel files recursively
        csv_files = [
            f for f in raw_data_path.rglob("*.csv") if "DataRTCGM" in f.name
        ]  # Only include relevant files
        all_files = csv_files  # Don't need: + excel_files

        if not all_files:
            raise FileNotFoundError(f"No data files found in {raw_data_path}")

        logger.info(f"Found {len(all_files)} data files")

        # Load and combine all files
        dfs = []
        for file in all_files:
            logger.info(f"Loading {file.name}...")
            try:
                if file.suffix == ".csv":
                    # Try to infer delimiter and encoding
                    try:
                        df = pd.read_csv(file)
                    except UnicodeDecodeError:
                        # Try with different encoding
                        df = pd.read_csv(file, encoding="latin1")
                else:
                    # Excel file
                    df = pd.read_excel(file)

                logger.info(f"  Loaded {len(df)} rows from {file.name}")
                logger.info(
                    f"  Columns: {list(df.columns)[:10]}"
                )  # Show first 10 columns
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load {file.name}: {e}")
                continue

        if not dfs:
            raise ValueError("No data could be loaded from any files")

        # Combine all dataframes
        if len(dfs) > 1:
            combined_df = pd.concat(dfs, ignore_index=True, sort=False)
            logger.info(
                f"Combined {len(dfs)} files into dataset with {len(combined_df)} total rows"
            )
        else:
            combined_df = dfs[0]

        # Log data summary
        logger.info(f"Raw data shape: {combined_df.shape}")
        logger.info(f"Raw data columns: {list(combined_df.columns)}")

        # Check for expected columns
        expected_cols = ["RecID", "PtID", "DeviceDate", "DeviceTime", "GlucoseValue"]
        found_cols = [col for col in expected_cols if col in combined_df.columns]
        missing_cols = [col for col in expected_cols if col not in combined_df.columns]

        if found_cols:
            logger.info(f"Found expected columns: {found_cols}")
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}")

        return combined_df

    # ==================== Protected Methods ====================
    def _process_and_cache_data(self) -> ProcessedPatientDataFrames:
        """Process raw data and cache the results.

        Loads raw data via load_raw(), processes it via _process_raw_data(),
        and stores results in self.processed_data. Caching is handled within
        _process_raw_data().

        Side Effects:
            Sets self.raw_data and self.processed_data.
        """
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()
        return self.processed_data

    def _process_raw_data(self) -> ProcessedPatientDataFrames:
        """Process raw data with cleaning and feature extraction.

        Cleans the raw CGM data, splits by patient, processes each patient's
        data (in parallel or sequentially based on self.parallel), extracts
        CGM features if enabled, and caches the results.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their
                processed DataFrames with cleaned glucose values and optional
                extracted features.

        Raises:
            AssertionError: If raw_data is None (load_raw() not called first).
        """
        assert self.raw_data is not None, "Raw data not loaded. Call load_raw() first."

        logger.info("Processing Tamborlane 2008 raw data...")

        # Clean the data using the updated cleaner
        cleaned_data = clean_dataset_data(self.raw_data)

        # Split by patient
        if "patient_id" not in cleaned_data.columns:
            # If no patient column, treat as single patient
            logger.warning(
                "No patient ID column found, treating as single patient dataset"
            )
            cleaned_data["patient_id"] = "patient_001"

        # Use the splitter function
        multipatient_data_dict = split_multipatient_dataframe(
            cleaned_data, "patient_id"
        )
        logger.info(f"Processing {len(multipatient_data_dict)} patients")

        # Log sample of data for each patient
        for patient_id, patient_df in list(multipatient_data_dict.items())[
            :3
        ]:  # First 3 patients
            logger.info(f"Patient {patient_id}: {len(patient_df)} readings")
            if "datetime" in patient_df.columns:
                logger.info(
                    f"  Date range: {patient_df['datetime'].min()} to {patient_df['datetime'].max()}"
                )

        # Process each patient
        if self.parallel:
            processed_results = self._process_patients_parallel(multipatient_data_dict)
        else:
            processed_results = self._process_patients_sequential(
                multipatient_data_dict
            )

        # Extract features if requested
        if self.extract_features:
            logger.info("Extracting CGM-specific features...")
            for patient_id in processed_results:
                processed_results[patient_id] = extract_cgm_features(
                    processed_results[patient_id]
                )

        # Save to cache if cache manager is available
        if self.cache_manager:
            logger.info("Saving processed data to cache...")
            try:
                self.cache_manager.save_full_processed_data(
                    self.dataset_name, processed_results
                )
            except Exception as e:
                logger.warning(f"Failed to save to cache: {e}")

        return processed_results

    def _process_patients_parallel(
        self, multipatient_data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Process multiple patients' data in parallel using ProcessPoolExecutor.

        Args:
            multipatient_data_dict: Dictionary mapping patient IDs to their
                raw DataFrames.

        Returns:
            dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their
                processed DataFrames. Patients that fail processing are logged
                but excluded from the result.
        """
        processed_results = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Prepare data tuples
            patient_data_tuples = [
                (patient_id, patient_df, self.generic_patient_start_date)
                for patient_id, patient_df in multipatient_data_dict.items()
            ]

            # Submit all tasks
            future_to_patient = {
                executor.submit(
                    process_single_patient_data, patient_tuple
                ): patient_tuple[0]
                for patient_tuple in patient_data_tuples
            }

            # Collect results
            for future in as_completed(future_to_patient):
                patient_id = future_to_patient[future]
                try:
                    patient_id, result = future.result()
                    processed_results[patient_id] = result
                    logger.info(f"Successfully processed patient {patient_id}")
                except Exception as exc:
                    logger.error(f"Patient {patient_id} generated an exception: {exc}")

        return processed_results

    def _process_patients_sequential(
        self, multipatient_data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Process multiple patients' data sequentially.

        Args:
            multipatient_data_dict: Dictionary mapping patient IDs to their
                raw DataFrames.

        Returns:
            dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their
                processed DataFrames.
        """
        processed_results = {}

        for patient_id, patient_df in multipatient_data_dict.items():
            logger.info(f"Processing patient {patient_id}...")
            patient_data_tuple = (
                patient_id,
                patient_df,
                self.generic_patient_start_date,
            )
            patient_id, processed_df = process_single_patient_data(patient_data_tuple)
            processed_results[patient_id] = processed_df

        return processed_results
