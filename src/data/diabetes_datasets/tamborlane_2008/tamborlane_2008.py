"""
Tamborlane 2008 CGM Dataset Loader.

This module provides functionality to load and process the Tamborlane 2008 CGM dataset.
The dataset contains continuous glucose monitoring data from pediatric patients with
Type 1 diabetes, collected as part of the DirecNet study.

The data format includes columns like RecID, PtID, DeviceDate, DeviceTime, GlucoseValue,
and GlucoseDisplayTime based on the actual CGM export format.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# Import the data cleaner functions
try:
    from .data_cleaner import (
        clean_tamborlane_2008_data,
        process_single_patient_tamborlane,
        extract_cgm_features,
        validate_tamborlane_data,
    )
except ImportError:
    # If relative import fails, try absolute import
    from data_cleaner import (
        clean_tamborlane_2008_data,
        process_single_patient_tamborlane,
        extract_cgm_features,
        validate_tamborlane_data,
    )

# Import other required modules
try:
    from src.data.cache_manager import get_cache_manager
    from src.data.dataset_configs import get_dataset_config
    from src.data.diabetes_datasets.dataset_base import DatasetBase
    from src.data.preprocessing.data_splitting import split_multipatient_dataframe
    from src.data.preprocessing.time_processing import (
        get_train_validation_split_by_percentage,
    )
except ImportError:
    from data.cache_manager import get_cache_manager
    from data.dataset_configs import get_dataset_config
    from data.diabetes_datasets.dataset_base import DatasetBase
    from data.preprocessing.data_splitting import split_multipatient_dataframe
    from data.preprocessing.time_processing import (
        get_train_validation_split_by_percentage,
    )

logger = logging.getLogger(__name__)


class Tamborlane2008DataLoader(DatasetBase):
    """
    Data loader for the Tamborlane 2008 CGM dataset.

    This class handles loading, processing, and caching of the Tamborlane 2008 dataset,
    which contains continuous glucose monitoring data from pediatric Type 1 diabetes patients.

    The data format expects columns such as:
    - RecID: Record identifier
    - PtID: Patient identifier
    - DeviceDate: Date of the CGM reading (format: YYYY-MM-DD)
    - DeviceTime: Time of the CGM reading (format: HH:MM:SS)
    - GlucoseValue: Glucose value in mg/dL
    - GlucoseDisplayTime: Alternative timestamp for the glucose reading

    Key features of this dataset:
    - CGM data from pediatric patients (age 8-17)
    - High-frequency measurements (5-minute intervals)
    - Multi-day continuous monitoring periods
    - Useful for nocturnal hypoglycemia prediction

    Attributes:
        keep_columns (list[str] | None): Specific columns to load from the dataset
        num_validation_days (int): Number of days to use for validation
        use_cached (bool): Whether to use cached processed data if available
        dataset_type (str): Type of dataset ('train' or 'test')
        parallel (bool): Whether to use parallel processing
        generic_patient_start_date (pd.Timestamp): Starting date for all patients
        max_workers (int): Maximum number of workers for parallel processing
        extract_features (bool): Whether to extract CGM-specific features
        raw_data_path (Path): Path to the raw data files
    """

    def __init__(
        self,
        keep_columns: Optional[List[str]] = None,
        num_validation_days: int = 7,  # Shorter validation for CGM data
        train_percentage: float = 0.9,
        use_cached: bool = True,
        dataset_type: str = "train",
        parallel: bool = True,
        generic_patient_start_date: pd.Timestamp = pd.Timestamp("2008-01-01"),
        max_workers: int = 3,
        extract_features: bool = True,
        raw_data_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the Tamborlane 2008 data loader.

        Args:
            keep_columns: Specific columns to load from the dataset
            num_validation_days: Number of days to use for validation (default 7)
            train_percentage: Percentage of data to use for training (default 0.9)
            use_cached: Whether to use cached processed data if available
            dataset_type: Type of dataset to load ('train' or 'test')
            parallel: Whether to use parallel processing
            generic_patient_start_date: Starting date for patients (default 2008-01-01)
            max_workers: Maximum number of workers for parallel processing
            extract_features: Whether to extract CGM-specific features
            raw_data_path: Path to raw data files (optional)
        """
        # Ensure required columns are included
        if keep_columns is not None:
            required_cols = ["datetime", "bg_mM", "p_num", "bg_mg_dl"]
            for col in required_cols:
                if col not in keep_columns:
                    keep_columns.append(col)

        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.train_percentage = train_percentage
        self.use_cached = use_cached
        self.dataset_type = dataset_type
        self.parallel = parallel
        self.generic_patient_start_date = generic_patient_start_date
        self.max_workers = max_workers
        self.extract_features = extract_features
        self.raw_data_path = Path(raw_data_path) if raw_data_path else None

        # Initialize cache manager
        try:
            self.cache_manager = get_cache_manager()
            self.dataset_config = get_dataset_config(self.dataset_name)
        except (ImportError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Cache manager initialization failed: {e}")
            self.cache_manager = None
            self.dataset_config = {}

        # Data containers
        self.raw_data = None
        self.processed_data: Dict[str, pd.DataFrame] | None = {}
        self.train_data = None
        self.validation_data = None
        self.data_metrics = {}

        # Metadata
        self.train_dt_col_type = None
        self.val_dt_col_type = None
        self.num_train_days = None

        # Load data on initialization
        self.load_data()

        # Validate data after loading
        if self.processed_data:
            self._validate_dataset()

    # Properties
    @property
    def dataset_name(self) -> str:
        """Return the dataset name.

        Returns:
            str: The name identifier for this dataset ('tamborlane_2008').
        """
        return "tamborlane_2008"

    @property
    def num_patients(self) -> int:
        """Get the number of patients in the dataset.

        Returns:
            int: The count of patients, or 0 if no data is loaded.
        """
        if self.processed_data is None:
            return 0
        return len(self.processed_data)

    @property
    def patient_ids(self) -> List[str]:
        """Get list of patient IDs in the dataset.

        Returns:
            List[str]: List of patient ID strings, or empty list if no data.
        """
        if self.processed_data is None:
            return []
        return list(self.processed_data.keys())

    @property
    def train_data_shape_summary(self) -> Dict[str, Tuple[int, int]]:
        """Get shape summary for each patient's processed data.

        Returns:
            Dict[str, Tuple[int, int]]: Dictionary mapping patient IDs to their
                DataFrame shape as (num_rows, num_columns). Returns empty dict
                if processed_data is not available.
        """
        if not isinstance(self.processed_data, dict):
            return {}

        shape_summary = {}
        for patient_id, patient_df in self.processed_data.items():
            if isinstance(patient_df, pd.DataFrame):
                shape_summary[patient_id] = patient_df.shape
        return shape_summary

    @property
    def dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the dataset.

        Returns:
            Dict[str, Any]: Dictionary containing dataset statistics and metadata
                including dataset_name, num_patients, patient_ids, dataset_type,
                num_validation_days, extract_features, and optionally train_shapes,
                validation_shapes, and metrics.
        """
        info = {
            "dataset_name": self.dataset_name,
            "num_patients": self.num_patients,
            "patient_ids": self.patient_ids,
            "dataset_type": self.dataset_type,
            "num_validation_days": self.num_validation_days,
            "extract_features": self.extract_features,
        }

        # Add data shapes
        if self.train_data is not None:
            info["train_shapes"] = {
                pid: df.shape for pid, df in self.train_data.items()
            }
        if self.validation_data is not None:
            info["validation_shapes"] = {
                pid: df.shape for pid, df in self.validation_data.items()
            }

        # Add validation metrics if available
        if self.data_metrics:
            info["metrics"] = self.data_metrics

        return info

    # Public Abstract Method Implementations
    def load_data(self) -> None:
        """Load processed data from cache or process raw data and cache it.

        Attempts to load data from cache first if use_cached is True. If cache
        is not available or use_cached is False, processes raw data and saves
        to cache. After loading, splits data into train/validation sets if
        dataset_type is 'train'.

        Side Effects:
            Sets self.processed_data, self.train_data, and self.validation_data.
        """
        logger.info("=" * 60)
        logger.info("Beginning Tamborlane 2008 data loading process:")
        logger.info(f"\tDataset: {self.dataset_name} - {self.dataset_type}")
        logger.info(f"\tColumns: {self.keep_columns}")
        logger.info(f"\tExtract features: {self.extract_features}")
        logger.info(f"\tGeneric patient start date: {self.generic_patient_start_date}")
        logger.info(f"\tNumber of validation days: {self.num_validation_days}")

        if self.parallel:
            logger.info(f"\tUsing parallel processing with {self.max_workers} workers")
        else:
            logger.info("\tSequential processing")

        need_to_process_data = True

        if self.use_cached and self.cache_manager:
            # Try to load from cache
            try:
                cached_full_data = self.cache_manager.load_full_processed_data(
                    self.dataset_name
                )
                if cached_full_data is not None:
                    self.processed_data = cached_full_data
                    logger.info(
                        f"Loaded full processed data from cache for {len(cached_full_data)} patients"
                    )
                    need_to_process_data = False
                else:
                    # Try old cache format
                    cached_data = self.cache_manager.load_processed_data(
                        self.dataset_name, self.dataset_type
                    )
                    if cached_data is not None:
                        self._load_from_cache(cached_data)
                        need_to_process_data = False
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")

        # Process raw data if needed
        if need_to_process_data:
            logger.info("Processing raw data...")
            self._process_and_cache_data()

        # Split train/validation data
        if self.dataset_type == "train" and isinstance(self.processed_data, dict):
            self.train_data, self.validation_data = self._split_train_validation()

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

    # Public Methods
    def get_patient_data(self, patient_id: str) -> Optional[pd.DataFrame]:
        """Get processed data for a specific patient.

        Args:
            patient_id: The patient identifier string (e.g., 'p01', 'p02').

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the patient's CGM data,
                or None if the patient is not found or no data is loaded.
        """
        if self.processed_data is None:
            return None
        return self.processed_data.get(patient_id)

    def get_combined_data(self, data_type: str = "all") -> pd.DataFrame:
        """Get all patients' data combined into a single DataFrame.

        Args:
            data_type: Which data subset to return. Options are:
                - 'all': All processed data (default)
                - 'train': Only training data
                - 'validation': Only validation data

        Returns:
            pd.DataFrame: Combined DataFrame with all patients' data. Each
                patient's data includes a 'p_num' column identifying the patient.
                Returns empty DataFrame if no data is available.
        """
        if data_type == "train" and self.train_data:
            data_dict = self.train_data
        elif data_type == "validation" and self.validation_data:
            data_dict = self.validation_data
        else:
            data_dict = self.processed_data

        if not data_dict:
            return pd.DataFrame()

        # Combine all patient DataFrames
        all_dfs = []
        for patient_id, patient_df in data_dict.items():
            if isinstance(patient_df, pd.DataFrame):
                df_copy = patient_df.copy()
                if "p_num" not in df_copy.columns:
                    df_copy["p_num"] = patient_id
                all_dfs.append(df_copy)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=False)
        else:
            return pd.DataFrame()

    def save_processed_data(
        self, output_path: Union[str, Path], file_format: str = "csv"
    ) -> None:
        """Save processed data to files, one file per patient.

        Args:
            output_path: Directory path where files will be saved. Created if
                it does not exist.
            file_format: Output file format. Supported values are:
                - 'csv': Comma-separated values (default)
                - 'parquet': Apache Parquet columnar format

        Raises:
            ValueError: If an unsupported file format is specified.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.processed_data is None:
            logger.warning("No processed data to save")
            return

        for patient_id, patient_df in self.processed_data.items():
            if file_format == "csv":
                file_path = output_path / f"patient_{patient_id}.csv"
                patient_df.to_csv(file_path)
            elif file_format == "parquet":
                file_path = output_path / f"patient_{patient_id}.parquet"
                patient_df.to_parquet(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
        num_patients = len(self.processed_data) if self.processed_data else 0
        logger.info(f"Saved {num_patients} patient files to {output_path}")

    # Protected Abstract Method Implementations
    def _process_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Process raw data with cleaning and feature extraction.

        Cleans the raw CGM data, splits by patient, processes each patient's
        data (in parallel or sequentially based on self.parallel), extracts
        CGM features if enabled, and caches the results.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their
                processed DataFrames with cleaned glucose values and optional
                extracted features.

        Raises:
            ValueError: If raw_data is None (load_raw() not called first).
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")

        logger.info("Processing Tamborlane 2008 raw data...")

        # Clean the data using the updated cleaner
        cleaned_data = clean_tamborlane_2008_data(self.raw_data)

        # Split by patient
        if "p_num" not in cleaned_data.columns:
            # If no patient column, treat as single patient
            logger.warning(
                "No patient ID column found, treating as single patient dataset"
            )
            cleaned_data["p_num"] = "patient_001"

        # Use the splitter function
        multipatient_data_dict = split_multipatient_dataframe(cleaned_data, "p_num")
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

    # Protected Methods
    def _process_patients_parallel(
        self, multipatient_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Process multiple patients' data in parallel using ProcessPoolExecutor.

        Args:
            multipatient_data_dict: Dictionary mapping patient IDs to their
                raw DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their
                processed DataFrames. Patients that fail processing are logged
                but excluded from the result.
        """
        processed_results = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Prepare data tuples
            patient_data_tuples = [
                (p_num, patient_df, self.generic_patient_start_date)
                for p_num, patient_df in multipatient_data_dict.items()
            ]

            # Submit all tasks
            future_to_patient = {
                executor.submit(
                    process_single_patient_tamborlane, patient_tuple
                ): patient_tuple[0]
                for patient_tuple in patient_data_tuples
            }

            # Collect results
            for future in as_completed(future_to_patient):
                p_num = future_to_patient[future]
                try:
                    patient_id, result = future.result()
                    processed_results[patient_id] = result
                    logger.info(f"Successfully processed patient {patient_id}")
                except Exception as exc:
                    logger.error(f"Patient {p_num} generated an exception: {exc}")

        return processed_results

    def _process_patients_sequential(
        self, multipatient_data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Process multiple patients' data sequentially.

        Args:
            multipatient_data_dict: Dictionary mapping patient IDs to their
                raw DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their
                processed DataFrames.
        """
        processed_results = {}

        for p_num, patient_df in multipatient_data_dict.items():
            logger.info(f"Processing patient {p_num}...")
            patient_data_tuple = (p_num, patient_df, self.generic_patient_start_date)
            patient_id, processed_df = process_single_patient_tamborlane(
                patient_data_tuple
            )
            processed_results[patient_id] = processed_df

        return processed_results

    def _split_train_validation(
        self,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Split processed data into train and validation sets per patient.

        Uses self.train_percentage to determine the split ratio. Patients with
        insufficient data (less than 2 days) or missing datetime columns are
        skipped with a warning.

        Returns:
            tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: A tuple
                containing (train_dict, val_dict) where each dict maps patient
                IDs to their respective train/validation DataFrames.
        """
        train_dict: dict[str, pd.DataFrame] = {}
        val_dict: dict[str, pd.DataFrame] = {}

        if not isinstance(self.processed_data, dict):
            return {}, {}

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

    def _validate_dataset(self) -> None:
        """Validate the loaded dataset and compute quality metrics.

        Combines all patient data and computes validation metrics including
        total rows, unique patients, glucose statistics, and time-in-range
        percentages. Results are stored in self.data_metrics and logged.

        Side Effects:
            Sets self.data_metrics with computed validation metrics.
        """
        if not self.processed_data:
            logger.warning("No data to validate")
            return

        # Combine all patient data for overall metrics
        all_data = []
        for patient_df in self.processed_data.values():
            if isinstance(patient_df, pd.DataFrame):
                all_data.append(patient_df)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=False)
            self.data_metrics = validate_tamborlane_data(combined_df)

            logger.info("Dataset validation complete:")
            logger.info(f"  Total rows: {self.data_metrics.get('total_rows', 0)}")
            logger.info(
                f"  Unique patients: {self.data_metrics.get('unique_patients', 0)}"
            )
            if "glucose_mean" in self.data_metrics:
                logger.info(
                    f"  Mean glucose: {self.data_metrics['glucose_mean']:.2f} mmol/L"
                )
                logger.info(
                    f"  Std glucose: {self.data_metrics['glucose_std']:.2f} mmol/L"
                )
            elif "glucose_mean_mg_dl" in self.data_metrics:
                logger.info(
                    f"  Mean glucose: {self.data_metrics['glucose_mean_mg_dl']:.2f} mg/dL"
                )
                logger.info(
                    f"  Std glucose: {self.data_metrics['glucose_std_mg_dl']:.2f} mg/dL"
                )
            if "time_in_range" in self.data_metrics:
                logger.info(
                    f"  Time in range: {self.data_metrics['time_in_range']:.1f}%"
                )
                logger.info(
                    f"  Time below range: {self.data_metrics['time_below_range']:.1f}%"
                )
                logger.info(
                    f"  Time above range: {self.data_metrics['time_above_range']:.1f}%"
                )

    def _load_from_cache(self, cached_data: Dict[str, pd.DataFrame]) -> None:
        """Load and filter processed data from cache.

        Loads cached data into self.processed_data and optionally filters
        columns based on self.keep_columns.

        Args:
            cached_data: Dictionary mapping patient IDs to their cached
                DataFrames.

        Raises:
            TypeError: If cached_data is not a dictionary.

        Side Effects:
            Sets self.processed_data with the loaded (and optionally filtered)
            data.
        """
        logger.info("Loading processed data from cache...")

        if not isinstance(cached_data, dict):
            raise TypeError(f"Expected dict for cached data, got {type(cached_data)}")

        self.processed_data = cached_data

        # Apply column filtering if needed
        if self.keep_columns:
            filtered_data = {}
            for patient_id, patient_df in cached_data.items():
                columns_to_keep = [
                    col for col in self.keep_columns if col != "datetime"
                ]

                # Check column availability
                available_cols = [
                    col for col in columns_to_keep if col in patient_df.columns
                ]
                if len(available_cols) < len(columns_to_keep):
                    missing = set(columns_to_keep) - set(available_cols)
                    logger.warning(f"Patient {patient_id}: Missing columns {missing}")

                if available_cols:
                    filtered_data[patient_id] = patient_df[available_cols]
                else:
                    filtered_data[patient_id] = patient_df

            self.processed_data = filtered_data

    def _process_and_cache_data(self) -> None:
        """Process raw data and cache the results.

        Loads raw data via load_raw(), processes it via _process_raw_data(),
        and stores results in self.processed_data. Caching is handled within
        _process_raw_data().

        Side Effects:
            Sets self.raw_data and self.processed_data.
        """
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example: Load the dataset
    loader = Tamborlane2008DataLoader(
        use_cached=False,  # Process from raw data
        extract_features=True,  # Extract CGM features
        num_validation_days=7,  # Use 7 days for validation
        parallel=False,  # Use sequential processing for debugging
    )

    # Print dataset information
    print("\nDataset Information:")
    info = loader.dataset_info
    for key, value in info.items():
        if isinstance(value, dict) and len(str(value)) > 100:
            print(f"{key}: {type(value)} with {len(value)} items")
        else:
            print(f"{key}: {value}")

    # Get sample patient data
    if loader.patient_ids:
        sample_patient_id = loader.patient_ids[0]
        sample_data = loader.get_patient_data(sample_patient_id)

        if sample_data is not None:
            print(f"\nSample data for patient {sample_patient_id}:")
            print(f"  Shape: {sample_data.shape}")
            print(f"  Columns: {list(sample_data.columns)}")
            print(
                f"  Date range: {sample_data.index.min()} to {sample_data.index.max()}"
            )
            # Show first few rows
            print("\nFirst 5 rows:")
            print(sample_data.head())
