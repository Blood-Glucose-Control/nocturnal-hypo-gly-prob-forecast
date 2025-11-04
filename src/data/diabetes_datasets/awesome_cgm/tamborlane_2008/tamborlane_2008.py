"""
Tamborlane 2008 CGM Dataset Loader.

This module provides functionality to load and process the Tamborlane 2008 CGM dataset.
The dataset contains continuous glucose monitoring data from pediatric patients with
Type 1 diabetes, collected as part of the DirecNet study.

The module supports data preprocessing including time interval regularization,
feature extraction, and train/validation splitting.
"""

import logging
import gzip
import pickle
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

from src.data.cache_manager import get_cache_manager
from src.data.dataset_configs import get_dataset_config
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.diabetes_datasets.awesome_cgm.tamborlane_2008.data_cleaner import (
    clean_tamborlane_2008_data,
    process_single_patient_tamborlane,
    extract_cgm_features,
    validate_tamborlane_data,
)
from src.data.preprocessing.data_splitting import split_multipatient_dataframe
from src.data.preprocessing.time_processing import get_train_validation_split

logger = logging.getLogger(__name__)


class Tamborlane2008DataLoader(DatasetBase):
    """
    Data loader for the Tamborlane 2008 CGM dataset.

    This class handles loading, processing, and caching of the Tamborlane 2008 dataset,
    which contains continuous glucose monitoring data from pediatric Type 1 diabetes patients.
    The dataset provides high-frequency glucose measurements (typically every 5 minutes)
    and is valuable for developing glucose prediction models and studying glycemic patterns.

    Key features of this dataset:
    - CGM data from pediatric patients (age 8-17)
    - High-frequency measurements (5-minute intervals)
    - Multi-day continuous monitoring periods
    - Useful for nocturnal hypoglycemia prediction
    
    The loader supports intelligent caching at multiple levels:
    - Raw data caching to avoid re-downloading
    - Processed data caching to avoid re-processing
    - Train/validation split caching for consistent splits

    Attributes:
        keep_columns (list[str] | None): Specific columns to load from the dataset
        num_validation_days (int): Number of days to use for validation
        use_cached (bool): Whether to use cached processed data if available
        dataset_type (str): Type of dataset ('train' or 'test')
        parallel (bool): Whether to use parallel processing
        generic_patient_start_date (pd.Timestamp): Starting date for all patients
        max_workers (int): Maximum number of workers for parallel processing
        extract_features (bool): Whether to extract CGM-specific features
        processed_data (dict[str, pd.DataFrame]): The processed dataset
        train_data (dict[str, pd.DataFrame] | None): Training subset
        validation_data (dict[str, pd.DataFrame] | None): Validation subset
        data_metrics (dict): Validation metrics for the dataset
        raw_data (pd.DataFrame): The original unprocessed dataset

    Properties:
        dataset_name (str): Returns "tamborlane_2008"
        num_patients (int): Number of patients in the dataset
        patient_ids (list[str]): List of patient IDs
        train_data_shape_summary (dict[str, tuple[int, int]]): Shape summary for train data
    """

    def __init__(
        self,
        keep_columns: Optional[List[str]] = None,
        num_validation_days: int = 7,  # Shorter validation for CGM data
        use_cached: bool = True,
        dataset_type: str = "train",
        parallel: bool = True,
        generic_patient_start_date: pd.Timestamp = pd.Timestamp("2008-01-01"),
        max_workers: int = 3,
        extract_features: bool = True,
    ):
        """
        Initialize the Tamborlane 2008 data loader.

        Args:
            keep_columns: Specific columns to load from the dataset
            num_validation_days: Number of days to use for validation (default 7)
            use_cached: Whether to use cached processed data if available
            dataset_type: Type of dataset to load ('train' or 'test')
            parallel: Whether to use parallel processing
            generic_patient_start_date: Starting date for patients (default 2008-01-01)
            max_workers: Maximum number of workers for parallel processing
            extract_features: Whether to extract CGM-specific features
        """
        # Ensure required columns are included
        if keep_columns is not None:
            required_cols = ["datetime", "bg_mM", "p_num"]
            for col in required_cols:
                if col not in keep_columns:
                    keep_columns.append(col)
        
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.use_cached = use_cached
        self.dataset_type = dataset_type
        self.parallel = parallel
        self.generic_patient_start_date = generic_patient_start_date
        self.max_workers = max_workers
        self.extract_features = extract_features
        
        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        self.dataset_config = get_dataset_config(self.dataset_name)
        
        # Data containers
        self.raw_data = None
        self.processed_data = {}
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

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "tamborlane_2008"

    @property
    def num_patients(self) -> int:
        """Get the number of patients in the dataset."""
        if self.processed_data is None:
            return 0
        return len(self.processed_data)

    @property
    def patient_ids(self) -> List[str]:
        """Get list of patient IDs in the dataset."""
        if self.processed_data is None:
            return []
        return list(self.processed_data.keys())

    @property
    def train_data_shape_summary(self) -> Dict[str, Tuple[int, int]]:
        """
        Get shape summary for each patient's data.
        Returns a dict mapping patient_id to shape tuple.
        """
        if not isinstance(self.processed_data, dict):
            return {}
        
        shape_summary = {}
        for patient_id, patient_df in self.processed_data.items():
            if isinstance(patient_df, pd.DataFrame):
                shape_summary[patient_id] = patient_df.shape
        return shape_summary

    @property
    def dataset_info(self) -> Dict[str, any]:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics and metadata
        """
        info = {
            'dataset_name': self.dataset_name,
            'num_patients': self.num_patients,
            'patient_ids': self.patient_ids,
            'dataset_type': self.dataset_type,
            'num_validation_days': self.num_validation_days,
            'extract_features': self.extract_features,
        }
        
        # Add data shapes
        if self.train_data is not None:
            info['train_shapes'] = {pid: df.shape for pid, df in self.train_data.items()}
        if self.validation_data is not None:
            info['validation_shapes'] = {pid: df.shape for pid, df in self.validation_data.items()}
        
        # Add validation metrics if available
        if self.data_metrics:
            info['metrics'] = self.data_metrics
        
        return info

    def load_data(self):
        """
        Load processed data from cache or process raw data and save to cache.
        Then split train/validation data.
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
        
        if self.use_cached:
            # Try to load from cache
            cached_full_data = self.cache_manager.load_full_processed_data(
                self.dataset_name
            )
            if cached_full_data is not None:
                self.processed_data = cached_full_data
                logger.info(f"Loaded full processed data from cache for {len(cached_full_data)} patients")
                need_to_process_data = False
            else:
                # Try old cache format
                cached_data = self.cache_manager.load_processed_data(
                    self.dataset_name, self.dataset_type
                )
                if cached_data is not None:
                    self._load_from_cache(cached_data)
                    need_to_process_data = False
        
        # Process raw data if needed
        if need_to_process_data:
            logger.info("Processing raw data...")
            self._process_and_cache_data()
        
        # Split train/validation data
        if self.dataset_type == "train" and isinstance(self.processed_data, dict):
            self._split_train_validation()

    def load_raw(self):
        """
        Load raw data from CSV files.
        Searches in subdirectories if no files found in root.
        """
        raw_data_path = Path("/Users/kirby/BCG-WatAI/nocturnal-hypo-gly-prob-forecast/cache/data/awesome_cgm/tamborlane_2008/raw")
        
        # Search for CSV and Excel files recursively
        csv_files = list(raw_data_path.rglob("*.csv"))  
        
        if not csv_files:
            raise FileNotFoundError(f"No data files found in {raw_data_path}")
        
        if len(csv_files) > 1:
            dfs = []
            for file in csv_files:
                dfs.append(pd.read_csv(file))
            return pd.concat(dfs, ignore_index=True)
        
        # If single CSV file
        if len(csv_files) == 1:
            return pd.read_csv(csv_files[0])
        

    def _load_from_cache(self, cached_data: Dict[str, pd.DataFrame]):
        """
        Load processed data from cache.
        
        Args:
            cached_data: Dictionary mapping patient IDs to DataFrames
        """
        logger.info("Loading processed data from cache...")
        
        if not isinstance(cached_data, dict):
            raise TypeError(f"Expected dict for cached data, got {type(cached_data)}")
        
        self.processed_data = cached_data
        
        # Apply column filtering if needed
        if self.keep_columns:
            filtered_data = {}
            for patient_id, patient_df in cached_data.items():
                columns_to_keep = [col for col in self.keep_columns if col != "datetime"]
                
                # Check column availability
                available_cols = [col for col in columns_to_keep if col in patient_df.columns]
                if len(available_cols) < len(columns_to_keep):
                    missing = set(columns_to_keep) - set(available_cols)
                    logger.warning(f"Patient {patient_id}: Missing columns {missing}")
                
                if available_cols:
                    filtered_data[patient_id] = patient_df[available_cols]
                else:
                    filtered_data[patient_id] = patient_df
            
            self.processed_data = filtered_data

    def _process_and_cache_data(self):
        """
        Process raw data and save to cache.
        """
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()

    def _process_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process raw data with cleaning and feature extraction.
        
        Returns:
            Dictionary mapping patient IDs to processed DataFrames
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw() first.")
        
        logger.info("Processing Tamborlane 2008 raw data...")
        
        # Clean the data
        cleaned_data = clean_tamborlane_2008_data(self.raw_data)
        
        # Split by patient
        if 'p_num' not in cleaned_data.columns:
            # If no patient column, treat as single patient
            logger.warning("No patient ID column found, treating as single patient dataset")
            cleaned_data['p_num'] = 'patient_001'
        
        multipatient_data_dict = split_multipatient_dataframe(cleaned_data, 'p_num')
        logger.info(f"Processing {len(multipatient_data_dict)} patients")
        
        # Process each patient
        if self.parallel:
            processed_results = self._process_patients_parallel(multipatient_data_dict)
        else:
            processed_results = self._process_patients_sequential(multipatient_data_dict)
        
        # Extract features if requested
        if self.extract_features:
            for patient_id in processed_results:
                processed_results[patient_id] = extract_cgm_features(
                    processed_results[patient_id]
                )
        
        # Save to cache
        logger.info("Saving processed data to cache...")
        self.cache_manager.save_full_processed_data(
            self.dataset_name, processed_results
        )
        
        return processed_results

    def _process_patients_parallel(self, multipatient_data_dict: Dict) -> Dict:
        """
        Process patients in parallel.
        
        Args:
            multipatient_data_dict: Dictionary of patient data
            
        Returns:
            Dictionary of processed patient data
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
                executor.submit(process_single_patient_tamborlane, patient_tuple): patient_tuple[0]
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

    def _process_patients_sequential(self, multipatient_data_dict: Dict) -> Dict:
        """
        Process patients sequentially.
        
        Args:
            multipatient_data_dict: Dictionary of patient data
            
        Returns:
            Dictionary of processed patient data
        """
        processed_results = {}
        
        for p_num, patient_df in multipatient_data_dict.items():
            logger.info(f"Processing patient {p_num}...")
            patient_data_tuple = (p_num, patient_df, self.generic_patient_start_date)
            patient_id, processed_df = process_single_patient_tamborlane(patient_data_tuple)
            processed_results[patient_id] = processed_df
        
        return processed_results

    def _split_train_validation(self):
        """
        Split processed data into training and validation sets.
        """
        if not isinstance(self.processed_data, dict):
            raise TypeError("Cannot split data: processed_data must be a dictionary")
        
        # Define split parameters
        split_params = {
            'num_validation_days': self.num_validation_days,
            'split_method': 'get_train_validation_split',
            'dataset_type': self.dataset_type
        }
        
        # Try to load cached split
        cached_split_data = self.cache_manager.load_split_data(
            self.dataset_name, split_params
        )
        
        if cached_split_data is not None:
            train_data_dict, validation_data_dict = cached_split_data
            logger.info(f"Loaded cached train/validation split for {len(train_data_dict)} patients")
        else:
            logger.info(f"Creating new train/validation split with {self.num_validation_days} validation days")
            
            train_data_dict = {}
            validation_data_dict = {}
            
            for patient_id, patient_df in self.processed_data.items():
                if not isinstance(patient_df, pd.DataFrame):
                    logger.warning(f"Skipping patient {patient_id}: not a DataFrame")
                    continue
                
                # Ensure datetime index
                patient_data = patient_df.copy()
                if not isinstance(patient_data.index, pd.DatetimeIndex):
                    if 'datetime' in patient_data.columns:
                        patient_data = patient_data.set_index('datetime')
                    else:
                        logger.warning(f"No datetime index for patient {patient_id}")
                        continue
                
                # Ensure p_num column for compatibility
                if 'p_num' not in patient_data.columns:
                    patient_data['p_num'] = patient_id
                
                # Split the data
                patient_train, patient_validation, _ = get_train_validation_split(
                    patient_data, num_validation_days=self.num_validation_days
                )
                
                train_data_dict[patient_id] = patient_train
                validation_data_dict[patient_id] = patient_validation
            
            # Save to cache
            self.cache_manager.save_split_data(
                self.dataset_name, train_data_dict, validation_data_dict, split_params
            )
            logger.info(f"Cached train/validation split for {len(train_data_dict)} patients")
        
        self.train_data = train_data_dict
        self.validation_data = validation_data_dict
        
        # Calculate metadata
        if validation_data_dict:
            first_val_df = next(iter(validation_data_dict.values()))
            self.val_dt_col_type = first_val_df.index.dtype
        
        if train_data_dict:
            first_train_df = next(iter(train_data_dict.values()))
            self.train_dt_col_type = first_train_df.index.dtype
            
            # Calculate total training days
            all_train_dates = set()
            for patient_train_df in train_data_dict.values():
                datetime_index = pd.DatetimeIndex(patient_train_df.index)
                patient_dates = datetime_index.date
                all_train_dates.update(patient_dates)
            self.num_train_days = len(all_train_dates)

    def _validate_dataset(self):
        """
        Validate the loaded dataset and compute quality metrics.
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
            logger.info(f"  Unique patients: {self.data_metrics.get('unique_patients', 0)}")
            if 'glucose_mean' in self.data_metrics:
                logger.info(f"  Mean glucose: {self.data_metrics['glucose_mean']:.2f} mmol/L")
                logger.info(f"  Std glucose: {self.data_metrics['glucose_std']:.2f} mmol/L")
            if 'time_in_range' in self.data_metrics:
                logger.info(f"  Time in range: {self.data_metrics['time_in_range']:.1f}%")
                logger.info(f"  Time below range: {self.data_metrics['time_below_range']:.1f}%")
                logger.info(f"  Time above range: {self.data_metrics['time_above_range']:.1f}%")

    def get_patient_data(self, patient_id: str) -> Optional[pd.DataFrame]:
        """
        Get data for a specific patient.
        
        Args:
            patient_id: The patient identifier
            
        Returns:
            DataFrame for the patient or None if not found
        """
        return self.processed_data.get(patient_id)

    def get_combined_data(self, data_type: str = "all") -> pd.DataFrame:
        """
        Get all data combined into a single DataFrame.
        
        Args:
            data_type: Which data to return ('all', 'train', or 'validation')
            
        Returns:
            Combined DataFrame with all patients' data
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
                if 'p_num' not in df_copy.columns:
                    df_copy['p_num'] = patient_id
                all_dfs.append(df_copy)
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=False)
        else:
            return pd.DataFrame()