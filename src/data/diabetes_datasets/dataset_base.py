"""Dataset handling base module for the nocturnal project.

This module provides the foundation for all dataset-related functionality through
the DatasetBase abstract class. It establishes a consistent interface for loading,
processing, and validating different types of datasets used throughout the project.

The module implements a standard two-phase data loading pattern:
1. Load raw data from source
2. Process raw data into a format suitable for analysis and modeling

This design promotes:
- Consistency across different dataset implementations
- Separation of concerns between data acquisition and processing
- Easy extensibility for new dataset types
- Data validation to prevent downstream errors

Example:
    To use this module, create a subclass of DatasetBase for your specific dataset:

    ```python
    from src.data.dataset_base import DatasetBase

    class MyCustomDataset(DatasetBase):
        @property
        def dataset_name(self):
            return "custom_dataset"

        def load_raw(self):
            # Implementation for loading raw data
            return raw_df

        def load_data(self):
            if self.raw_data is None:
                self.raw_data = self.load_raw()
            return self._process_raw_data()

        def _process_raw_data(self):
            # Implementation for processing the raw data
            return processed_df
    ```
"""

import pandas as pd
from abc import ABC, abstractmethod


class DatasetBase(ABC):
    """Base class for dataset loading and processing.

    This abstract base class defines the interface for dataset handling classes
    in the nocturnal project. It provides a standardized way to load, process,
    and validate datasets. All dataset loaders should inherit from this class
    and implement its abstract methods.

    The class follows a two-phase data loading pattern:
    1. Load raw data using `load_raw()`
    2. Process that data into a usable form using `load_data()`

    Attributes:
        processed_data (pd.DataFrame or pd.Series): The processed dataset after loading
        raw_data (pd.DataFrame or pd.Series): The raw dataset without processing

    Example:
        ```python
        class MyDataset(DatasetBase):
            def load_raw(self):
                # Implementation
                return df

            @property
            def dataset_name(self):
                return "my_dataset"

            def load_data(self):
                # Implementation
                return processed_df

            def _process_raw_data(self):
                # Implementation
                return processed_df
        ```
    """

    def __init__(self):
        self.processed_data = None
        self.raw_data = None

    @abstractmethod
    def load_raw(self):
        """Load the raw dataset without any processing.

        Returns:
            pd.DataFrame or pd.Series: The raw dataset
        """
        raise NotImplementedError("load_raw must be implemented by subclass")

    @property
    def dataset_name(self):
        """Get the name of the dataset.

        Returns:
            str: Name of the dataset
        """
        raise NotImplementedError("get_dataset_name must be implemented by subclass")

    @abstractmethod
    def load_data(self):
        """Load the processed dataset.

        This method should handle any necessary preprocessing of the raw data
        before returning the final dataset ready for use.

        Returns:
            pd.DataFrame or pd.Series: The processed dataset ready for use
        """
        raise NotImplementedError("load_data must be implemented by subclass")

    def _process_raw_data(self):
        """Process the raw data.

        Returns:
            pd.DataFrame or pd.Series
        """
        raise NotImplementedError("_process_raw_data must be implemented by subclass")

    def _validate_data(self, data):
        """Validate the loaded data.

        Args:
            data (pd.DataFrame or pd.Series): Data to validate

        Returns:
            bool:True if data is valid, raises exception otherwise
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Data must be a pandas DataFrame or Series")
        if data.empty:
            raise ValueError("Dataset is empty")
        return True

    def create_validation_table(self):
        """Create a validation table for the dataset.

        This method extracts comprehensive statistics for each patient in the dataset,
        including temporal information, demographics, and physiological measurements.

        Returns:
            pd.DataFrame: A DataFrame containing validation results with columns:
                - patient_id: Patient identifier
                - num_days: Number of unique days in patient data
                - start_date: First timestamp in patient data
                - end_date: Last timestamp in patient data
                - date_type: 'artificial' or 'real' based on generic_patient_start_date
                - age: Patient age (if available)
                - sex: Patient sex (if available)
                - avg_bg_mM: Average blood glucose in mmol/L
                - min_bg_mM: Minimum blood glucose in mmol/L
                - max_bg_mM: Maximum blood glucose in mmol/L
                - avg_carbs_g: Average carbohydrate intake in grams
                - min_carbs_g: Minimum carbohydrate intake in grams
                - max_carbs_g: Maximum carbohydrate intake in grams
                - avg_insulin_units: Average insulin dose in units
                - min_insulin_units: Minimum insulin dose in units
                - max_insulin_units: Maximum insulin dose in units
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not loaded. Call load_data() first.")

        validation_rows = []
        
        # Determine if dates are artificial
        # Check if the loader has generic_patient_start_date attribute
        generic_date = getattr(self, 'generic_patient_start_date', None)
        date_type = 'artificial' if generic_date is not None else 'real'

        # Handle nested test data structure (patient_id -> {sub_id -> DataFrame})
        if isinstance(self.processed_data, dict):
            for patient_id, patient_data in self.processed_data.items():
                if isinstance(patient_data, dict):
                    # Nested structure (test data) - process each sub_id separately
                    for sub_id, patient_df in patient_data.items():
                        if isinstance(patient_df, pd.DataFrame) and not patient_df.empty:
                            row = self._extract_patient_stats(
                                f"{patient_id}_{sub_id}", patient_df, date_type
                            )
                            validation_rows.append(row)
                elif isinstance(patient_data, pd.DataFrame) and not patient_data.empty:
                    # Simple structure (train data) - one DataFrame per patient
                    row = self._extract_patient_stats(patient_id, patient_data, date_type)
                    validation_rows.append(row)

        return pd.DataFrame(validation_rows)

    def _extract_patient_stats(self, patient_id: str, patient_df: pd.DataFrame, date_type: str) -> dict:
        """Extract statistics for a single patient.

        Args:
            patient_id: Patient identifier
            patient_df: Patient's DataFrame with datetime index
            date_type: 'artificial' or 'real'

        Returns:
            dict: Dictionary containing patient statistics
        """
        # Ensure datetime index
        if not isinstance(patient_df.index, pd.DatetimeIndex):
            if 'datetime' in patient_df.columns:
                patient_df = patient_df.set_index('datetime')
            else:
                # Cannot process without datetime
                return {
                    'patient_id': patient_id,
                    'error': 'No datetime index available'
                }

        # Calculate temporal statistics
        num_days = patient_df.index.normalize().nunique()
        start_date = patient_df.index.min()
        end_date = patient_df.index.max()

        # Initialize stats dictionary
        stats = {
            'patient_id': patient_id,
            'num_days': num_days,
            'start_date': start_date,
            'end_date': end_date,
            'date_type': date_type,
        }

        # Extract demographics (if available in DataFrame)
        for demo_col in ['age', 'sex']:
            if demo_col in patient_df.columns:
                # Get the most common value (mode) for this patient
                values = patient_df[demo_col].dropna()
                if len(values) > 0:
                    stats[demo_col] = values.mode()[0] if len(values.mode()) > 0 else values.iloc[0]
                else:
                    stats[demo_col] = None
            else:
                stats[demo_col] = None

        # Extract blood glucose statistics (bg_mM column)
        if 'bg_mM' in patient_df.columns:
            bg_data = patient_df['bg_mM'].dropna()
            if len(bg_data) > 0:
                stats['avg_bg_mM'] = bg_data.mean()
                stats['min_bg_mM'] = bg_data.min()
                stats['max_bg_mM'] = bg_data.max()
            else:
                stats['avg_bg_mM'] = None
                stats['min_bg_mM'] = None
                stats['max_bg_mM'] = None
        else:
            stats['avg_bg_mM'] = None
            stats['min_bg_mM'] = None
            stats['max_bg_mM'] = None

        # Extract carbohydrate statistics (food_g column)
        if 'food_g' in patient_df.columns:
            carbs_data = patient_df['food_g'].dropna()
            # Filter out zeros for min calculation to get actual carb intake events
            carbs_nonzero = carbs_data[carbs_data > 0]
            if len(carbs_data) > 0:
                stats['avg_carbs_g'] = carbs_data.mean()
                stats['min_carbs_g'] = carbs_nonzero.min() if len(carbs_nonzero) > 0 else 0.0
                stats['max_carbs_g'] = carbs_data.max()
            else:
                stats['avg_carbs_g'] = None
                stats['min_carbs_g'] = None
                stats['max_carbs_g'] = None
        else:
            stats['avg_carbs_g'] = None
            stats['min_carbs_g'] = None
            stats['max_carbs_g'] = None

        # Extract insulin statistics (dose_units column)
        if 'dose_units' in patient_df.columns:
            insulin_data = patient_df['dose_units'].dropna()
            # Filter out zeros for min calculation to get actual insulin doses
            insulin_nonzero = insulin_data[insulin_data > 0]
            if len(insulin_data) > 0:
                stats['avg_insulin_units'] = insulin_data.mean()
                stats['min_insulin_units'] = insulin_nonzero.min() if len(insulin_nonzero) > 0 else 0.0
                stats['max_insulin_units'] = insulin_data.max()
            else:
                stats['avg_insulin_units'] = None
                stats['min_insulin_units'] = None
                stats['max_insulin_units'] = None
        else:
            stats['avg_insulin_units'] = None
            stats['min_insulin_units'] = None
            stats['max_insulin_units'] = None

        return stats
