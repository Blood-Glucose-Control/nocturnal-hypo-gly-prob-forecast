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

    def _clean_and_format_raw_data(
        self, raw_data: pd.DataFrame
    ) -> pd.DataFrame | dict[str, dict[str, pd.DataFrame]]:
        """
        Translate the raw data to the correct format for the preprocessing pipeline.
        Convert column names and units to standardized format.
        """
        raise NotImplementedError("_translate_raw_data must be implemented by subclass")

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
