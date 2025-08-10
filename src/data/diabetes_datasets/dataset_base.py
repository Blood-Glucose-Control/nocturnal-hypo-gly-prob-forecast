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
from src.data.cache_manager import CacheManager, get_cache_manager
from src.data.models.data import (
    DatasetType,
    DatasetConfig,
    DatasetStructure,
    DataEnvelope,
)


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

    use_cached: bool = True
    dataset_type: DatasetType = DatasetType.TRAIN
    raw_data: pd.DataFrame | pd.Series | None = None
    processed_data: pd.DataFrame | pd.Series | None = None
    cache_manager: CacheManager = get_cache_manager()
    dataset_config: DatasetConfig

    @abstractmethod
    def _load_raw(self, raw_data_path: str):
        """Load the raw dataset without any processing.

        Args:
            raw_data_path (str): Path to the raw data directory.
                The child class must determine which file to read and return
                based on the configs
        Returns:
            pd.DataFrame or pd.Series: The raw dataset
        """
        raise NotImplementedError("load_raw must be implemented by subclass")

    def load_raw(self):
        # does necassary fetching to get raw data; _load_raw only needs to read necassary files from this path
        raw_data_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )
        return self._load_raw(raw_data_path)

    @property
    def dataset_name(self):
        """Get the name of the dataset.

        Returns:
            str: Name of the dataset
        """
        raise NotImplementedError("get_dataset_name must be implemented by subclass")

    @property
    def dataset_structure(self) -> DatasetStructure:
        raise NotImplementedError(
            "get_dataset_structure must be implemented by subclass"
        )

    @abstractmethod
    def _load_data(self):
        """Load the processed dataset.

        This method should handle any necessary preprocessing of the raw data
        before returning the final dataset ready for use.

        Returns:
            pd.DataFrame or pd.Series: The processed dataset ready for use
        """
        raise NotImplementedError("load_data must be implemented by subclass")

    def load_data(self):
        if not self.use_cached:
            preprocessed = self._load_data()
            envelope = DataEnvelope(
                dataset_structure=self.dataset_structure, data=preprocessed
            )
            self.cache_manager.save_processed_data(
                self.dataset_name, self.dataset_type, envelope
            )
            return preprocessed

        cached_data = self.cache_manager.load_processed_data(
            self.dataset_name, self.dataset_type
        )
        if cached_data is None:
            preprocessed = self._load_data()
            envelope = DataEnvelope(
                dataset_structure=self.dataset_structure, data=preprocessed
            )
            self.cache_manager.save_processed_data(
                self.dataset_name, self.dataset_type, envelope
            )
            return preprocessed

        return self._preprocess_cached_data(cached_data)

    @abstractmethod
    def _preprocess_cached_data(self, cached_data: pd.DataFrame):
        """
        This is an optional hook that the data loader can implement
        to preprocess the cached data before returning it
        (eg: train test split; column selection etc)
        """
        return cached_data
