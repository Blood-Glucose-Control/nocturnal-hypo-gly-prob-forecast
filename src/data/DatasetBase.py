import pandas as pd
from abc import ABC, abstractmethod


class DatasetBase(ABC):
    """Base class for dataset loading and processing.

    This class defines the interface for dataset handling classes.
    All dataset loaders should inherit from this class and implement its methods.
    """

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
