"""
Data loading module for accessing and processing various diabetes datasets.

This module provides a unified interface to access different data sources
through a factory function that returns the appropriate data loader based
on the requested data source name.
Please remember to update the __init__.py file in the datasets directory
to include any new dataset loaders you create, so they can be accessed
through the unified interface.
This allows for easy extensibility and maintainability of the data loading
process across different datasets.
"""

from src.data.datasets.dataset_base import DatasetBase
from src.data.datasets import BrisT1DDataLoader
from src.data.datasets import GlurooDataLoader


def get_loader(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    file_path: str | None = None,
    config: dict | None = None,
) -> DatasetBase:
    """
    Factory function to create and return the appropriate data loader instance.

    This function creates a data loader for the specified data source, configured
    according to the provided parameters. Each loader implements the DatasetBase
    interface, providing consistent data access methods across different sources.

    Parameters:
        data_source_name (str): The name of the data source to load.
                               Currently supports 'kaggle_brisT1D' and 'gluroo'.
                               Default: 'kaggle_brisT1D'
        dataset_type (str): The subset of data to load ('train', 'test', 'validation').
                           Default: 'train'
        keep_columns (list[str] | None): Specific columns to retain in the dataset.
                                       If None, all columns are loaded. Default: None
        use_cached (bool): Whether to use cached data if available. Default: False
        num_validation_days (int): Number of days to use for validation. Default: 20
        file_path (str | None): Custom path to data files. If None, uses default paths.
                              Default: None
        config (dict | None): Additional configuration parameters for the data loader.
                            Default: None

    Returns:
        DatasetBase: A data loader instance implementing the DatasetBase interface.

    Raises:
        ValueError: If an unsupported data source name is provided.
    """
    if data_source_name == "kaggle_brisT1D":
        return BrisT1DDataLoader(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            file_path=file_path,
            use_cached=use_cached,
            dataset_type=dataset_type,
        )
    elif data_source_name == "gluroo":
        return GlurooDataLoader(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            file_path=file_path,
            config=config,
        )
    else:
        raise ValueError(f"Invalid dataset_name: {data_source_name}.")
