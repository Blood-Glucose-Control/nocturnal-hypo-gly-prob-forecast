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
Pleae remember to update the overload signatures in this file
to match the actual parameters of the data loader classes.
This ensures that type checking and autocompletion work correctly in IDEs.
"""

from typing import Union, Optional, Dict, Any, overload, Literal
from src.data.diabetes_datasets import BrisT1DDataLoader
from src.data.diabetes_datasets import GlurooDataLoader
from src.data.diabetes_datasets import Lynch2022DataLoader
from src.data.diabetes_datasets.awesome_cgm.aleppo.aleppo import AleppoDataLoader


# TODO: Add train_percentage parameter
@overload
def get_loader(
    data_source_name: Literal["lynch_2022"],
    dataset_type: str = "train",
    keep_columns: Optional[list[str]] = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    config: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    max_workers: int = 3,
) -> Lynch2022DataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["kaggle_brisT1D"],
    dataset_type: str = "train",
    keep_columns: Optional[list[str]] = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    config: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    max_workers: int = 3,
) -> BrisT1DDataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["gluroo"],
    dataset_type: str = "train",
    keep_columns: Optional[list[str]] = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    config: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
) -> GlurooDataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["aleppo"],
    keep_columns: Optional[list[str]] = None,
    use_cached: bool = False,
    # config: Optional[Dict[str, Any]] = None,
    train_percentage: float = 0.9,
    parallel: bool = True,
    max_workers: int = 3,
) -> AleppoDataLoader: ...


def get_loader(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    train_percentage: float = 0.9,
    config: dict | None = None,
    parallel: bool = True,
    max_workers: int = 3,
) -> Union[BrisT1DDataLoader, GlurooDataLoader, Lynch2022DataLoader, AleppoDataLoader]:
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
        train_percentage (float): Percentage of the data to use for training. Default: 0.9
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
            use_cached=use_cached,
            dataset_type=dataset_type,
            parallel=parallel,
            max_workers=max_workers,
        )
    elif data_source_name == "gluroo":
        return GlurooDataLoader(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            config=config,
            parallel=parallel,
        )
    elif data_source_name == "aleppo":
        return AleppoDataLoader(
            keep_columns=keep_columns,
            use_cached=use_cached,
            train_percentage=train_percentage,
            parallel=parallel,
            max_workers=max_workers,
        )
    elif data_source_name == "lynch_2022":
        return Lynch2022DataLoader(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            use_cached=use_cached,
            dataset_type=dataset_type,
            parallel=parallel,
            max_workers=max_workers,
        )
    else:
        raise ValueError(f"Invalid dataset_name: {data_source_name}.")
