# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

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
Please remember to update the overload signatures in this file
to match the actual parameters of the data loader classes.
This ensures that type checking and autocompletion work correctly in IDEs.
Overloads facilitate type narrowing:

When to Use @overload
Return type depends on a literal input value (like our case)
Return type depends on presence/absence of parameters
Different parameter types produce different return types

Key Rules for @overload
Parameter order must match exactly between all overloads and the implementation
Parameter names must match exactly
All parameters should be present in each overload (use ... for defaults)
The implementation signature must be a superset of all overload signatures
"""

from typing import overload, Literal, Union
from src.data.diabetes_datasets import Brown2019DataLoader
from src.data.diabetes_datasets import BrisT1DDataLoader
from src.data.diabetes_datasets import GlurooDataLoader
from src.data.diabetes_datasets import Lynch2022DataLoader
from src.data.diabetes_datasets import Aleppo2017DataLoader
from src.data.diabetes_datasets import Tamborlane2008DataLoader


@overload
def get_loader(
    data_source_name: Literal["lynch_2022"],
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    train_percentage: float = ...,
    parallel: bool = True,
    max_workers: int = 14,
) -> Lynch2022DataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["brown_2019"],
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    train_percentage: float = ...,
    parallel: bool = True,
    max_workers: int = 14,
) -> Brown2019DataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["kaggle_brisT1D"],
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    train_percentage: float = ...,
    parallel: bool = True,
    max_workers: int = 14,
) -> BrisT1DDataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["gluroo"],
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    train_percentage: float = ...,
    parallel: bool = True,
    max_workers: int = 14,
    load_all: bool = False,
) -> GlurooDataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["aleppo_2017"],
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    train_percentage: float = ...,
    parallel: bool = True,
    max_workers: int = 14,
) -> Aleppo2017DataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["tamborlane_2008"],
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 7,
    train_percentage: float = ...,
    parallel: bool = True,
    max_workers: int = 14,
) -> Tamborlane2008DataLoader: ...


def get_loader(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    train_percentage: float = 0.9,
    parallel: bool = True,
    max_workers: int = 14,
    load_all: bool = False,
) -> Union[
    BrisT1DDataLoader,
    GlurooDataLoader,
    Aleppo2017DataLoader,
    Lynch2022DataLoader,
    Brown2019DataLoader,
    Tamborlane2008DataLoader,
]:
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
            # num_validation_days=num_validation_days,
            config=config,
            # parallel=parallel,
            max_workers=max_workers,
            load_all=load_all,
        )
    elif data_source_name == "aleppo_2017":
        return Aleppo2017DataLoader(
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
    elif data_source_name == "brown_2019":
        return Brown2019DataLoader(
            keep_columns=keep_columns,
            use_cached=use_cached,
        )
    elif data_source_name == "tamborlane_2008":
        return Tamborlane2008DataLoader(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            use_cached=use_cached,
            dataset_type=dataset_type,
            parallel=parallel,
            max_workers=max_workers,
            extract_features=config.get("extract_features", True) if config else True,
        )
    else:
        raise ValueError(f"Invalid dataset_name: {data_source_name}.")
