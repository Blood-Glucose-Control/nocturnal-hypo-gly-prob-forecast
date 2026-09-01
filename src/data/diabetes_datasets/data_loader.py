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

from typing import Literal, Union, overload

from . import (
    Aleppo2017DataLoader,
    Brown2019DataLoader,
    GlurooDataLoader,
    Lynch2022DataLoader,
    Tamborlane2008DataLoader,
)


@overload
def get_loader(
    data_source_name: Literal["lynch_2022"],
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    parallel: bool = True,
    max_workers: int = 14,
) -> Lynch2022DataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["brown_2019"],
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    parallel: bool = True,
    max_workers: int = 14,
) -> Brown2019DataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["gluroo"],
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    parallel: bool = True,
    max_workers: int = 14,
    load_all: bool = False,
) -> GlurooDataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["aleppo_2017"],
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    parallel: bool = True,
    max_workers: int = 14,
) -> Aleppo2017DataLoader: ...


@overload
def get_loader(
    data_source_name: Literal["tamborlane_2008"],
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    parallel: bool = True,
    max_workers: int = 14,
) -> Tamborlane2008DataLoader: ...


def get_loader(
    data_source_name: str = "aleppo_2017",
    keep_columns: list[str] | None = None,
    use_cached: bool = False,
    parallel: bool = True,
    max_workers: int = 14,
    load_all: bool = False,
) -> Union[
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
                               Currently supports 'aleppo_2017', 'lynch_2022', 'brown_2019', 'tamborlane_2008', and 'gluroo'.
                               Default: 'aleppo_2017'
        keep_columns (list[str] | None): Specific columns to retain in the dataset.
                                       If None, all columns are loaded. Default: None
        use_cached (bool): Whether to use cached data if available. Default: False

    Returns:
        DatasetBase: A data loader instance implementing the DatasetBase interface.

    Raises:
        ValueError: If an unsupported data source name is provided.
    """
    if data_source_name == "gluroo":
        return GlurooDataLoader(
            keep_columns=keep_columns,
            # parallel=parallel,
            max_workers=max_workers,
            load_all=load_all,
        )
    elif data_source_name == "aleppo_2017":
        return Aleppo2017DataLoader(
            keep_columns=keep_columns,
            use_cached=use_cached,
            parallel=parallel,
            max_workers=max_workers,
        )
    elif data_source_name == "lynch_2022":
        return Lynch2022DataLoader(
            keep_columns=keep_columns,
            use_cached=use_cached,
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
            use_cached=use_cached,
            parallel=parallel,
            max_workers=max_workers,
            extract_features=True,
        )
    else:
        raise ValueError(f"Invalid dataset_name: {data_source_name}.")
