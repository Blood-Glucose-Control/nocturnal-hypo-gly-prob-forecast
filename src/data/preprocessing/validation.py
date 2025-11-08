# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Data validation utilities for preprocessing pipelines.

This module provides validation functions to ensure data integrity and format
compliance across different stages of the diabetes data processing pipeline.
The validators check for required columns, data types, and structural requirements
needed for successful feature engineering and analysis.

Functions:
    validate_required_columns: Validates that a DataFrame contains all required columns
                              for processing, with support for index column checking.

Example:
    >>> import pandas as pd
    >>> from src.data.preprocessing.validation import validate_required_columns
    >>>
    >>> df = pd.DataFrame({'datetime': [...], 'bg_mM': [...], 'food_g': [...]})
    >>> required_cols = ['datetime', 'bg_mM', 'food_g']
    >>> validate_required_columns(df, required_cols)  # Returns True if valid

    >>> # Will raise ValueError if columns are missing
    >>> validate_required_columns(df, ['missing_column'])
    ValueError: Data is not in the correct format. Missing columns: {'missing_column'}...

Notes:
    - Validation functions consider both DataFrame columns and index names
    - Error messages include both missing and available columns for debugging
    - Designed to be reusable across different pipeline stages with varying requirements
"""

from typing import List

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validates that the DataFrame contains all required columns.

    Checks if the input DataFrame has the necessary columns.
    The function also considers the DataFrame's index name as a potential column.

    Args:
        df (pd.DataFrame): Input DataFrame to validate
        required_columns (List[str]): List of column names that must be present

    Returns:
        bool: True if all required columns are present

    Raises:
        ValueError: If any required columns are missing from the DataFrame,
                   with details about which columns are missing and available
    """
    required_columns_set = set(required_columns)

    # Get all available columns including the index
    df_columns_set = set(df.columns)
    if df.index.name:
        df_columns_set.add(str(df.index.name))

    if not required_columns_set.issubset(df_columns_set):
        missing_columns = required_columns_set - df_columns_set
        raise ValueError(
            f"Data is not in the correct format. Missing columns: {missing_columns}. "
            f"Available columns: {sorted(df_columns_set)}"
        )
    return True
