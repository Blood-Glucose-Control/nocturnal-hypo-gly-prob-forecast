# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Data preprocessing pipeline for diabetes patient monitoring data.

This module provides the main preprocessing pipeline that orchestrates data validation
and feature engineering for diabetes patient datasets. The pipeline ensures data
quality through validation checks and enriches raw monitoring data with derived
physiological features necessary for glucose prediction modeling.

The pipeline handles:
- Input data validation and format checking
- Patient-specific data processing with logging
- Integration of feature engineering workflows
- Error handling and reporting for data quality issues

Functions:
    preprocessing_pipeline: Main preprocessing function that validates input data
                          and applies feature engineering to create physiological
                          features from raw diabetes monitoring data.

Dependencies:
    - Feature engineering from src.data.preprocessing.feature_engineering
    - Data validation from src.data.preprocessing.validation

Example:
    >>> import pandas as pd
    >>> from src.data.preprocessing.pipeline import preprocessing_pipeline
    >>>
    >>> # Raw patient data with required columns
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
    ...     'p_num': ['001'] * 100,
    ...     'bg_mM': [5.5, 6.1, 5.8, ...],
    ...     'msg_type': ['', 'ANNOUNCE_MEAL', '', ...],
    ...     'food_g': [0, 30, 0, ...],
    ...     'dose_units': [0, 0, 5, ...]
    ... })
    >>>
    >>> processed_df = preprocessing_pipeline('001', df)
    >>> # Returns DataFrame with original data plus IOB, COB, and availability features

Notes:
    - Input DataFrame must contain core columns: datetime, p_num, bg_mM
    - Optional columns (msg_type, food_g, dose_units, rate) enhance features but don't block processing
    - CGM-only datasets are supported (COB/IOB will be set to NaN)
    - Patient identifier is used for logging and tracking purposes
    - Processing includes comprehensive logging for monitoring pipeline execution
    - Output preserves all original data while adding derived physiological features
    - TODO: Future enhancement planned for parallel processing of multiple patients
"""

import logging

import pandas as pd

from src.data.preprocessing.feature_engineering import create_physiological_features
from src.data.preprocessing.validation import validate_required_columns

logger = logging.getLogger(__name__)

# Core columns - always required for pipeline to function
REQUIRED_COLUMNS = [
    "datetime",  # Datetime of the data (not the index)
    "p_num",  # Patient number (id)
    "bg_mM",  # Blood glucose in mmol/L
]

# Optional columns - enhance features but don't block processing
OPTIONAL_COLUMNS = [
    "msg_type",  # Message type: ANNOUNCE_MEAL | ''
    "food_g",  # Carbs in grams (enables COB calculation)
    "dose_units",  # Insulin units (enables IOB calculation)
    "rate",  # Basal rate U/hr (enables basal rollover)
]


def preprocessing_pipeline(
    p_num: str, df: pd.DataFrame, use_aggregation: bool = False
) -> pd.DataFrame:
    """
    Preprocesses patient data through feature engineering pipeline.

    This function validates the input data format and applies feature derivation
    to create additional metrics including insulin-on-board (IOB), carbs-on-board (COB),
    insulin availability, and carb availability features.

    Args:
        p_num (str): Patient identifier for logging purposes
        df (pd.DataFrame): Raw patient data containing datetime, blood glucose,
                          meal announcements, carb intake, and insulin doses
        use_aggregation (bool, optional): Whether to use aggregation to ensure regular time intervals.
                                          If True, will consider all rows within the same regular time interval.
                                          If False, will only consider the first row within the regular time interval.
    Returns:
        pd.DataFrame: Processed DataFrame with original data plus derived features

    Raises:
        ValueError: If required columns are missing from the input DataFrame
    """
    # TODO: Create an option for both serial and parallel processing of the multipatient files.
    logger.info("==============================")
    logger.info(f"Preprocessing patient {p_num}")
    logger.info("==============================")

    validate_required_columns(df, REQUIRED_COLUMNS)

    # Log which optional columns are present
    present_optional = [col for col in OPTIONAL_COLUMNS if col in df.columns]
    missing_optional = [col for col in OPTIONAL_COLUMNS if col not in df.columns]
    if present_optional:
        logger.info(f"Optional columns present: {present_optional}")
    if missing_optional:
        logger.info(
            f"Optional columns missing (features will be skipped): {missing_optional}"
        )

    patient_df = df.copy(deep=True)
    processed_df = create_physiological_features(
        patient_df, use_aggregation=use_aggregation
    )
    return processed_df
