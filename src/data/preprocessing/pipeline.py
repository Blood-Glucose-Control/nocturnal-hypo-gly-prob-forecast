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
    - Input DataFrame must contain all required columns: datetime, p_num, bg_mM,
      msg_type, food_g, dose_units
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

required_columns = [
    "datetime",  # Datetime of the data (not the index)
    "p_num",  # Patient number (id)
    "bg_mM",  # Blood glucose in mmol/L
    "msg_type",  # Message type: ANNOUNCE_MEAL | ''
    "food_g",  # Carbs in grams
    "dose_units",  # Insulin units
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

    validate_required_columns(df, required_columns)
    patient_df = df.copy(deep=True)
    processed_df = create_physiological_features(
        patient_df, use_aggregation=use_aggregation
    )
    return processed_df
