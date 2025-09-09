"""
Feature engineering utilities for diabetes patient data analysis.

This module provides functions to derive physiological features from raw diabetes
monitoring data, including insulin-on-board (IOB), carbs-on-board (COB), and
availability metrics. The feature engineering process transforms time-series data
into a format suitable for glucose prediction modeling and analysis.

The module handles:
- Time series regularization and gap filling
- Physiological modeling of insulin and carbohydrate dynamics
- Creation of derived features that capture metabolic state
- Validation of datetime indexing requirements

Functions:
    create_physiological_features: Main feature engineering pipeline that creates
                                 physiological features from glucose monitoring data
                                 including IOB, COB, and availability metrics.

Dependencies:
    - Carbohydrate absorption modeling from src.data.physiological.carb_model
    - Insulin absorption modeling from src.data.physiological.insulin_model
    - Time series sampling utilities from src.data.preprocessing.sampling

Example:
    >>> import pandas as pd
    >>> from src.data.preprocessing.feature_engineering import create_physiological_features
    >>>
    >>> # DataFrame with datetime index and required columns
    >>> df = pd.DataFrame({
    ...     'food_g': [0, 30, 0, 0],
    ...     'dose_units': [0, 0, 5, 0]
    ... }, index=pd.date_range('2024-01-01', periods=4, freq='5min'))
    >>>
    >>> enhanced_df = create_physiological_features(df)
    >>> # Returns DataFrame with COB, IOB, and availability columns added

Notes:
    - Input DataFrame must have a DatetimeIndex
    - Minimum required columns: 'food_g' (carbs in grams), 'dose_units' (insulin units)
    - Processing time scales with data size due to physiological model calculations
    - Output includes all original columns plus derived physiological features
"""

import logging

import pandas as pd

from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)
from src.data.preprocessing.sampling import ensure_regular_time_intervals

logger = logging.getLogger(__name__)

required_columns = [
    "datetime",
    "food_g",  # Carbs in grams
    "dose_units",  # Insulin units
]


def create_physiological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives physiological features from patient glucose monitoring data.

    This function performs comprehensive feature engineering by:
    1. Validating the DataFrame has a datetime index
    2. Ensuring regular time intervals by filling gaps in the time series
    3. Creating carbs-on-board (COB) and carb availability features from food intake
    4. Creating insulin-on-board (IOB) and insulin availability features from dose data

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index containing at minimum
                          'food_g' (carbs in grams) and 'dose_units' (insulin units)


    Returns:
        pd.DataFrame: Enhanced DataFrame with original data plus derived physiological
                     features including COB, IOB, and availability metrics

    Raises:
        ValueError: If the DataFrame does not have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    logger.info("create_physiological_features(): Deriving features...")
    df, freq = ensure_regular_time_intervals(df)

    logger.info(
        "\tCreating COB/IOB and availability columns. This may take a while depending on the size of the data."
    )
    logger.info("\tCreating COB and carb availability columns...")
    df = create_cob_and_carb_availability_cols(df, freq)

    logger.info("\tCreating IOB and insulin availability columns...")
    df = create_iob_and_ins_availability_cols(df, freq)

    logger.info("\tDone deriving features.\n")
    return df
