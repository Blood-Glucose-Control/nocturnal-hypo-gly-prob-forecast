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

from src.data.models import ColumnNames
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)
from src.data.preprocessing.generic_cleaning import reduce_fp_precision
from src.data.preprocessing.sampling import (
    ensure_regular_time_intervals,
    ensure_regular_time_intervals_with_aggregation,
)
from src.data.preprocessing.time_processing import get_most_common_time_interval

logger = logging.getLogger(__name__)

required_columns = [
    ColumnNames.DATETIME.value,  # Datetime of the data (not the index)
    ColumnNames.FOOD_G.value,  # Carbs in grams
    ColumnNames.DOSE_UNITS.value,  # Insulin units
]


# TODO: We should't really be adding dose_units because each insulin has different activation curves.
def rollover_basal_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll over the basal rate to the next few rows if the rate is not null.
    The rollover is based on the duration of the basal rate in minutes.
    For example, if a row has a basal rate of 1 unit/hr and the interval is 5 minutes for a basal duration of one hour,
    then the next 12 rows (one hour) will have a dose of 1/12 units.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index and constant interval (e.g. 5 minutes)
                          ColumnNames.RATE.value (basal rate units/hr)
                          ColumnNames.BASAL_DURATION_MINS.value (basal duration in minutes)
    Returns:
        pd.DataFrame: Enhanced DataFrame with basal rate added to dose_units
    """
    if (
        ColumnNames.RATE.value not in df.columns
        or ColumnNames.BASAL_DURATION_MINS.value not in df.columns
    ):
        logger.warning(
            f"No {ColumnNames.RATE.value} or {ColumnNames.BASAL_DURATION_MINS.value} column found. Returning original dataframe."
        )
        return df

    if ColumnNames.DOSE_UNITS.value not in df.columns:
        logger.warning(
            f"No {ColumnNames.DOSE_UNITS.value} column found. Returning original dataframe."
        )
        raise ValueError(
            f"ROLLOVER_BASAL_RATE function: DataFrame must contain {ColumnNames.DOSE_UNITS.value} column"
        )

    df = df.copy()
    freq = get_most_common_time_interval(df)

    # Calculate rows per hour (e.g., 12 rows for 5-minute intervals)
    rows_per_hour = 60 // freq
    for i in range(len(df)):
        if pd.notna(df[ColumnNames.RATE.value].iloc[i]):
            rate = df[ColumnNames.RATE.value].iloc[i]
            duration_mins = df[ColumnNames.BASAL_DURATION_MINS.value].iloc[i]
            dose_per_row = rate / rows_per_hour
            rows_to_rollover = int(duration_mins / freq)

            # Find where this rate stops (end of data or end of duration)
            end_idx = min(i + rows_to_rollover, len(df))

            # Check if there's another rate change within this range
            # The next rate change should override the current rate.
            for j in range(i + 1, end_idx):
                if pd.notna(df[ColumnNames.RATE.value].iloc[j]):
                    end_idx = j  # Stop before the next rate change
                    break

            df.iloc[i:end_idx][ColumnNames.DOSE_UNITS.value] += dose_per_row

    return df


def create_physiological_features(
    df: pd.DataFrame,
    use_aggregation: bool = False,
) -> pd.DataFrame:
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
        use_aggregation (bool, optional): Whether to use aggregation to ensure regular time intervals.
                                          If True, will consider all rows within the same regular time interval.
                                          If False, will only consider the first row within the regular time interval.

    Returns:
        pd.DataFrame: Enhanced DataFrame with original data plus derived physiological
                     features including COB, IOB, and availability metrics

    Raises:
        ValueError: If the DataFrame does not have a datetime index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    logger.info("create_physiological_features(): Deriving features...")
    if use_aggregation:
        logger.info("\tEnsuring regular time intervals with aggregation...")
        df, freq = ensure_regular_time_intervals_with_aggregation(df)
        # df.to_csv("resampled_with_aggregation.csv")  # TODO: Remove this. Debugging only
    else:
        logger.info("\tEnsuring regular time intervals...")
        df, freq = ensure_regular_time_intervals(df)

    logger.info("\tRollover basal rate...")
    df = rollover_basal_rate(df)

    logger.info(
        "\tCreating COB/IOB and availability columns. This may take a while depending on the size of the data."
    )
    logger.info("\tCreating COB and carb availability columns...")
    df = create_cob_and_carb_availability_cols(df, freq)

    logger.info("\tCreating IOB and insulin availability columns...")
    df = create_iob_and_ins_availability_cols(df, freq)

    logger.info("\tReducing floating point precision...")
    df = reduce_fp_precision(df)

    logger.info("\tDone deriving features.\n")
    return df
