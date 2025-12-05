# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

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
from typing import Literal

import pandas as pd
import numpy as np

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


# TODO: We should't really be adding dose_units because each insulin has different activation curves.
# NOTE: Adding basal to dose_units is appropriate for pump therapy since both basal and bolus
# use the same rapid-acting insulin (Humalog, Novolog, etc.) with identical pharmacokinetics.
# OpenAPS uses netIOB (basal + bolus combined) for BG prediction.
# This would NOT be appropriate for MDI therapy where long-acting basal (e.g., Lantus ~24hr)
# has different pharmacokinetics than rapid-acting bolus (~5hr).
# See: https://openaps.readthedocs.io/en/latest/docs/While%20You%20Wait%20For%20Gear/understanding-insulin-on-board-calculations.html
# See: https://developer.tidepool.org/data-model/device-data/types/basal/
def rollover_basal_rate(
    df: pd.DataFrame,
    delivery_type: Literal["temp", "automated"],
) -> pd.DataFrame:
    """
    Roll over the basal rate to add basal insulin doses to dose_units column.

    Supports two delivery types based on Tidepool data model:
    - temp: User-initiated temporary basal with explicit duration (basal_duration_mins column)
    - automated: Algorithm-driven basal (Control-IQ, Loop, OpenAPS) where duration is
                 calculated from event sequence (rate applies until next rate change)

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index and constant interval (e.g. 5 minutes)
                          Required: ColumnNames.RATE.value (basal rate units/hr)
                          Required for temp: ColumnNames.BASAL_DURATION_MINS.value (duration in minutes)
        delivery_type: Type of basal delivery - "temp" or "automated" (required, no default)

    Returns:
        pd.DataFrame: DataFrame with basal rate converted to doses and added to dose_units
    """
    if ColumnNames.RATE.value not in df.columns:
        logger.warning(
            f"No {ColumnNames.RATE.value} column found. Returning original dataframe."
        )
        return df

    if ColumnNames.DOSE_UNITS.value not in df.columns:
        raise ValueError(
            f"ROLLOVER_BASAL_RATE function: DataFrame must contain {ColumnNames.DOSE_UNITS.value} column"
        )

    if delivery_type == "temp":
        if ColumnNames.BASAL_DURATION_MINS.value not in df.columns:
            raise ValueError(
                f"delivery_type='temp' requires {ColumnNames.BASAL_DURATION_MINS.value} column"
            )
        return _rollover_basal_temp(df)
    elif delivery_type == "automated":
        return _rollover_basal_automated(df)
    else:
        raise ValueError(
            f"Unknown delivery_type: {delivery_type}. Must be 'temp' or 'automated'"
        )


def _rollover_basal_temp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle temp basal with explicit duration from basal_duration_mins column.
    """
    df = df.copy()
    freq = get_most_common_time_interval(df)
    rows_per_hour = 60 // freq

    for i in range(len(df)):
        if pd.notna(df[ColumnNames.RATE.value].iloc[i]):
            rate = df[ColumnNames.RATE.value].iloc[i]
            duration_mins = df[ColumnNames.BASAL_DURATION_MINS.value].iloc[i]
            if pd.isna(duration_mins):
                continue  # Skip if duration is missing
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

            df.loc[df.index[i:end_idx], ColumnNames.DOSE_UNITS.value] += dose_per_row

    return df


def _rollover_basal_automated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle automated basal (Control-IQ, Loop, OpenAPS) where duration is calculated
    from event sequence - rate applies until next rate change event.

    Uses forward-fill to propagate rates, then converts to dose per interval.
    """
    df = df.copy()
    freq = get_most_common_time_interval(df)
    rows_per_hour = 60 // freq

    # Forward-fill rate: each rate persists until the next rate change
    filled_rate = df[ColumnNames.RATE.value].ffill()

    # Convert rate (U/hr) to dose per interval and add to dose_units
    dose_per_row = filled_rate / rows_per_hour
    df[ColumnNames.DOSE_UNITS.value] += dose_per_row.fillna(0)

    return df


def create_physiological_features(
    df: pd.DataFrame,
    use_aggregation: bool = False,
    basal_delivery_type: Literal["temp", "automated"] | None = None,
) -> pd.DataFrame:
    """
    Derives physiological features from patient glucose monitoring data.

    This function performs comprehensive feature engineering by:
    1. Validating the DataFrame has a datetime index
    2. Ensuring regular time intervals by filling gaps in the time series
    3. Creating carbs-on-board (COB) and carb availability features from food intake
    4. Creating insulin-on-board (IOB) and insulin availability features from dose data

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index. Optional columns:
                          'food_g' (carbs) enables COB calculation,
                          'dose_units' (insulin) enables IOB calculation,
                          'rate' (basal) enables basal rollover.
        use_aggregation (bool, optional): Whether to use aggregation to ensure regular time intervals.
                                          If True, will consider all rows within the same regular time interval.
                                          If False, will only consider the first row within the regular time interval.
        basal_delivery_type: Type of basal delivery for rollover calculation.
                            Required if 'rate' column is present. Options:
                            - "temp": User-initiated temp basal with explicit duration
                            - "automated": Algorithm-driven basal (Control-IQ, Loop, OpenAPS)

    Returns:
        pd.DataFrame: Enhanced DataFrame with original data plus derived physiological
                     features. COB/IOB will be NaN if the corresponding columns are missing.

    Raises:
        ValueError: If the DataFrame does not have a datetime index
        ValueError: If 'rate' column exists but basal_delivery_type is not provided
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    logger.info("create_physiological_features(): Deriving features...")
    if use_aggregation:
        logger.info("\tEnsuring regular time intervals with aggregation...")
        df, freq = ensure_regular_time_intervals_with_aggregation(df)
    else:
        logger.info("\tEnsuring regular time intervals...")
        df, freq = ensure_regular_time_intervals(df)

    # Conditionally apply basal rollover (requires both dose_units and rate columns)
    if (
        ColumnNames.DOSE_UNITS.value in df.columns
        and ColumnNames.RATE.value in df.columns
    ):
        if basal_delivery_type is None:
            raise ValueError(
                "basal_delivery_type must be specified when 'rate' column is present. "
                "Use 'temp' for user-initiated temp basals with explicit duration, "
                "or 'automated' for algorithm-driven basals (Control-IQ, Loop, OpenAPS)."
            )
        logger.info(f"\tRollover basal rate (delivery_type={basal_delivery_type})...")
        df = rollover_basal_rate(df, delivery_type=basal_delivery_type)
    else:
        logger.info("\tSkipping basal rollover (missing dose_units or rate column)")

    logger.info(
        "\tCreating COB/IOB and availability columns. "
        "This may take a while depending on the size of the data."
    )

    # Conditionally compute COB (requires food_g column with at least some data)
    if (
        ColumnNames.FOOD_G.value in df.columns
        and df[ColumnNames.FOOD_G.value].notna().any()
    ):
        logger.info("\tCreating COB and carb availability columns...")
        df = create_cob_and_carb_availability_cols(df, freq)
    else:
        logger.info("\tSkipping COB (no food_g column or all NaN) - setting to NaN")
        df[ColumnNames.COB.value] = np.nan
        df[ColumnNames.CARB_AVAILABILITY.value] = np.nan

    # Conditionally compute IOB (requires dose_units column with at least some data)
    if (
        ColumnNames.DOSE_UNITS.value in df.columns
        and df[ColumnNames.DOSE_UNITS.value].notna().any()
    ):
        logger.info("\tCreating IOB and insulin availability columns...")
        df = create_iob_and_ins_availability_cols(df, freq)
    else:
        logger.info("\tSkipping IOB (no dose_units column or all NaN) - setting to NaN")
        df[ColumnNames.IOB.value] = np.nan
        df[ColumnNames.INSULIN_AVAILABILITY.value] = np.nan

    logger.info("\tReducing floating point precision...")
    df = reduce_fp_precision(df)

    logger.info("\tDone deriving features.\n")
    return df
