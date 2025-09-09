import logging

import pandas as pd

from src.data.preprocessing.feature_engineering import derive_features

logger = logging.getLogger(__name__)

required_columns = [
    "datetime",  # Datetime of the data (not the index)
    "p_num",  # Patient number (id)
    "bg_mM",  # Blood glucose in mmol/L
    "msg_type",  # Message type: ANNOUNCE_MEAL | ''
    "food_g",  # Carbs in grams
    "dose_units",  # Insulin units
]


def preprocessing_pipeline(p_num: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    The entry point for the preprocessing pipeline.
    This function does the following:
    1. Ensures datetime index
    2. Groups data by day starting at configured time
    3. Derive iob, cob, insulin availability and carb availability features

    Returns:
        Dictionary mapping patient IDs to their processed DataFrames
    """
    # TODO: Create an option for both serial and parallel processing of the multipatient files.
    logger.info("==============================")
    logger.info(f"Preprocessing patient {p_num}")
    logger.info("==============================")

    check_data_format(df)
    patient_df = df.copy(deep=True)
    processed_df = derive_features(patient_df)
    return processed_df


def check_data_format(df: pd.DataFrame) -> bool:
    """
    Checks if the data is in the correct format.
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
