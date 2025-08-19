import pandas as pd
import logging
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
""" from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
) """


logger = logging.getLogger(__name__)

required_columns = [
    "datetime",
    "food_g",  # Carbs in grams
    "dose_units",  # Insulin units
]


def check_data_format(df: pd.DataFrame) -> bool:
    """
    Checks if the data is in the correct format.
    """
    required_columns_set = set(required_columns)
    df_columns_set = set(df.columns)
    if not required_columns_set.issubset(df_columns_set):
        raise ValueError(
            f"Data is not in the correct format. Please make sure these columns are present: {required_columns_set - df_columns_set}"
        )
    return True


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function does the following:
    - Check if the data has a datetime index
    - Fill in gaps so that the data doesn't have time jump and has regular time intervals
    - Create COB and carb availability columns from carbs (food_g)
    - Create IOB and insulin availability columns from insulin (dose_units)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    logger.info("Deriving features...")
    logger.info("Filling in gaps...")
    # df = ensure_regular_time_intervals(df)

    logger.info("Creating COB and carb availability columns...")
    df = create_cob_and_carb_availability_cols(df)
    #print(df.head())
    logger.info(
        "Creating IOB and insulin availability columns. This may take a while depending on the size of the data."
    )
    #df = create_iob_and_ins_availability_cols(df)

    logger.info("Done deriving features.\n")
    return df
