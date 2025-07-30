import pandas as pd
from src.data.preprocessing.generic_cleaning import clean_dataset
from src.data.preprocessing.feature_engineering import derive_features


required_columns = [
    "datetime",  # Datetime of the data (not the index)
    "p_num",  # Patient number (id)
    "bg_mM",  # Blood glucose in mmol/L
    "msg_type",  # Message type: ANNOUNCE_MEAL | ''
    "food_g",  # Carbs in grams
    "steps",  # Steps
    "dose_units",  # Insulin units
]


def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    The entry point for the preprocessing pipeline.
    This function does the following:
    1. Ensures datetime index
    2. Coerces timestamps to regular intervals
    3. Groups data by day starting at configured time
    4. Removes consecutive NaN values exceeding threshold
    5. Handles meal overlaps
    6. Keeps only top N carb meals
    7. Derive iob, cob, insulin availability and carb availability features
    """
    check_data_format(df)
    df = df.copy(deep=True)
    df = clean_dataset(df)
    return derive_features(df)


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
