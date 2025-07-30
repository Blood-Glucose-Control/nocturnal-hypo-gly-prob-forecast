import pandas as pd


def data_translation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Translates the data to a standardized format for data_cleaning_pipeline

    1. blood glucose values from mg/dL to mmol/L.
    2. Convert date to datetime
    3. Adds patient identifier "glu001" as p_num column

    Args:
        df_raw (pd.DataFrame): Input DataFrame with raw Gluroo data

    Returns:
        pd.DataFrame: DataFrame with standardized column names and formats

    TODO:
    - Gluroo's data might have HR, steps and activity data in the future.
    """

    df = df_raw.copy()
    # TODO: Remove the dependency of p_num. Kaggle data is the very few dataset where there are multiple patients in the same file.
    df["p_num"] = "glu001"
    df["datetime"] = df["date"]

    # Convert blood glucose from mg/dL to mmol/L
    df["bg_mM"] = (df["bgl"] / 18.0).round(2)

    return df
