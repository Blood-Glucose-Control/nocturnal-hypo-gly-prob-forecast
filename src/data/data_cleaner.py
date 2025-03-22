"""Data cleaning functions for the various datasets."""

import pandas as pd
from src.data.kaggle_brisT1D.kaggle_data_cleaner import clean_brist1d_train_data
from src.data.gluroo.gluroo_data_cleaner import clean_gluroo_data


def clean_data(
    data: pd.DataFrame, data_source_name="kaggle_brisT1D", data_type="train"
) -> pd.DataFrame:
    """
    Cleans the input data based on the specified data source name.

    Args:
        data (pd.DataFrame): The input data to be cleaned.
        data_source_name (str): The name of the data source. Default is "kaggle_brisT1D".

    Returns:
        pd.DataFrame: The cleaned data.
    """
    # leaving data_type in for now, because we may use test/train in the near future.
    if data_source_name == "kaggle_brisT1D":
        if data_type == "train":
            clean_brist1d_train_data(data)
        else:
            raise NotImplementedError(
                "Use method clean_bris1d_test_data for test brist1d data"
            )
    elif data_source_name == "gluroo":
        clean_gluroo_data(data)
    else:
        raise NotImplementedError("data_source_name not supported")

    return data
