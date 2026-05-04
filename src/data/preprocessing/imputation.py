"""Imputation functions for handling missing data in time series datasets.

TODO: Add per-column imputation configuration to the data config layer
(e.g. in HoldoutConfig or a dedicated preprocessing config YAML) so that
imputation strategies can be specified per dataset/column rather than
relying on column-name heuristics. See configs/data/ for data configs.
"""

import pandas as pd
from sktime.transformations.series.impute import Imputer


def impute_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    bg_method="linear",
    hr_method="linear",
    step_method="constant",
    cal_method="constant",
) -> pd.DataFrame:
    """Imputes missing values in specified columns of a dataframe using different methods based on the data type.

    Args:
        df (pd.DataFrame): Input dataframe containing missing values
        columns (list): List of column names to impute missing values for
        bg_method (str, optional): Imputation method for blood glucose data.
            Valid values: 'linear', 'nearest'. Defaults to "linear".
        hr_method (str, optional): Imputation method for heart rate data.
            Valid values: 'linear', 'nearest'. Defaults to "linear".
        step_method (str, optional): Imputation method for step count data.
            Valid values: 'constant'.
        cal_method (str, optional): Imputation method for calorie data.
            Valid values: 'constant'.

    Returns:
        pd.DataFrame: Copy of input dataframe with missing values imputed using appropriate methods for each data type
    """
    df_imputed = df.copy()
    transform = None

    for col in columns:
        if col in df.columns:
            if "bg" in col.lower():
                transform = Imputer(method=bg_method)
            elif "hr" in col.lower():
                # Use linear or nearest neighbor interpolation for heart rate
                # TODO: Need more research on this
                transform = Imputer(method=hr_method)
            elif "step" in col.lower():
                # Use constant imputation with 0 for steps
                transform = Imputer(method=step_method, value=0)
            elif "cals" in col.lower():
                # Use constant imputation with minimum value for calories
                min_val = df[col].min()
                transform = Imputer(method=cal_method, value=min_val)

            if transform is not None:
                df_imputed[col] = transform.fit_transform(df[col].to_frame())

    return df_imputed
