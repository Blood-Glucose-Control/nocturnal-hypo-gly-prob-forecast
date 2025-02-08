import pandas as pd

from src.data.carb_model.carb_model import (
    calculate_carb_availability_and_cob_single_meal,
)
from src.data.carb_model.constants import (
    CARB_ABSORPTION,
    TS_MIN,
    T_ACTION_MAX_MIN,
    COB_COL,
    CARB_AVAIL_COL,
)


def create_time_diff_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'time_diff' column to the DataFrame that represents the time difference between
    consecutive rows for each patient, grouped by patient. The time difference is in minutes.
    Also adds a 'datetime' column to the DataFrame.
    """
    df["datetime"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    df["time_diff"] = df.groupby("p_num")[
        "datetime"
    ].diff()  # Compute time differences grouped by 'p_num'
    df.drop(columns=["datetime"], inplace=True)
    return df


def create_cob_and_carb_availability_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: This method assumes that TS_MIN = 1 minute.

    Computes the carbohydrate availability (CARB_AVAIL_COL) and carbohydrate on board (COB_COL)
    for each meal announcement in the dataframe.

    Process:
    - Identify rows where a meal is announced (`carbs-0:00` is not null).
    - Simulate carb availability and COB for each meal using the `calculate_carb_availability_and_cob_single_meal` function.
    - Determine the number of simulation steps (`sim_steps`), which is based on the total absorption period
      (T_ACTION_MAX_MIN) and the time step size (TS_MIN).
    - Iterate through subsequent time steps until the full absorption time is covered:
        - If the time step exists in the dataframe, update `carb_availability` and `cob` accordingly.
        - Stop iterating if `time_since_meal` exceeds the total absorption time (T_ACTION_MAX_MIN).
    - Handles missing `time_diff` values by skipping affected rows.

    Parameters:
    df (pd.DataFrame): DataFrame containing meal announcement times and time differences.

    Returns:
    pd.DataFrame: Updated DataFrame with computed `carb_availability` and `cob` columns.
    """

    df[COB_COL] = 0.0
    df[CARB_AVAIL_COL] = 0.0

    for meal_time in df.index[df["carbs-0:00"].notna()]:
        meal_value = df.loc[meal_time, "carbs-0:00"]
        meal_avail, cob = calculate_carb_availability_and_cob_single_meal(
            meal_value, CARB_ABSORPTION, TS_MIN, T_ACTION_MAX_MIN
        )

        next_index = meal_time + 1
        time_since_meal = 0

        while next_index in df.index and time_since_meal < T_ACTION_MAX_MIN:
            time_diff = df.loc[next_index, "time_diff"]
            if pd.isna(time_diff):
                break

            time_diff_int = abs(time_diff.components.minutes)
            time_since_meal += time_diff_int
            if time_since_meal < T_ACTION_MAX_MIN:
                df.loc[next_index, CARB_AVAIL_COL] += meal_avail[time_since_meal]
                df.loc[next_index, COB_COL] += cob[time_since_meal]

            next_index += 1

    return df
