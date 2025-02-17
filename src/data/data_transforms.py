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
from src.data.insulin_model.constants import IOB_COL, INSULIN_AVAIL_COL
from src.data.insulin_model.insulin_model import (
    calculate_insulin_availability_and_iob_single_delivery,
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


def create_iob_and_ins_availability_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the insulin availability (INS_AVAIL_COL) and insulin on board (IOB_COL)
    for each insulin dose in the dataframe.

    Process:
    - Identify rows where insulin is administered (`insulin-0:00` is not null and greater than 0).
    - Simulate insulin availability and IOB for each dose using the `calculate_insulin_availability_and_iob_single_delivery` function.
    - Iterate through subsequent time steps until the full insulin absorption time is covered.
    - Handles missing `time_diff` values by skipping affected rows.

    Parameters:
    df (pd.DataFrame): DataFrame containing insulin administration times and time differences.

    Returns:
    pd.DataFrame: Updated DataFrame with computed `ins_availability` and `iob` columns.
    """

    # Add new columns initialized to 0
    df[INSULIN_AVAIL_COL] = 0.0
    df[IOB_COL] = 0.0

    # Loop through each insulin injection event, excluding 0.0 values
    for ins_time in df.index[(df["insulin-0:00"].notna()) & (df["insulin-0:00"] > 0)]:
        insulin_dose = df.loc[ins_time, "insulin-0:00"]

        # Simulate insulin dynamics
        ins_avail, iob, _, _, _ = (
            calculate_insulin_availability_and_iob_single_delivery(
                insulin_dose, TS_MIN, T_ACTION_MAX_MIN
            )
        )

        next_index = ins_time + 1
        time_since_insulin = 0

        while next_index in df.index and time_since_insulin < T_ACTION_MAX_MIN:
            time_diff = df.loc[next_index, "time_diff"]
            if pd.isna(time_diff):
                break  # Stop processing if time_diff is missing

            time_diff_int = abs(time_diff.components.minutes)
            time_since_insulin += time_diff_int

            if time_since_insulin < T_ACTION_MAX_MIN:
                df.loc[next_index, INSULIN_AVAIL_COL] += ins_avail[time_since_insulin]
                df.loc[next_index, IOB_COL] += iob[time_since_insulin]

            next_index += 1

    return df


def create_datetime_index(
    df: pd.DataFrame, start_date: str = "2025-01-01"
) -> pd.DataFrame:
    if "time_diff" not in df.columns:
        df = create_time_diff_cols(df)

    # Create a str datetime column
    df["datetime"] = pd.to_datetime(
        start_date + " " + df["time"], format="%Y-%m-%d %H:%M:%S"
    )

    # Convert time_diff to a timedelta object
    # time_diff is the difference in time between the current row and the next row
    df["time_diff"] = pd.to_timedelta(df["time_diff"])

    # time_diff is negative at times e.g. when the one current row is at 23:59:59 and the next is at 00:00:00
    # Create a new column with the absolute value of time_diff
    # Have to assume that two entries are not separated by more than 24 hours
    # Since we were not given date, only time, so not considering days in this calculation
    df["time_diff_abs"] = df["time_diff"].apply(
        lambda x: pd.Timedelta(
            hours=x.components.hours,
            minutes=x.components.minutes,
            seconds=x.components.seconds,
        )
        if pd.notna(x)
        else pd.Timedelta(0)
    )

    # Create a cumulative sum of the absolute time differences
    df["cumulative_diff"] = (
        df.groupby("p_num")["time_diff_abs"].cumsum().fillna(pd.Timedelta(0))
    )

    # Create a fixed start timestamp for each patient
    start_times = df.groupby("p_num")["datetime"].transform("first")

    # Add cumulative time difference to the fixed start timestamp
    df["datetime"] = start_times + df["cumulative_diff"]

    # Drop the intermediate columns
    df.drop(columns=["time_diff", "time_diff_abs", "cumulative_diff"], inplace=True)

    return df
