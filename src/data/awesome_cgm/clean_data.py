import pandas as pd


# Gluroo data one dataframe per patient
def clean_cgm_data(
    df_raw: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    Cleans Aleppo dataset
    1. Ensures datetime index
    2. Coerces timestamps to regular intervals
    3. Groups data by day starting at configured time
    4. Removes consecutive NaN values exceeding threshold

    Args:
        df_raw (pd.DataFrame): Input DataFrame containing Gluroo data. One dataframe per patient.
    config : dict
        Configuration dictionary containing:
        - max_consecutive_nan_values_per_day: Maximum allowed consecutive NaN values per day. The entire day is removed if this threshold is exceeded.
        - coerse_time_interval: Time interval to coerce timestamps to
        - day_start_time: Time of day to use as start of day
    Returns:
        pd.DataFrame: Cleaned DataFrame with all transformations applied
    """

    if config is None:
        config = {
            "max_consecutive_nan_values_per_day": 36,
            "coerse_time_interval": pd.Timedelta(minutes=5),
            "day_start_time": pd.Timedelta(hours=4),
        }

    max_consecutive_nan_values_per_day: int = config[
        "max_consecutive_nan_values_per_day"
    ]

    day_start_time: pd.Timedelta = config["day_start_time"]

    df = df_raw.copy()
    df = ensure_datetime_index(df)
    df.index = df.index.tz_localize(None)

    # From meal identification repo
    # df = coerce_time_fn(data=df, coerse_time_interval=coerse_time_interval)
    df["day_start_shift"] = (df.index - day_start_time).date
    df = erase_consecutive_nan_values(df, max_consecutive_nan_values_per_day)

    # Translate data to the correct format
    df = data_translation(df)

    return df


def data_translation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Translates the data to the correct format.
    TODO:
    - Translate dose units to iob and insulin_availability
    - Translate carbs to cob, carb_availability
    - Greg's data might have HR, steps and activity data?
    """

    df = df_raw.copy()
    df = df.rename(
        columns={
            "bgl": "bg-0:00",
            "food_g": "carbs-0:00",
        }
    )
    df["datetime"] = df.index

    # Convert blood glucose from mg/dL to mmol/L
    df["bg-0:00"] = (df["bg-0:00"] / 18.0).round(2)

    return df


def ensure_datetime_index(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensures DataFrame has a datetime index.

    Args:
        data (pd.DataFrame): Input DataFrame that either has a datetime index or a 'date' column
        that can be converted to datetime.

    Returns:
        pd.DataFrame: DataFrame with sorted datetime index.
    """
    df = data.copy()

    # Check if the index is already a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # If not, set 'date' column as index and convert to DatetimeIndex
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            raise KeyError(
                "DataFrame must have either a 'date' column or a DatetimeIndex."
            )

    df.index = pd.DatetimeIndex(df.index)

    return df


def erase_meal_overlap_fn(patient_df, meal_length, min_carbs):
    """
    Process the DataFrame to handle meal overlaps.

    Parameters
    ----------
    patient_df : pd.DataFrame
        The input DataFrame with columns 'msg_type', 'food_g', and a datetime index.
    meal_length : pd.Timedelta
        The duration to look ahead for meal events.
    min_carbs : int
        Minimum amount of carbohydrates to consider a meal.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with meal overlaps handled.
    """
    announce_meal_mask = patient_df["msg_type"] == "ANNOUNCE_MEAL"
    announce_meal_indices = patient_df[announce_meal_mask].index

    for idx in announce_meal_indices:
        # Skip meals below the carbohydrate threshold
        if patient_df.at[idx, "food_g"] <= min_carbs:
            patient_df.at[idx, "msg_type"] = "LOW_CARB_MEAL"
            continue

        # Define the time window
        window_end = idx + meal_length

        # Get the events within the time window, excluding the current event
        window_events = patient_df.loc[idx + pd.Timedelta(seconds=1) : window_end]

        # Sum the 'food_g' counts greater than 0 within the window
        food_g_sum = window_events[window_events["food_g"] > 0]["food_g"].sum()

        # Add the sum to the original 'ANNOUNCE_MEAL' event
        patient_df.at[idx, "food_g"] += food_g_sum

        # Erase the other events that fell within the window
        patient_df.loc[window_events.index, ["food_g", "msg_type"]] = [0, ""]

    return patient_df


def erase_consecutive_nan_values(
    patient_df: pd.DataFrame, max_consecutive_nan_values_per_day: int
):
    """
    1. If there are more than max_consecutive_nan_values_per_day consecutive NaN values in a given day, then delete that day from the dataframe.
    2. If there are less than max_consecutive_nan_values_per_day consecutive NaN values in a given day, then delete the NaN values from that day.
    ------
    Parameters:
        patient_df: pd.DataFrame
            The input DataFrame with a datetime index.
        max_consecutive_nan_values_per_day: int
            The maximum number of consecutive NaN values allowed in a given day. If more than this number of consecutive NaN values are found in a day, then delete that day from the dataframe. Otherwise, delete the NaN values from that day.
    Returns:
        pd.DataFrame
            The processed DataFrame with consecutive NaN values handled.
    """
    # Create a copy to avoid modifying original
    df = patient_df.copy()

    # Add day column for grouping
    df["day"] = df.index.date

    # Process each day
    days_to_keep = []
    for day, day_data in df.groupby("day"):
        # Get boolean mask of NaN values
        nan_mask = day_data["bgl"].isnull()

        # Count consecutive NaNs
        consecutive_nans = 0
        max_consecutive = 0
        for is_nan in nan_mask:
            if is_nan:
                consecutive_nans += 1
                max_consecutive = max(max_consecutive, consecutive_nans)
            else:
                consecutive_nans = 0

        # Keep day if max consecutive NaNs is within limit
        if max_consecutive <= max_consecutive_nan_values_per_day:
            days_to_keep.append(day)

    # Filter to keep only valid days
    result_df = df[df["day"].isin(days_to_keep)].copy()

    # Drop the temporary day column
    result_df.drop("day", axis=1, inplace=True)

    # Drop remaining NaN values since they consecutively dont form a long enough chain
    result_df = result_df.dropna(subset=["bgl"])

    return result_df
