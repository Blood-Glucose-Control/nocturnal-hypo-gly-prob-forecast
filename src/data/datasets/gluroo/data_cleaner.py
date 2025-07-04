"""
Gluroo Data Cleaning Module

This module provides functions for processing and cleaning Gluroo diabetes data.
It handles tasks such as standardizing timestamps, handling missing values,
processing meal data, and transforming data into a consistent format.

Key functionality includes:
- Converting timestamps to regular intervals
- Identifying and filtering days with excessive missing data
- Processing meal announcements and handling overlaps
- Selecting top carbohydrate meals per day
- Standardizing column names and formats
"""

import pandas as pd


# Gluroo data one dataframe per patient
def clean_gluroo_data(
    df_raw: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Cleans the Gluroo dataset by applying several transformations:
    1. Ensures datetime index
    2. Coerces timestamps to regular intervals
    3. Groups data by day starting at configured time
    4. Removes consecutive NaN values exceeding threshold
    5. Handles meal overlaps
    6. Keeps only top N carb meals

    Args:
        df_raw (pd.DataFrame): Input DataFrame containing Gluroo data. One dataframe per patient.
    config : dict
        Configuration dictionary containing:
        - max_consecutive_nan_values_per_day: Maximum allowed consecutive NaN values per day. The entire day is removed if this threshold is exceeded.
        - coerse_time_interval: Time interval to coerce timestamps to
        - day_start_time: Time of day to use as start of day
        - min_carbs: Minimum carbs threshold for meal. Meal less than this threshold are filtered out.
        - meal_length: Time window for considering meal duration
        - n_top_carb_meals: Number of highest-carb meals to keep per day

    Returns:
        pd.DataFrame: Cleaned DataFrame with all transformations applied
    """

    if config is None:
        config = {
            "max_consecutive_nan_values_per_day": 36,
            "coerse_time_interval": pd.Timedelta(minutes=5),
            "day_start_time": pd.Timedelta(hours=4),
            "min_carbs": 5,
            "meal_length": pd.Timedelta(hours=2),
            "n_top_carb_meals": 3,
        }

    max_consecutive_nan_values_per_day: int = config[
        "max_consecutive_nan_values_per_day"
    ]
    coerse_time_interval: pd.Timedelta = config["coerse_time_interval"]
    day_start_time: pd.Timedelta = config["day_start_time"]
    min_carbs: int = config["min_carbs"]
    meal_length: pd.Timedelta = config["meal_length"]
    n_top_carb_meals: int = config["n_top_carb_meals"]

    df = df_raw.copy()

    df = ensure_datetime_index(df)
    # Explicitly tell the type checker that df.index is a DatetimeIndex
    from typing import cast

    df.index = cast(pd.DatetimeIndex, df.index)

    # From meal identification repo
    df = coerce_time_fn(data=df, coerse_time_interval=coerse_time_interval)
    df["day_start_shift"] = (df.index - day_start_time).to_series().dt.date
    df = erase_consecutive_nan_values(df, max_consecutive_nan_values_per_day)
    df = erase_meal_overlap_fn(df, meal_length, min_carbs)
    df = keep_top_n_carb_meals(df, n_top_carb_meals=n_top_carb_meals)

    # Translate data to the correct format
    df = data_translation(df)

    return df


def data_translation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Translates the data to a standardized format with consistent column names.

    Performs column renaming, adds required metadata columns, and converts
    blood glucose values from mg/dL to mmol/L.

    Args:
        df_raw (pd.DataFrame): Input DataFrame with raw Gluroo data

    Returns:
        pd.DataFrame: DataFrame with standardized column names and formats

    Notes:
        - Blood glucose values are converted from mg/dL to mmol/L using the factor 18.0
        - Adds a datetime column that duplicates the index for compatibility
        - Adds patient identifier "glu001" as p_num column
    TODO:
    - Gluroo's data might have HR, steps and activity data in the future.
    """

    df = df_raw.copy()
    df = df.rename(
        columns={
            "bgl": "bg-0:00",
            "food_g": "carbs-0:00",
            "dose_units": "insulin-0:00",
        }
    )
    df["datetime"] = df.index
    df["p_num"] = (
        "glu001"  # TODO: Remove the dependency of p_num. Kaggle data is the very few dataset where there are multiple patients in the same file.
    )

    # Convert blood glucose from mg/dL to mmol/L
    df["bg-0:00"] = (df["bg-0:00"] / 18.0).round(2)

    return df


def ensure_datetime_index(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensures DataFrame has a datetime index in UTC format.

    Converts index to DatetimeIndex if not already in that format or uses the 'date' column
    as the index if available. Handles timezone conversion to ensure consistent datetime handling.

    Args:
        data (pd.DataFrame): Input DataFrame that either has a datetime index or a 'date' column
                           that can be converted to datetime.

    Returns:
        pd.DataFrame: DataFrame with sorted UTC datetime index.

    Raises:
        KeyError: If neither a DatetimeIndex nor a 'date' column is available
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

    # Convert to UTC to handle DST transitions
    df.index = pd.to_datetime(df.index, utc=True)

    return df


def coerce_time_fn(data, coerse_time_interval):
    """
    Coerces the time interval of data to a regular frequency and handles meal announcements.

    This function separates meal announcement data from other data, resamples both at the specified
    interval, and then recombines them while preserving meal information. It ensures consistent
    time intervals throughout the dataset.

    Args:
        data (pd.DataFrame): The input DataFrame with a 'date' index and columns including
                           'msg_type', 'bgl', and 'food_g'.
        coerse_time_interval (pd.Timedelta): The interval for time resampling.

    Returns:
        pd.DataFrame: DataFrame with regularized time intervals and preserved meal data.

    Raises:
        KeyError: If 'date' is not the index name
        TypeError: If coerse_time_interval is not a pandas Timedelta object
    """
    # Ensure 'date' column exists
    if "date" != data.index.name:
        raise KeyError(f"'date' column should be index, got {data.index.name} instead")

    if not isinstance(coerse_time_interval, pd.Timedelta):
        raise TypeError(
            f"coerse_time_interval must be a pandas Timedelta object, got {type(coerse_time_interval)} instead"
        )

    # Separate meal announcements and non-meal data
    meal_announcements = data[data["msg_type"] == "ANNOUNCE_MEAL"].copy()
    non_meals = data[data["msg_type"] != "ANNOUNCE_MEAL"].copy()

    non_meals = non_meals.resample(coerse_time_interval).first()
    start_time = non_meals.index.min()

    # Resample meal announcements separately and align with non_meal
    meal_announcements = meal_announcements.resample(
        coerse_time_interval, origin=start_time
    ).first()

    # Join the two DataFrames
    data_resampled = non_meals.join(meal_announcements, how="left", rsuffix="_meal")

    # Combine the columns
    for col in ["bgl", "msg_type", "food_g"]:
        meal_col = f"{col}_meal"
        if meal_col in data_resampled.columns:
            data_resampled[col] = data_resampled[col + "_meal"].combine_first(
                data_resampled[col]
            )

    # Retain 'food_g_keep' from meal announcements data_resampled = data.resample(resample_rule).first()
    data_resampled["food_g_keep"] = data_resampled.get("food_g_meal", 0)

    # Identify columns that end with '_meal'
    columns_to_drop = data_resampled.filter(regex="_meal$").columns

    # Drop the identified columns
    data_resampled = data_resampled.drop(columns=columns_to_drop)

    print("Columns after coercing time:", data_resampled.columns.tolist())

    return data_resampled


def remove_num_meal(patient_df, num_meal):
    """
    Remove all days that have exactly the specified number of meals.

    Identifies days with exactly num_meal meals and removes all data from those days.
    This function is useful for filtering out days with atypical meal patterns.
    Essentially, it cleans the dataset by removing days that do not meet the expected meal frequency.

    Args:
        patient_df (pd.DataFrame): DataFrame with columns 'msg_type' and datetime index.
        num_meal (int): The specific number of meals in a day to identify and remove.

    Returns:
        pd.DataFrame: Processed DataFrame with days containing num_meal meals removed.

    TODO:
    - This function assumes that the 'msg_type' column contains 'ANNOUNCE_MEAL' for meal events.
    - We need to modify the function to handle more than just one num_meal value.
    - Currently, it only removes days with exactly num_meal meals. It could be extended to handle ranges or other conditions.
    """
    # Ensure a 'day' column based on the date from the datetime index
    patient_df = patient_df.copy()
    patient_df["day"] = patient_df.index.date

    # Filter to only ANNOUNCE_MEAL rows
    announce_meal_df = patient_df[patient_df["msg_type"] == "ANNOUNCE_MEAL"]

    # Count the number of meals per day
    meal_counts = announce_meal_df.groupby("day").size()

    # Identify days with the specified number of meals
    days_to_remove = meal_counts[meal_counts == num_meal].index

    # Remove rows corresponding to these days
    result_df = patient_df[~patient_df["day"].isin(days_to_remove)]

    return result_df.drop(columns=["day"])


def erase_meal_overlap_fn(patient_df, meal_length, min_carbs):
    """
    Process the DataFrame to handle meal overlaps and combine closely timed meals.

    For each meal announcement:
    1. Marks low carb meals (below min_carbs threshold)
    2. For regular meals, identifies any other meals within meal_length window
    3. Combines carbohydrates from overlapping meals into the first meal
    4. Clears the overlapping meals to avoid double-counting

    Args:
        patient_df (pd.DataFrame): DataFrame with columns 'msg_type', 'food_g', and datetime index.
        meal_length (pd.Timedelta): The duration to look ahead for detecting overlapping meals.
        min_carbs (int): Minimum amount of carbohydrates to consider a significant meal.

    Returns:
        pd.DataFrame: Processed DataFrame with meal overlaps resolved.
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


def keep_top_n_carb_meals(patient_df, n_top_carb_meals):
    """
    Keep only the top n carbohydrate meals per day in the DataFrame.

    For each day (defined by day_start_shift):
    1. Identifies meal announcements
    2. Ranks meals by carbohydrate content
    3. Keeps only the top n_top_carb_meals meals
    4. Resets other meals to zero carbs

    Args:
        patient_df (pd.DataFrame): DataFrame with columns 'msg_type', 'food_g',
                                 'day_start_shift', and datetime index.
        n_top_carb_meals (int): Number of top carbohydrate meals to keep per day.

    Returns:
        pd.DataFrame: Processed DataFrame with only top n carbohydrate meals preserved.

    Raises:
        KeyError: If 'day_start_shift' column is not found in the DataFrame
    """
    # Ensure 'day_start_shift' exists
    if "day_start_shift" not in patient_df.columns:
        raise KeyError(
            "'day_start_shift' column not found. Ensure day_start_index_change is True in dataset_creator."
        )

    # Filter the DataFrame to include only 'ANNOUNCE_MEAL' events
    announce_meal_df = patient_df[patient_df["msg_type"] == "ANNOUNCE_MEAL"].copy()

    if announce_meal_df.empty:
        print("No 'ANNOUNCE_MEAL' events to process for top N meals.")
        return patient_df

    # Group by the shifted day
    grouped = announce_meal_df.groupby("day_start_shift")

    # Identify top n meal indices per group
    top_meal_indices = grouped.apply(
        lambda x: x.nlargest(n_top_carb_meals, "food_g"), include_groups=False
    ).index.get_level_values(1)

    # Mask to identify meals to keep
    keep_mask = patient_df.index.isin(top_meal_indices) & (
        patient_df["msg_type"] == "ANNOUNCE_MEAL"
    )

    # Set 'food_g' and 'msg_type' for non-top meals to 0 and '0' respectively
    patient_df.loc[
        ~keep_mask & (patient_df["msg_type"] == "ANNOUNCE_MEAL"), ["food_g", "msg_type"]
    ] = [0, "0"]

    return patient_df


def erase_consecutive_nan_values(
    patient_df: pd.DataFrame, max_consecutive_nan_values_per_day: int
):
    """
    Handle consecutive NaN values in blood glucose data based on configurable threshold.

    This function processes each day separately to:
    1. Count the maximum number of consecutive NaN values in the 'bgl' column
    2. If max consecutive NaNs exceeds threshold, remove the entire day
    3. If max consecutive NaNs is within threshold, keep the day but remove individual NaN values

    Args:
        patient_df (pd.DataFrame): DataFrame with a 'bgl' column and datetime index.
        max_consecutive_nan_values_per_day (int): Maximum allowed consecutive NaN values per day.
            Days exceeding this threshold are completely removed.

    Returns:
        pd.DataFrame: Processed DataFrame with consecutive NaN values handled according to the rules.
    """
    # Create a copy to avoid modifying original
    df = patient_df.copy()

    # Add day column for grouping
    df["day"] = df.index.to_series().dt.date

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
