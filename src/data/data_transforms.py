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
from src.data.wavelet_transformer import WaveletTransformer
from src.data.insulin_model.constants import IOB_COL, INSULIN_AVAIL_COL
from src.data.insulin_model.insulin_model import (
    calculate_insulin_availability_and_iob_single_delivery,
)


def create_datetime_index(
    df: pd.DataFrame, start_date: str = "2025-01-01"
) -> pd.DataFrame:
    """
    Creates a datetime index for the dataframe.
    """
    if "time_diff" not in df.columns:
        df = _create_time_diff_cols(df)

    # Create a str datetime column
    df["datetime"] = pd.to_datetime(
        start_date + " " + df["time"], format="%Y-%m-%d %H:%M:%S"
    )

    # Convert time_diff to a timedelta object
    # time_diff is the difference in time between the current row and the next row
    df["time_diff"] = pd.to_timedelta(df["time_diff"])

    # Create a cumulative sum of the absolute time differences
    df["cumulative_diff"] = (
        df.groupby("p_num")["time_diff"].cumsum().fillna(pd.Timedelta(0))
    )

    # Create a fixed start timestamp for each patient
    start_times = df.groupby("p_num")["datetime"].transform("first")

    # Add cumulative time difference to the fixed start timestamp
    df["datetime"] = start_times + df["cumulative_diff"]

    # Drop the intermediate columns
    df.drop(columns=["time_diff", "cumulative_diff"], inplace=True)

    return df


def _create_time_diff_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'time_diff' column to the DataFrame that represents the time difference between
    consecutive rows for each patient, grouped by patient. The time difference is in minutes.
    Also adds a 'datetime' column to the DataFrame.
    """
    df["temp_datetime"] = pd.to_datetime(df["time"], format="%H:%M:%S")

    # First create raw time differences
    df["time_diff_raw"] = df.groupby("p_num")["temp_datetime"].diff()

    # time_diff_raw is negative across day boundaries e.g. when the current row is at 23:59:59 and the next is at 00:00:00
    # Create a new column with the absolute value of time_diff
    # Have to assume that two entries are not separated by more than 24 hours
    # Since we were not given date, only time, so not considering days in this calculation
    df["time_diff"] = df["time_diff_raw"].apply(
        lambda x: (
            pd.Timedelta(
                hours=x.components.hours,
                minutes=x.components.minutes,
                seconds=x.components.seconds,
            )
            if pd.notna(x)
            else pd.Timedelta(0)
        )
    )

    # Clean up intermediate columns
    df.drop(columns=["temp_datetime", "time_diff_raw"], inplace=True)
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
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    result_df[COB_COL] = 0.0
    result_df[CARB_AVAIL_COL] = 0.0

    if "datetime" not in result_df.columns:
        result_df = create_datetime_index(result_df)

    # Process each patient separately
    for _, patient_df in result_df.groupby("p_num"):
        # Calculate for this patient
        for meal_time in patient_df.index[patient_df["carbs-0:00"].notna()]:
            meal_value = patient_df.loc[meal_time, "carbs-0:00"]
            meal_avail, cob = calculate_carb_availability_and_cob_single_meal(
                meal_value, CARB_ABSORPTION, TS_MIN, T_ACTION_MAX_MIN
            )

            # Add values for the current time
            result_df.loc[meal_time, CARB_AVAIL_COL] += meal_avail[0]
            result_df.loc[meal_time, COB_COL] += cob[0]

            # Continue with future times
            next_index = meal_time + 1
            time_since_meal = 0

            while next_index in patient_df.index and time_since_meal < T_ACTION_MAX_MIN:
                # Ensure datetime columns are in datetime format before subtraction
                dt_next = pd.to_datetime(patient_df.loc[next_index, "datetime"])
                dt_meal = pd.to_datetime(patient_df.loc[meal_time, "datetime"])
                time_since_meal = int((dt_next - dt_meal).total_seconds() / 60)

                if time_since_meal < T_ACTION_MAX_MIN:
                    result_df.loc[next_index, CARB_AVAIL_COL] += meal_avail[
                        time_since_meal
                    ]
                    result_df.loc[next_index, COB_COL] += cob[time_since_meal]

                next_index += 1

    return result_df


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
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    result_df[INSULIN_AVAIL_COL] = 0.0
    result_df[IOB_COL] = 0.0

    if "datetime" not in result_df.columns:
        result_df = create_datetime_index(result_df)

    # Process each patient separately
    for _, patient_df in result_df.groupby("p_num"):
        for ins_time in patient_df.index[
            (patient_df["insulin-0:00"].notna()) & (patient_df["insulin-0:00"] > 0)
        ]:
            insulin_dose = patient_df.loc[ins_time, "insulin-0:00"]

            # Simulate insulin dynamics
            ins_avail, iob, _, _, _ = (
                calculate_insulin_availability_and_iob_single_delivery(
                    insulin_dose, TS_MIN, T_ACTION_MAX_MIN
                )
            )

            # Add values for the current time
            result_df.loc[ins_time, INSULIN_AVAIL_COL] += ins_avail[0]
            result_df.loc[ins_time, IOB_COL] += iob[0]

            # Continue with future times
            next_index = ins_time + 1
            time_since_insulin = 0

            while (
                next_index in patient_df.index and time_since_insulin < T_ACTION_MAX_MIN
            ):
                # Calculate time difference in minutes directly from datetime
                time_since_insulin = int(
                    (
                        patient_df.loc[next_index, "datetime"]
                        - patient_df.loc[ins_time, "datetime"]
                    ).total_seconds()
                    / 60
                )

                if time_since_insulin < T_ACTION_MAX_MIN:
                    result_df.loc[next_index, INSULIN_AVAIL_COL] += ins_avail[
                        time_since_insulin
                    ]
                    result_df.loc[next_index, IOB_COL] += iob[time_since_insulin]

                next_index += 1

    return result_df


def get_most_common_time_interval(df: pd.DataFrame) -> int:
    df_copy = df.copy()
    df_copy["datetime"] = pd.to_datetime(df_copy["datetime"])
    df_copy["time_diff"] = (df_copy["datetime"] - df_copy["datetime"].shift(1)).apply(
        lambda x: x.components.minutes if pd.notnull(x) else None
    )

    return df_copy["time_diff"].value_counts().index[0]


def ensure_regular_time_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures regular time intervals exist in the dataframe by adding rows with NaN values
    where timestamps are missing.

    Args:
        df (pd.DataFrame): Input dataframe with datetime column

    Returns:
        pd.DataFrame: DataFrame with regular time intervals, missing times filled with NaN
    """
    result_df = df.copy()

    # Process each patient separately
    reindexed_dfs = []
    for patient_id, patient_df in result_df.groupby("p_num"):
        freq = get_most_common_time_interval(patient_df)

        # Create complete time range for this patient
        full_time_range = pd.date_range(
            start=patient_df["datetime"].min(),
            end=patient_df["datetime"].max(),
            freq=f"{freq}min",
        )

        # Create a DataFrame with the complete time range and ensure datetime type
        full_df = pd.DataFrame({"datetime": full_time_range})
        full_df["datetime"] = pd.to_datetime(full_df["datetime"])

        # Ensure patient_df datetime is also datetime type
        patient_df["datetime"] = pd.to_datetime(patient_df["datetime"])

        # Merge with original data
        reindexed_df = pd.merge(full_df, patient_df, on="datetime", how="left")

        # Restore patient ID for all rows
        reindexed_df["p_num"] = patient_id

        # Generate new sequential IDs for all rows
        sequence_numbers = range(len(reindexed_df))
        reindexed_df["id"] = [f"{patient_id}_{i}" for i in sequence_numbers]

        reindexed_dfs.append(reindexed_df)

    # Combine all reindexed patient data
    result_df = pd.concat(reindexed_dfs)

    return result_df


def apply_wavelet_transform(
    df: pd.DataFrame,
    wavelet="sym16",
    wavelet_window=36,
    patient_identifying_col="p_num",
    bgl_column="bg-0:00",
    level=3,
) -> pd.DataFrame:
    """
    Applies wavelet transform on each patient's data
    Separately uses the WaveletTransformer per patient

    Args:
        df: the input dataframe containing the cumulative data
        wavelet: the wavelet to use. See WaveletTransformer for more info
        wavelet_window: window size for the transform. Also see WaveletTransformer
        patient_identifying_col: the column name that distinguishes each patient's data
        bgl_column: the column to apply this over
    """
    # list to collect the transformed data for each patient
    transformed_data = []

    # group by patient. separately process them
    for patient_id, patient_data in df.groupby(patient_identifying_col):
        bgl_series = patient_data[bgl_column]
        # Initialize the WaveletTransformer for the current patient
        transformer = WaveletTransformer(
            window_len=wavelet_window, wavelet=wavelet, num_levels=level
        )

        transformed_bgl = transformer.fit_transform(bgl_series)

        patient_data[bgl_column] = transformed_bgl

        # adds new col to idntify the patient
        patient_data[patient_identifying_col] = patient_id

        # Add the transformed data to the list
        transformed_data.append(patient_data)

    result_df = pd.concat(transformed_data, axis=0)

    # Sort the result DataFrame by the original index to preserve the row order
    result_df = result_df.sort_index()

    return result_df


def take_moving_average(
    df: pd.DataFrame, window_size: int = 36, bg_col="bg-0:00"
) -> pd.DataFrame:
    """
    Takes the moving average of the dataframe over bg_col

    Args:
        df: the dataframe
        window_size: the moving average window size
        bg_col: the column take moving average over
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")
    if bg_col not in df.columns:
        raise ValueError(f"Column '{bg_col}' not found in the DataFrame")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Apply moving average
    result_df[bg_col] = (
        result_df[bg_col].rolling(window=window_size, min_periods=1).mean()
    )
    return result_df
