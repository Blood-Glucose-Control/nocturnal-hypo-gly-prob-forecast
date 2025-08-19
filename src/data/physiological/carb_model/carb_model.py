import pandas as pd
import numpy as np

from src.data.physiological.carb_model.constants import (
    CARB_ABSORPTION,
    TS_MIN,
    T_ACTION_MAX_MIN,
    COB_COL,
    CARB_AVAIL_COL,
)


# https://ieeexplore.ieee.org/ielx7/4664312/10398544/10313965/supp1-3331297.pdf?arnumber=10313965
# This function was ported to Python from a Matlab function provided by PJacobs on (08/10/2023)
# CMosquera verified/modified the code and verified correctness of outputs
def calculate_carb_availability_and_cob_single_meal(
    meal_carbs, carb_absorption, ts_min, t_action_max_min
):
    """
    Calculate carbohydrate availability and meal on board (MOB) for a single meal event
    using a 2-compartment model.

    Parameters:
      meal_carbs      : Total grams of carbohydrates ingested.
      carb_absorption : Fraction of the meal that is absorbed (0-1), e.g., 0.8 means 80% absorbed.
      ts_min          : Time step in minutes for the simulation.
      t_action_max_min: Total simulation time in minutes (e.g., duration over which the meal has an effect).

    Returns:
      meal_availability : An array (over time) of the carbs in plasma (q2).
      mob               : An array (over time) of the meal on board (MOB), defined as the effective meal
                          (carb_absorption*meal_carbs) minus the cumulative absorption.
    """
    # Set the time constant for absorption (default from paper: 40 minutes)
    tmax = 40

    # Determine the number of discrete time steps
    result_array_size = t_action_max_min // ts_min

    # Initialize arrays for the compartments:
    # q1: Carbs in the gut (undigested)
    # q2: Carbs in plasma (absorbed and active)
    # q3: Cumulative carbs that have been “processed” (used for calculating MOB)
    q1 = np.zeros(result_array_size)
    q2 = np.zeros(result_array_size)
    q3 = np.zeros(result_array_size)

    # Arrays to store the derivative (rate of change) for each compartment
    dq1 = np.zeros(result_array_size)
    dq2 = np.zeros(result_array_size)
    dq3 = np.zeros(result_array_size)

    # Euler integration over the simulation period
    for tt in range(result_array_size - 1):
        if tt == 0:
            # At time zero, inject the meal into q1
            try:
                dq1[tt] = -(q1[tt] / tmax) + (carb_absorption * meal_carbs / ts_min)
            except:
                raise ValueError(f"Error in calculating dq1. tt: {tt}, Meal Carbs: {meal_carbs}, Carb Absorption: {carb_absorption}, Time Step: {ts_min}")
        else:
            # After the first time step, no additional input is added
            dq1[tt] = -(q1[tt] / tmax)

        # The plasma compartment (q2) receives input from q1 and loses content at the same rate
        dq2[tt] = (q1[tt] / tmax) - (q2[tt] / tmax)

        # q3 accumulates the output from q2 (this is used to compute the MOB)
        dq3[tt] = q2[tt] / tmax

        # Update each compartment using Euler's method: new_value = current_value + (derivative * dt)
        q1[tt + 1] = q1[tt] + dq1[tt] * ts_min
        q2[tt + 1] = q2[tt] + dq2[tt] * ts_min
        q3[tt + 1] = q3[tt] + dq3[tt] * ts_min

    # Meal availability is directly represented by the plasma compartment (q2)
    meal_availability = q2
    # Meal on Board (MOB) is the effective meal size minus the cumulative absorption
    mob = carb_absorption * meal_carbs - q3

    return meal_availability, mob

def create_cob_and_carb_availability_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the carbohydrate availability (CARB_AVAIL_COL) and carbohydrate on board (COB_COL)
    for each meal announcement in the dataframe.

    Assumes TS_MIN = 1 minute and that the DataFrame contains only one patient.

    Parameters:
    df (pd.DataFrame): DataFrame with datetime index containing meal announcement times.

    Returns:
    pd.DataFrame: Updated DataFrame with computed `carb_availability` and `cob` columns.
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    result_df[COB_COL] = 0.0
    result_df[CARB_AVAIL_COL] = 0.0

    if not isinstance(result_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    # Single patient processing only
    for meal_time in result_df.index[result_df["food_g"].notna()]:
        #print(f"\n Processing meal at {meal_time}")
        #print(f" Meal time type: {type(meal_time)}")
        #print(f" Meal time index: {result_df.loc[meal_time, 'id']}")
        meal_value = result_df.loc[meal_time, "food_g"]
        #print(f" Meal value: {meal_value} g")
        meal_avail, cob = calculate_carb_availability_and_cob_single_meal(
            meal_value, CARB_ABSORPTION, TS_MIN, T_ACTION_MAX_MIN
        )

        # Add values for the current time
        result_df.loc[meal_time, CARB_AVAIL_COL] += meal_avail[0]
        result_df.loc[meal_time, COB_COL] += cob[0]

        # Continue with future times
        next_index = meal_time + pd.Timedelta(minutes=1)
        time_since_meal_mins = 0

        while (
            next_index in result_df.index
            and time_since_meal_mins < T_ACTION_MAX_MIN
        ):
            # Calculate time difference using index
            time_since_meal_mins = int(
                (next_index - meal_time).total_seconds() / 60
            )

            if time_since_meal_mins < T_ACTION_MAX_MIN:
                result_df.loc[next_index, CARB_AVAIL_COL] += meal_avail[
                    time_since_meal_mins
                ]
                result_df.loc[next_index, COB_COL] += cob[time_since_meal_mins]

            next_index += pd.Timedelta(minutes=1)

    return result_df

def _deprecated_create_cob_and_carb_availability_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: This method assumes that TS_MIN = 1 minute.
    DEPRECATION REASION: WE NO LONGER WANT GENERIC FUNCTIONS HANDLING PARALLEL PROCESSING, DATA LOADERS HANDLE THAT.
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
    df (pd.DataFrame): DataFrame with datetime index containing meal announcement times.

    Returns:
    pd.DataFrame: Updated DataFrame with computed `carb_availability` and `cob` columns.
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    result_df[COB_COL] = 0.0
    result_df[CARB_AVAIL_COL] = 0.0

    if not isinstance(result_df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    n_patients = result_df["p_num"].nunique()

    if n_patients > 1:
        # Process each patient separately
        # TODO:TONY - This is a bottleneck. We should parallelize this.

        for _, patient_df in result_df.groupby("p_num"):
            # Calculate for this patient
            for meal_time in patient_df.index[patient_df["food_g"].notna()]:
                meal_value = patient_df.loc[meal_time, "food_g"]
                meal_avail, cob = calculate_carb_availability_and_cob_single_meal(
                    meal_value, CARB_ABSORPTION, TS_MIN, T_ACTION_MAX_MIN
                )

                # Add values for the current time
                result_df.loc[meal_time, CARB_AVAIL_COL] += meal_avail[0]
                result_df.loc[meal_time, COB_COL] += cob[0]

                # Continue with future times
                next_index = meal_time + pd.Timedelta(
                    minutes=1
                )  # Use timedelta instead of +1
                time_since_meal_mins = 0

                while (
                    next_index in patient_df.index
                    and time_since_meal_mins < T_ACTION_MAX_MIN
                ):
                    # Calculate time difference using index
                    time_since_meal_mins = int(
                        (next_index - meal_time).total_seconds() / 60
                    )

                    if time_since_meal_mins < T_ACTION_MAX_MIN:
                        result_df.loc[next_index, CARB_AVAIL_COL] += meal_avail[
                            time_since_meal_mins
                        ]
                        result_df.loc[next_index, COB_COL] += cob[time_since_meal_mins]

                    next_index += pd.Timedelta(minutes=1)  # Use timedelta
    else:
        # Single patient processing
        for meal_time in result_df.index[result_df["food_g"].notna()]:
            meal_value = result_df.loc[meal_time, "food_g"]
            meal_avail, cob = calculate_carb_availability_and_cob_single_meal(
                meal_value, CARB_ABSORPTION, TS_MIN, T_ACTION_MAX_MIN
            )

            # Add values for the current time
            result_df.loc[meal_time, CARB_AVAIL_COL] += meal_avail[0]
            result_df.loc[meal_time, COB_COL] += cob[0]

            # Continue with future times
            next_index = meal_time + pd.Timedelta(minutes=1)
            time_since_meal_mins = 0

            while (
                next_index in result_df.index
                and time_since_meal_mins < T_ACTION_MAX_MIN
            ):
                # Calculate time difference using index
                time_since_meal_mins = int(
                    (next_index - meal_time).total_seconds() / 60
                )

                if time_since_meal_mins < T_ACTION_MAX_MIN:
                    result_df.loc[next_index, CARB_AVAIL_COL] += meal_avail[
                        time_since_meal_mins
                    ]
                    result_df.loc[next_index, COB_COL] += cob[time_since_meal_mins]

                next_index += pd.Timedelta(minutes=1)

    return result_df
