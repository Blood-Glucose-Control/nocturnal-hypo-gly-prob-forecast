import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from src.data.physiological.insulin_model.constants import (
    TMAX,
    KE,
    IOB_COL,
    INSULIN_AVAIL_COL,
)
from src.data.physiological.carb_model.constants import (
    TS_MIN,
    T_ACTION_MAX_MIN,
)
from src.data.preprocessing.time_processing import create_datetime_index


# This function was ported to Python from a Matlab function provided by PJacobs on (08/10/2023)
# CMosquera verified/modified the code and verified correctness of outputs
# Uses a 2-compartment model to represent insulin on board
# From Hovorka 2004
# https://ieeexplore.ieee.org/ielx7/4664312/10398544/10313965/supp1-3331297.pdf?arnumber=10313965&tag=1
def calculate_insulin_availability_and_iob_single_delivery(
    insulin, ts_min, t_action_max_min
):
    """
    Simulates insulin absorption and availability using Hovorka's 3-compartment model (2004).

    Parameters:
        insulin (float): Amount of insulin injected (units).
        ts_min (int): Time step in minutes.
        t_action_max_min (int): Maximum duration of insulin action (minutes).

    Returns:
        ins_availability (numpy array): Insulin available in plasma over time.
        iob (numpy array): Insulin on board over time.
        s1 (numpy array): Subcutaneous insulin compartment 1 (q1).
        s2 (numpy array): Subcutaneous insulin compartment 2 (q2).
        I (numpy array): Plasma insulin concentration.
    """

    # Number of discrete time steps for the simulation
    result_array_size = t_action_max_min // ts_min

    # Initialize compartments
    q1 = np.zeros(result_array_size)  # Subcutaneous insulin (slow absorption)
    q2 = np.zeros(result_array_size)  # Insulin transitioning to plasma
    I_p = np.zeros(result_array_size)  # Plasma insulin concentration
    q4 = np.zeros(result_array_size)  # Insulin that has been eliminated

    # Initialize rate of change variables
    dq1 = np.zeros(result_array_size)
    dq2 = np.zeros(result_array_size)
    dI = np.zeros(result_array_size)
    dq4 = np.zeros(result_array_size)

    # Euler integration loop (iterating over time steps)
    for tt in range(result_array_size - 1):
        if tt == 0:
            # First time step: insulin is injected into q1
            dq1[tt] = -(q1[tt] / TMAX) + (insulin / ts_min)
        else:
            # Subsequent time steps: insulin decays from q1
            dq1[tt] = -(q1[tt] / TMAX)

        # Transfer insulin from q1 → q2
        dq2[tt] = (q1[tt] / TMAX) - (q2[tt] / TMAX)

        # Transfer insulin from q2 → plasma (I)
        dI[tt] = (q2[tt] / TMAX) - (KE * I_p[tt])

        # Track insulin that has been metabolized
        dq4[tt] = KE * I_p[tt]

        # Update compartments using Euler integration
        q1[tt + 1] = q1[tt] + dq1[tt] * ts_min
        q2[tt + 1] = q2[tt] + dq2[tt] * ts_min
        I_p[tt + 1] = I_p[tt] + dI[tt] * ts_min
        q4[tt + 1] = q4[tt] + dq4[tt] * ts_min

    # Compute insulin availability (plasma insulin)
    ins_availability = I_p

    # Compute Insulin on Board (IOB)
    iob = insulin - q4
    iob[iob < 0] = 0  # Ensure IOB never goes negative

    # Return all relevant states
    return ins_availability, iob, q1, q2, I_p


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

            # TODO: Verify that this should be T_ACTION_MAX_MIN and not TMAX
            while (
                next_index in patient_df.index and time_since_insulin < T_ACTION_MAX_MIN
            ):
                # Extract the dateime values safely
                dt_next = patient_df.at[next_index, "datetime"]
                dt_ins = patient_df.at[ins_time, "datetime"]

                # Convert to datetime if they aren't already
                if not isinstance(dt_next, pd.Timestamp):
                    dt_next = pd.to_datetime(dt_next)
                if not isinstance(dt_ins, pd.Timestamp):
                    dt_ins = pd.to_datetime(dt_ins)

                time_since_insulin = int((dt_next - dt_ins).total_seconds() / 60)

                if time_since_insulin < T_ACTION_MAX_MIN:
                    result_df.loc[next_index, INSULIN_AVAIL_COL] += ins_avail[
                        time_since_insulin
                    ]
                    result_df.loc[next_index, IOB_COL] += iob[time_since_insulin]

                next_index += 1

    return result_df


def calculate_insulin_availability_and_iob_single_delivery_new(
    insulin, ts_min, t_action_max_min
):
    """
    Simulates insulin absorption and availability using Hovorka's 3-compartment model (2004).
    Using SciPy's ODE solver for improved accuracy and performance.
    TODO: Verify that this is the correct model to use, that it matches the original implementation, and is more performant.
    TODO: See if there are ways of improving the performance of this function further.
        1. JIT compilation with Numba
        2. Vectorization of the ODE system
        3. Parallel processing for multiple insulin doses/days
        4. Caching results for repeated doses

    Parameters:
        insulin (float): Amount of insulin injected (units).
        ts_min (int): Time step in minutes.
        t_action_max_min (int): Maximum duration of insulin action (minutes).

    Returns:
        ins_availability (numpy array): Insulin available in plasma over time.
        iob (numpy array): Insulin on board over time.
        s1 (numpy array): Subcutaneous insulin compartment 1 (q1).
        s2 (numpy array): Subcutaneous insulin compartment 2 (q2).
        I (numpy array): Plasma insulin concentration.
    """

    # Define the differential equation system
    def insulin_dynamics(t, y):
        q1, q2, I_p, q4 = y

        # At t=0, add the insulin injection
        u = insulin if t < ts_min else 0

        dq1 = -(q1 / TMAX) + (u / ts_min if t < ts_min else 0)
        dq2 = (q1 / TMAX) - (q2 / TMAX)
        dI = (q2 / TMAX) - (KE * I_p)
        dq4 = KE * I_p

        return [dq1, dq2, dI, dq4]

    # Initial conditions
    y0 = [0, 0, 0, 0]  # [q1, q2, I_p, q4]

    # Time points to solve for
    t_span = (0, t_action_max_min)
    t_eval = np.linspace(0, t_action_max_min, t_action_max_min // ts_min)

    # Solve the ODE system
    solution = solve_ivp(
        insulin_dynamics,
        t_span,
        y0,
        method="RK45",  # 4th-order Runge-Kutta method
        t_eval=t_eval,
        rtol=1e-4,  # Relative tolerance
        atol=1e-6,  # Absolute tolerance
    )

    # Extract solutions
    q1 = solution.y[0]
    q2 = solution.y[1]
    I_p = solution.y[2]
    q4 = solution.y[3]

    # Compute insulin availability (plasma insulin)
    ins_availability = I_p

    # Compute Insulin on Board (IOB)
    iob = insulin - q4
    iob[iob < 0] = 0  # Ensure IOB never goes negative

    return ins_availability, iob, q1, q2, I_p
