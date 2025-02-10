import numpy as np
from src.data.insulin_model.constants import TMAX, KE


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
