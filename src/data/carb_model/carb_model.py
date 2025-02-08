import numpy as np
import matplotlib.pyplot as plt

# https://ieeexplore.ieee.org/ielx7/4664312/10398544/10313965/supp1-3331297.pdf?arnumber=10313965
# This function was ported to Python from a Matlab function provided by PJacobs on (08/10/2023)
# CMosquera verified/modified the code and verified correctness of outputs
def calculate_carb_availability_and_cob_single_meal(meal_carbs, carb_absorption, ts_min, t_action_max_min):
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
            dq1[tt] = -(q1[tt] / tmax) + (carb_absorption * meal_carbs / ts_min)
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
