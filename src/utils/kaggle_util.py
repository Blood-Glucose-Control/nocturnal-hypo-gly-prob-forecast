"""
Utility functions for working with the BrisT1D dataset from Kaggle competition.

This module contains helper functions that simplify working with the BrisT1D Kaggle
competition dataset for nocturnal hypoglycemia prediction in patients with Type 1 diabetes.
"""


def create_time_variable_lists():
    """
    Create lists of time-stamped variable names for each measurement type in the dataset.

    Generates variable names for blood glucose, insulin, carbs, heart rate, steps,
    calories and activity variables at 5-minute intervals spanning 6 hours (from t-5h to t-0h).
    Each measurement type has its own list of timestamped variable names.

    Returns:
        list: A list containing 7 sublists, each with variable names for a specific measurement
              type (bg, insulin, carbs, hr, steps, cals, activity) at 5-minute intervals.
              Format of variable names is "{measurement}-{hour}:{minute}" (e.g. "bg-5:30").

    Example:
        >>> lists = create_time_variable_lists()
        >>> print(lists[0][:3])  # First 3 blood glucose variable names
        ['bg-5:55', 'bg-5:50', 'bg-5:45']
    """

    all_value_var_lists = []
    var_strs = ["bg-", "insulin-", "carbs-", "hr-", "steps-", "cals-", "activity-"]

    for var in var_strs:
        var_str = var
        var_list = []
        for hour in range(5, -1, -1):
            time_hour = var_str + str(hour)
            for minutes in range(55, -1, -5):
                if minutes < 10:
                    time = time_hour + ":0" + str(minutes)
                else:
                    time = time_hour + ":" + str(minutes)
                var_list.append(time)
        all_value_var_lists.append(var_list)
    return all_value_var_lists
