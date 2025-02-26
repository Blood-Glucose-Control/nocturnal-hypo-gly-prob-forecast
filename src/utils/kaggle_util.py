"""Utility functions for the BrisT1D dataset on Kaggle."""


def create_time_variable_lists():
    """Creates lists of time variables for each measurement type (bg, insulin, etc)

    Returns:
        list: List of lists containing time variables for each measurement type
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
