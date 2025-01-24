import os
import pandas as pd

from data_loader import load_data

# Simulate insulin on board
# Convert each row to a single df
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    
    


    keep_columns = [
        "id",
        "p_num",
        "time",
        "bg",
        "insulin",
        "carbs",
        "hr",
        "steps",
        "cals",
        "activity",
    ]







import pandas as pd

def melt_data(data: pd.DataFrame, id_vars: list, value_vars: list) -> pd.DataFrame:
    """
    Transforms column data into row data using pandas melt.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    id_vars (list): Column(s) to use as identifier variables.
    value_vars (list): Column(s) to unpivot.

    Returns:
    pd.DataFrame: The melted DataFrame.
    """
    melted_data = pd.melt(data, id_vars=id_vars, value_vars=value_vars)
    return melted_data

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'p_num': [101, 102, 103],
        'bg': [120, 130, 140],
        'insulin': [10, 20, 30],
        'carbs': [50, 60, 70]
    })

    id_vars = ['id', 'p_num']
    value_vars = ['bg', 'insulin', 'carbs']

    melted_data = melt_data(data, id_vars, value_vars)
    print(melted_data)