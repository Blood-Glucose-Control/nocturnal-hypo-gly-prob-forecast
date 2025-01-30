import pandas as pd


# Simulate insulin on board
# Convert each row to a single df
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    ''''''
    # keep_columns = [
    #     "id",
    #     "p_num",
    #     "time",
    #     "bg",
    #     "insulin",
    #     "carbs",
    #     "hr",
    #     "steps",
    #     "cals",
    #     "activity",
    # ]
    pass


def melt_data(df: pd.DataFrame, id_vars: list, value_vars: list) -> pd.DataFrame:
    """
    Transforms column data into row data using pandas melt.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    id_vars (list): Column(s) to use as identifier variables.
    value_vars (list): Column(s) to unpivot.

    Returns:
    pd.DataFrame: The melted DataFrame.
    """
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
    return melted_df


# Example usage
if __name__ == "__main__":
    # Sample data
    example_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "p_num": [101, 102, 103],
            "bg": [120, 130, 140],
            "insulin": [10, 20, 30],
            "carbs": [50, 60, 70],
        }
    )

    id_variables = ["id", "p_num"]
    value_variables = ["bg", "insulin", "carbs"]

    melted_data = melt_data(example_df, id_variables, value_variables)
    print(melted_data)
