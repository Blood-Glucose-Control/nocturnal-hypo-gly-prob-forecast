import pandas as pd


# Simulate insulin on board
# Convert each row to a single df
def clean_data(
        data: pd.DataFrame,
        data_source_name="kaggle_brisT1D"
    ) -> pd.DataFrame:
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

    if (data_source_name == "kaggle_brisT1D"):
        # modifies in place
        _clean_bris_data(data)
        return

    pass

def _clean_bris_data(data: pd.DataFrame):
    '''
    Cleans the bris1TD Kaggle data with the following transformations:
        1. Deletes columns of historic data (eg: bg-5:55, ..., activity-5:55, ...) --> but does not remove -0:00 timestamp
    Args:
        data: the df for the Bris1TD dataset
    Mutations:
        Modifies the data in place
    '''
    # get columns with bg- or activity-
    columns_to_drop = [col for col in data.columns if (('activity' in col or 'bg' in col or 'cals' in col or 'insulin' in col) and '-' in col and not col.endswith('-0:00'))]
    data.drop(columns=columns_to_drop, inplace=True)

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
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "p_num": [101, 102, 103],
            "bg": [120, 130, 140],
            "insulin": [10, 20, 30],
            "carbs": [50, 60, 70],
        }
    )

    id_vars = ["id", "p_num"]
    value_vars = ["bg", "insulin", "carbs"]

    melted_data = melt_data(data, id_vars, value_vars)
    print(melted_data)
