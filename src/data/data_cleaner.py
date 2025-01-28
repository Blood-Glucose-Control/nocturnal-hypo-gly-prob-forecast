import pandas as pd
from sktime.split import temporal_train_test_split

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
    prefixes_to_check = ['activity', 'bg', 'cals', 'insulin', 'steps', 'carbs', 'hr']

    # Create the list of columns to drop
    columns_to_drop = [
        col for col in data.columns
        if any(prefix in col for prefix in prefixes_to_check) and '-' in col and not col.endswith('-0:00')
    ]
    data.drop(columns=columns_to_drop, inplace=True)

def perform_train_test_split(df: pd.DataFrame, target_col = 'bg-0:00', test_size=0.2):
    '''
    Splits the data into training and testing sets
    Args:
        df: the dataframe to split
        target_col: the column that you are trying to predict (i.e the "y" column)
        test_size: the size of the test data (0.0 - 1.0)
    Returns:
        y_train, y_test, X_train, X_test
    '''
    y = df[target_col]
    x = df.drop(columns=[target_col])
    return temporal_train_test_split(y, x, test_size=test_size)

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
