import pandas as pd
from sktime.split import temporal_train_test_split
from sktime.transformations.series.impute import Imputer
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
    
    data = handle_missing_values(data)

    return data

def _clean_bris_data(data: pd.DataFrame):
    '''
    Cleans the bris1TD Kaggle data with the following transformations:
        1. Deletes columns of historic data (eg: bg-5:55, ..., activity-5:55, ...) --> but does not remove -0:00 timestamp
        2. Deletes activity-0:00
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
    columns_to_drop.append('activity-0:00')
    data.drop(columns=columns_to_drop, inplace=True)

def handle_missing_values(data: pd.DataFrame, strategy="mean") -> pd.DataFrame:
    """
    Handles missing values in the DataFrame using sktime's Imputer.

    Args:
        data: Input DataFrame with possible missing values.
        strategy: Imputation strategy ('mean', 'median', 'constant', etc.).
    Returns:
        DataFrame with missing values handled.
    """
    # Create an Imputer instance
    imputer = Imputer(method=strategy)

    # Apply the imputer to all columns
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = imputer.fit_transform(data[col])

    return data

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
