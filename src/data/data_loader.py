import pandas as pd
import os

def load_data(data_source_name: str = 'kaggle_brisT1D', dataset_type: str = 'train', keep_columns: list = None) -> pd.DataFrame:
    """
    Load data from a specified dataset and type, optionally selecting specific columns.

    Parameters:
    data_source_name (str): The name of the data source. Default is 'kaggle_brisT1D'.
    dataset_type (str): The type of the dataset, e.g., 'train' or 'test'. Default is 'train'.
    columns_to_keep (list): A list of column names to keep. If None, all columns are loaded. Default is None.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
    ValueError: If the specified file does not exist.
    """
    local_path = os.path.dirname(__file__)
    if data_source_name == 'kaggle_brisT1D':
        file_path = os.path.join(local_path, f'kaggle_brisT1D/{dataset_type}.csv')

    if not os.path.exists(file_path):
        raise ValueError("Invalid dataset_name or dataset_type")

    return pd.read_csv(file_path, usecols=keep_columns)
