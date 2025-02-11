import pandas as pd
import os


def load_data(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list = None,
    use_cached: bool = False,
) -> pd.DataFrame:
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
    if data_source_name == "kaggle_brisT1D":
        default_path = os.path.join(local_path, f"kaggle_brisT1D/{dataset_type}.csv")
        cached_path = os.path.join(local_path, "kaggle_brisT1D/train_cached.csv")
        file_path = (
            cached_path
            if use_cached and dataset_type == "train" and os.path.exists(cached_path)
            else default_path
        )
        if use_cached and dataset_type == "train" and not os.path.exists(cached_path):
            raise ValueError(
                f"Unable to find train_cached.csv at "
                f"\n {cached_path}. \n"
                f"Verify that the file exists if you want to use the cached version."
            )

    if not os.path.exists(file_path):
        raise ValueError("Invalid dataset_name or dataset_type")

    return pd.read_csv(file_path, usecols=keep_columns, low_memory=False)
