"""Data loading functions for the various datasets."""

from src.data.kaggle_brisT1D.bristT1D import BrisT1DDataLoader
from src.data.gluroo.gluroo import Gluroo
from src.data.dataset_base import DatasetBase


def get_loader(
    data_source_name: str = "kaggle_brisT1D",
    dataset_type: str = "train",
    keep_columns: list = None,
    use_cached: bool = False,
    num_validation_days: int = 20,
    file_path: str = None,
    config: dict = None,
) -> DatasetBase:
    """
    Get the data loader for the given data source name.

    Parameters:
        data_source_name (str): The name of the data source. Default is 'kaggle_brisT1D'.
        dataset_type (str): The type of the dataset, e.g., 'train' or 'test'. Default is 'train'.
        keep_columns (list): A list of column names to keep. If None, all columns are loaded.
            Default is None.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if data_source_name == "kaggle_brisT1D":
        return BrisT1DDataLoader(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            file_path=file_path,
            use_cached=use_cached,
            dataset_type=dataset_type,
        )
    elif data_source_name == "gluroo":
        return Gluroo(
            keep_columns=keep_columns,
            num_validation_days=num_validation_days,
            file_path=file_path,
            config=config,
        )
    else:
        raise ValueError("Invalid dataset_name or dataset_type")
