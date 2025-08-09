from src.data.datasets.dataset_base import DatasetBase
from .data_cleaner import PreprocessConfig, default_config, clean_cgm_data
from src.data.preprocessing.time_processing import get_train_validation_split
from .preprocess import create_aleppo_csv
import os
import pandas as pd


def get_storage_location():
    cache_dir = os.path.expanduser("~/.cache/nocturnal")
    os.makedirs(cache_dir, exist_ok=True)

    return os.path.join(cache_dir, "aleppo2017.csv")


class AleppoDataLoader(DatasetBase):
    def __init__(
        self,
        keep_columns: list = None,
        num_validation_days: int = 20,
        config: PreprocessConfig = default_config,
        raw_folder_path: str | None = None,
    ):
        """
        Args:
            keep_columns (list): List of columns to keep from the raw data.
            num_validation_days (int): Number of days to use for validation.
            csv_file_path (str): Path to the CSV file containing the raw data.
            config (dict): Configuration dictionary for data cleaning. passed to your cleaning function
        """
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.file_path = get_storage_location()
        self.config = config  # config for data cleaning
        self.raw_folder_path = raw_folder_path

        # Preload data
        self.load_data()

    @property
    def dataset_name(self):
        """Return the name of the dataset."""
        return "aleppo2017"

    def load_raw(self):
        """Load the raw dataset.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.

        """

        if os.path.isfile(self.file_path):
            return pd.read_csv(self.file_path, low_memory=False)
        assert (
            self.raw_folder_path is not None
        ), "Seems like you haven't preprocessed the aleppo dataset before. Ensure you pass in the path to the `raw` folder directory to do this"
        create_aleppo_csv(self.raw_folder_path, self.file_path)
        return pd.read_csv(self.file_path, low_memory=False)

    def _make_processed_data(self):
        self.raw_data = self.load_raw()
        self.processed_data = self._process_raw_data()

    def load_data(self):
        """
        The function will load the raw data, process data and split it into train and validation.
        If the dataset is not cached, the function will process the raw data and save it to the cache.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        self._make_processed_data()
        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=self.num_validation_days
        )
        self.train_data = self.train_data.sort_values(by=["p_num", "datetime"])
        self.validation_data = self.validation_data.sort_values(
            by=["p_num", "datetime"]
        )

    def _process_raw_data(self) -> pd.DataFrame:
        assert self.raw_data is not None, "Raw data not loaded!"

        raw_df = self.raw_data[self.keep_columns].copy()

        return clean_cgm_data(raw_df, self.config)

    def get_validation_day_splits(self, patient_id: str):
        """
        Get day splits for validation data for a single patient.

        Yields:
            tuple: (patient_id, train_period, test_period)
        """
        patient_data = self.validation_data[self.validation_data["p_num"] == patient_id]
        for train_period, test_period in self._get_day_splits(patient_data):
            yield patient_id, train_period, test_period

    # TODO: MOVE THIS TO THE splitter.py
    def _get_day_splits(self, patient_data: pd.DataFrame):
        """
        Split each day's data into training period (6am-12am) and test period (12am-6am next day).

        Args:
            patient_data (pd.DataFrame): Data for a single patient

        Yields:
            tuple: (train_period, test_period) where:
                - train_period is the data from 6am to 12am of a day
                - test_period is the data from 12am to 6am of the next day
        """

        patient_data.loc[:, "datetime"] = pd.to_datetime(patient_data["datetime"])

        # Ensure data is sorted by datetime
        patient_data = patient_data.sort_values("datetime")

        # Group by date
        for date, day_data in patient_data.groupby(patient_data["datetime"].dt.date):
            # Get next day's early morning data (12am-6am)
            next_date = date + pd.Timedelta(days=1)
            next_day_data = patient_data[
                (patient_data["datetime"].dt.date == next_date)
                & (patient_data["datetime"].dt.hour < 6)
            ]

            # Get current day's data (6am-12am)
            current_day_data = day_data[day_data["datetime"].dt.hour >= 6]

            if len(next_day_data) > 0 and len(current_day_data) > 0:
                yield current_day_data, next_day_data
