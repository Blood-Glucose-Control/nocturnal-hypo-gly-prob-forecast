from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.cache_manager import get_cache_manager
from src.data.models.data import Dataset
from src.data.dataset_configs import get_dataset_config
from .data_cleaner import PreprocessConfig, default_config, clean_cgm_data
from src.data.preprocessing.time_processing import get_train_validation_split
import pandas as pd


class AleppoDataLoader(DatasetBase):
    def __init__(
        self,
        keep_columns: list = None,
        num_validation_days: int = 20,
        config: PreprocessConfig = default_config,
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
        self.cache_manager = get_cache_manager()
        self.config = config
        self.dataset_config = get_dataset_config(self.dataset_name)

        self.load_data()

    @property
    def dataset_name(self):
        return Dataset.ALEPPO.value

    def load_raw(self):
        """Load the raw dataset.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.

        """

        raw_data_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )

        return pd.read_csv(raw_data_path, low_memory=False)

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
