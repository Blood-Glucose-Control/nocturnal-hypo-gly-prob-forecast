import pandas as pd
from src.data.data_cleaner import clean_data
from src.data.data_transforms import (
    create_datetime_index,
    create_cob_and_carb_availability_cols,
    create_iob_and_ins_availability_cols,
    ensure_regular_time_intervals,
)
from src.data.data_spillter import get_train_validation_split
from src.data.DatasetBase import DatasetBase
import os


class BrisT1DDataLoader(DatasetBase):
    def __init__(
        self,
        keep_columns: list = None,
        num_validation_days: int = 20,
        file_path: str = None,
        use_cached: bool = True,
        dataset_type: str = "train",
    ):
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.default_path = os.path.join(
            os.path.dirname(__file__), f"{dataset_type}.csv"
        )
        self.cached_path = os.path.join(os.path.dirname(__file__), "train_cached.csv")
        self.file_path = file_path if file_path is not None else self.default_path
        self.use_cached = use_cached
        self.dataset_type = dataset_type
        # Preload data
        self.load_data()

    @property
    def dataset_name(self):
        """Return the name of the dataset."""
        return "kaggle_brisT1D"

    def load_raw(self):
        """Load the raw dataset.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.

        """
        # Return all columns
        return pd.read_csv(self.file_path, low_memory=False)

    def load_data(self):
        """
        Kaggle dataset is not big and can have multiple patients in the same file. We can cache the dataset
        and load it from the cache.
        The function will load the raw data, process data and split it into train and validation.
        If the dataset is not cached, the function will process the raw data and save it to the cache.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        """
        self.raw_data = self.load_raw()
        if self.dataset_type == "train":
            if self.use_cached:
                # Check if cache exists, if not process and save
                if not os.path.exists(self.cached_path):
                    self.processed_data = self._process_raw_data()
                    # Save processed data to cache
                    self.processed_data.to_csv(self.cached_path, index=True)
                else:
                    self.processed_data = pd.read_csv(
                        self.cached_path, usecols=self.keep_columns
                    )
            else:
                # Not using cache, process and save
                self.processed_data = self._process_raw_data()
                self.processed_data.to_csv(self.cached_path, index=True)

        # Split data into train and validation
        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=self.num_validation_days
        )

    def _process_raw_data(self) -> pd.DataFrame:
        # Not cached, process the raw data
        data = clean_data(self.raw_data)
        data = create_datetime_index(data)
        data = ensure_regular_time_intervals(data)
        data = create_cob_and_carb_availability_cols(data)
        data = create_iob_and_ins_availability_cols(data)
        return data

    # TODO: MOVE THIS TO THE spillter.py
    def get_validation_day_splits(self, patient_id: str):
        """
        Get day splits for validation data for a single patient.

        Yields:
            tuple: (patient_id, train_period, test_period)
        """
        patient_data = self.validation_data[self.validation_data["p_num"] == patient_id]
        for train_period, test_period in self._get_day_splits(patient_data):
            yield patient_id, train_period, test_period

    # TODO: MOVE THIS TO THE spillter.py
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
