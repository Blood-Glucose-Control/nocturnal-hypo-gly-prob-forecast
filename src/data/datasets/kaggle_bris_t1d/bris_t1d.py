import pandas as pd
from src.data.preprocessing.time_processing import (
    create_datetime_index,
)
from src.data.preprocessing.sampling import (
    ensure_regular_time_intervals,
)
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)

from src.data.preprocessing.time_processing import get_train_validation_split
from src.data.datasets.dataset_base import DatasetBase
import os

from src.data.datasets.kaggle_bris_t1d.data_cleaner import (
    clean_brist1d_train_data,
    clean_brist1d_test_data,
)
from collections import defaultdict


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
        # TODO: Raw train.csv is quite large. Need to download from kaggle
        # Set this to the cached path for now as it is already processed
        # self.default_path = os.path.join(
        #     os.path.dirname(__file__), f"{dataset_type}.csv"
        self.default_path = os.path.join(
            os.path.dirname(__file__), f"{dataset_type}.csv"
        )
        self.cached_path = os.path.join(
            os.path.dirname(__file__),
            f"{dataset_type}_cached{'.csv' if dataset_type == 'train' else ''}",
        )
        self.use_cached = use_cached
        self.dataset_type = dataset_type
        self.file_path = self.default_path
        if file_path is not None:
            self.file_path = file_path
        elif use_cached and os.path.exists(self.cached_path):
            self.file_path = self.cached_path

        # Preload data
        self.load_data()

    @property
    def dFataset_name(self):
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
            pd.DataFrame or dict[str, dict[str, pd.DataFrame]]: The loaded data. A pandas DataFrame for train data, a dict of dict of DataFrames for test data.
        """
        if self.use_cached and os.path.exists(self.cached_path):
            if self.dataset_type == "train":
                self.processed_data = pd.read_csv(
                    self.cached_path, usecols=self.keep_columns
                )
            elif self.dataset_type == "test":
                self.processed_data = defaultdict(dict)

                if not os.path.isdir(self.cached_path):
                    raise NotADirectoryError(
                        f"Cache path '{self.cached_path}' is not a directory."
                    )

                for pid in os.listdir(self.cached_path):
                    patient_dir = os.path.join(self.cached_path, pid)
                    if not os.path.isdir(patient_dir):
                        raise NotADirectoryError(
                            f"Expected a directory for patient '{pid}', but got something else: {patient_dir}"
                        )

                    for filename in os.listdir(patient_dir):
                        if filename.endswith(".csv"):
                            file_path = os.path.join(patient_dir, filename)
                            df = pd.read_csv(file_path)

                            row_id = filename.replace(".csv", "")
                            self.processed_data[pid][row_id] = df
        else:
            # Not using cache or cache does not exist, process raw data and save
            self.raw_data = self.load_raw()
            self.processed_data = self._process_raw_data()

        if self.dataset_type == "train":
            # Split data into train and validation
            self.train_data, self.validation_data = get_train_validation_split(
                self.processed_data, num_validation_days=self.num_validation_days
            )

    def _process_raw_data(self) -> pd.DataFrame | dict[str, dict[str, pd.DataFrame]]:
        # Not cached, process the raw data
        if self.dataset_type == "train":
            data = clean_brist1d_train_data(self.raw_data)
            data = create_datetime_index(data)
            data = ensure_regular_time_intervals(data)
            data = create_cob_and_carb_availability_cols(data)
            data = create_iob_and_ins_availability_cols(data)

            # Save processed data to cache
            data.to_csv(self.cached_path, index=True)

            return data
        elif self.dataset_type == "test":  # test data: {pid : {rowid: df }}
            data = clean_brist1d_test_data(self.raw_data)
            processed_data = defaultdict(dict)

            # Setup cache dir
            os.makedirs(self.cached_path, exist_ok=True)

            for pid in data:
                # Make new dir for each patient
                patient_cache_dir = os.path.join(self.cached_path, pid)
                os.makedirs(patient_cache_dir, exist_ok=True)

                for row_id in data[pid]:
                    row_df = data[pid][row_id]
                    row_df = create_datetime_index(row_df)
                    row_df = ensure_regular_time_intervals(row_df)
                    row_df = create_cob_and_carb_availability_cols(row_df)
                    row_df = create_iob_and_ins_availability_cols(row_df)

                    cache_file = os.path.join(patient_cache_dir, f"{row_id}.csv")
                    row_df.to_csv(cache_file, index=True)

                    processed_data[pid][row_id] = row_df

            return processed_data

    # TODO: MOVE THIS TO THE splitter.py
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
