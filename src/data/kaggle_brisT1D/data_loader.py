import pandas as pd
from src.data.data_cleaner import clean_data
from src.data.data_transforms import (
    create_datetime_index,
    create_cob_and_carb_availability_cols,
    create_iob_and_ins_availability_cols,
    ensure_regular_time_intervals,
)
from src.data.data_loader import load_data, get_train_validation_split


class BrisT1DDataLoader:
    def __init__(
        self,
        keep_columns: list = None,
        use_cached: bool = True,
        num_validation_days: int = 20,
    ):
        self.keep_columns = keep_columns
        self.use_cached = use_cached
        self.raw_data = load_data(
            data_source_name="kaggle_brisT1D",
            dataset_type="train",
            keep_columns=keep_columns,
            use_cached=use_cached,
        )
        self.processed_data = self._process_raw_data()
        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=num_validation_days
        )

    def get_validation_day_splits(self, patient_id: str):
        """
        Get day splits for validation data for a single patient.

        Yields:
            tuple: (patient_id, train_period, test_period)
        """
        patient_data = self.validation_data[self.validation_data["p_num"] == patient_id]
        for train_period, test_period in self._get_day_splits(patient_data):
            yield patient_id, train_period, test_period

    def _process_raw_data(self) -> pd.DataFrame:
        if self.use_cached:
            self.processed_data = self.raw_data
            return self.processed_data
        # Not cached, process the raw data
        self.processed_data = clean_data(self.raw_data)
        self.processed_data = create_datetime_index(self.processed_data)
        self.processed_data = ensure_regular_time_intervals(self.processed_data)
        self.processed_data = create_cob_and_carb_availability_cols(self.processed_data)
        self.processed_data = create_iob_and_ins_availability_cols(self.processed_data)
        return self.processed_data

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
