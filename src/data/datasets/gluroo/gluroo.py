import pandas as pd
from src.data.datasets.dataset_base import DatasetBase
from src.data.datasets.gluroo.data_cleaner import clean_gluroo_data
from src.data.data_splitter import get_train_validation_split
from src.data.preprocessing.sampling import ensure_regular_time_intervals
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)


# TODO: Maybe need to return the test set too.
class Gluroo(DatasetBase):
    def __init__(
        self,
        keep_columns: list = [],
        num_validation_days: int = 20,
        file_path: str = "",
        config: dict = {},
        use_cached: bool = False,
    ):
        self.keep_columns = keep_columns
        self.num_validation_days = num_validation_days
        self.file_path = file_path  # Raw file path
        self.config = config
        self.use_cached = use_cached
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.validation_data = None
        self.load_data()

    @property
    def dataset_name(self):
        """Return the name of the dataset."""
        return "gluroo"

    def load_raw(self):
        """Load the raw dataset.

        Returns:
            pd.DataFrame: The raw data loaded from the CSV file.

        """
        if self.file_path is None:
            raise ValueError("File path is required")
        return pd.read_csv(self.file_path, usecols=self.keep_columns)

    def load_data(self):
        """Load and process the raw data, setting up train/validation splits."""
        if self.use_cached:
            cached_data = pd.read_csv("gluroo_cached.csv")
            self.processed_data = cached_data
        else:
            self.raw_data = self.load_raw()
            self.processed_data = self._process_raw_data()
            self.processed_data.to_csv("/gluroo_cached.csv", index=False, mode="w")

        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=self.num_validation_days
        )

    def _process_raw_data(self):
        """Process the raw data using the Gluroo-specific cleaning function.

        Returns:
            pd.DataFrame: The processed data.

        """
        if self.raw_data is None:
            raise ValueError("Raw data is required")
        raw = self.raw_data[self.keep_columns].copy()
        cleaned_df = clean_gluroo_data(raw, self.config)
        processed_df_regular = ensure_regular_time_intervals(cleaned_df)
        processed_df_cob = create_cob_and_carb_availability_cols(processed_df_regular)
        processed_df_iob = create_iob_and_ins_availability_cols(processed_df_cob)
        return processed_df_iob

    # Split the validation data into train and validation sets
    def get_validation_day_splits(self, patient_id: str):
        if self.validation_data is None:
            raise ValueError(
                "Validation data is not loaded. Please ensure data is loaded before calling this method."
            )
        patient_data = self.validation_data[self.validation_data["p_num"] == patient_id]
        for train_period, test_period in self._get_day_splits(patient_data):
            yield patient_id, train_period, test_period

    def _get_day_splits(self, patient_data: pd.DataFrame):
        """
        Split each day's data into training period (6am-12am) and test period (12am-6am next day).

        Args:
            patient_data (pd.DataFrame): Data for a single patient
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
