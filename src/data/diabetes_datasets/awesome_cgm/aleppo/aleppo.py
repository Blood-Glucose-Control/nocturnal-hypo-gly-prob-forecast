from src.data.diabetes_datasets.awesome_cgm.aleppo.preprocess import create_aleppo_csv
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.cache_manager import get_cache_manager

# from src.data.data_models import Dataset
from src.data.dataset_configs import get_dataset_config
from .data_cleaner import PreprocessConfig, clean_all_patients, default_config
from src.data.preprocessing.time_processing import get_train_validation_split
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class AleppoDataLoader(DatasetBase):
    def __init__(
        self,
        keep_columns: list = None,
        num_validation_days: int = 20,
        config: PreprocessConfig = default_config,
        use_cached: bool = True,
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
        self.raw_data_path = None
        self.use_cached = use_cached
        self.load_data()

    @property
    def dataset_name(self):
        # return Dataset.ALEPPO.value
        return "aleppo"

    @property
    def description(self):
        return """
                The purpose of this study was to determine whether the use of continuous glucose monitoring (CGM) without blood glucose monitoring (BGM) measurements is as safe and effective as using CGM with BGM in adults (25-40) with type 1 diabetes.
                The total sample size was 225 participants. The Dexcom G4 was used to continuously monitor glucose levels for a span of 6 months.
           """

    def load_raw(self):
        # This should guarantee the raw data exist or throw the error if it does not
        self.raw_data_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )
        logger.info("Raw data is not in csv format, please load processed data instead")

        need_to_process_data = True
        if self.use_cached:
            cached_data = self.cache_manager.load_processed_data(
                self.dataset_name, "train", file_format="csv"
            )
            if cached_data is not None:
                self.processed_data = cached_data
                need_to_process_data = False

        if need_to_process_data:
            # we should have a path here that guarantees the raw data exist to be processed
            self._process_and_cache_data()

        return None

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
        ## Every patient has different time span so number of days won't make sense here. Maybe just 10%
        self.train_data, self.validation_data = get_train_validation_split(
            self.processed_data, num_validation_days=self.num_validation_days
        )
        self.train_data = self.train_data.sort_values(by=["p_num", "datetime"])
        self.validation_data = self.validation_data.sort_values(
            by=["p_num", "datetime"]
        )

    def _process_and_cache_data(self):
        """Process the raw data and save it to the cache."""
        self.processed_data = self._process_raw_data()
        self.cache_manager.save_processed_data(
            self.dataset_name, "train", self.processed_data
        )

    def _process_raw_data(self) -> dict[str, pd.DataFrame]:
        """
        1.Transform the raw data from text to csv by patients (saved to interim folder)
        2.Do the processing on the csv files.
        """
        if self.raw_data_path is None:
            raise ValueError("Raw data path not loaded! Please call load_raw() first.")

        processed_path = self.cache_manager.get_processed_data_path(self.dataset_name)
        processed_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Create parent directory

        interim_path = (
            self.cache_manager.get_dataset_cache_path(self.dataset_name) / "interim"
        )

        # Raw -> interim ({pid}_full.csv)
        # TODO: Maybe we can even skip this if interim folder already exists
        create_aleppo_csv(self.raw_data_path)

        # interim -> processed ({pid}_full.csv)
        return clean_all_patients(interim_path, processed_path, self.config)
