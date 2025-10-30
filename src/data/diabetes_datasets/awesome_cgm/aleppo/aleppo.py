from src.data.diabetes_datasets.awesome_cgm.aleppo.preprocess import create_aleppo_csv
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.cache_manager import get_cache_manager

# from src.data.data_models import Dataset
from src.data.dataset_configs import get_dataset_config
from .data_cleaner import PreprocessConfig, clean_all_patients, default_config
import pandas as pd
import logging


logger = logging.getLogger(__name__)


# TODO: ISF/CR is not dropped in the dataset. We could use this to calculate slope of the glucose curve.
# to give models some hints about trend of the glucose curve.
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

    def load_data(self):
        """
        The function will load the raw data, process data and split it into train and validation.
        If the dataset is not cached, the function will process the raw data and save it to the cache.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        need_to_process_data = True
        if self.use_cached:
            cached_data = self.cache_manager.load_processed_data(
                self.dataset_name, "train", file_format="csv"
            )
            if cached_data is not None:
                self.processed_data = cached_data
                need_to_process_data = False
        if need_to_process_data:
            self._process_and_cache_data()

        ## TODO: Every patient has different time span so number of days won't make sense here. Maybe just 10%
        self.train_data = self.processed_data
        # self.train_data, self.validation_data = get_train_validation_split(
        #     self.processed_data, num_validation_days=self.num_validation_days
        # )
        # self.train_data = self.train_data.sort_values(by=["p_num", "datetime"])
        # self.validation_data = self.validation_data.sort_values(
        #     by=["p_num", "datetime"]
        # )

    def load_raw(self):
        """
        Raw data of this dataset is not loadable (not in csv format). So we only check if the raw data exists.
        If not we throw an error and give instructions to the user on how to download the data and place it in the correct cache directory.
        """
        self.raw_data_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )

    def _process_and_cache_data(self):
        """
        We don't have the processed data cached so we need to load raw data then process it and save it to the cache.
        """
        # This will guarantee the raw data exists or throw an error if it does not.
        self.load_raw()
        self.processed_data = self._process_raw_data()
        self.cache_manager.save_processed_data(
            self.dataset_name, "train", self.processed_data
        )

    def _process_raw_data(self) -> dict[str, pd.DataFrame]:
        """
        1.Transform the raw data from text to csv by patients (saved to interim folder)
        2.Do the processing on the csv files.
        """

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
