from src.data.datasets.base_cgm import BaseAwesomeCGMLoader
from src.data.datasets.aleppo.clean_data import clean_cgm_data


class AleppoDataLoader(BaseAwesomeCGMLoader):
    def __init__(self, *args, **kwargs):
        self.cleaner = clean_cgm_data
        super().__init__(*args, **kwargs)

    @property
    def dataset_name(self):
        """Return the name of the dataset."""
        return "aleppo2017"
