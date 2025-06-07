from src.data.awesome_cgm.base_loader import BaseAwesomeCGMLoader


class AleppoDataLoader(BaseAwesomeCGMLoader):
    @property
    def dataset_name(self):
        """Return the name of the dataset."""
        return "aleppo2017"
