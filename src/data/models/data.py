from pydantic import BaseModel, model_validator
from enum import Enum


class Dataset(str, Enum):
    KAGGLE_BRIS_T1D = "kaggle_bris_t1d"
    ALEPPO = "aleppo"
    ANDERSON = "anderson"
    GLUROO = "gluroo"
    SIMGLUCOSE = "simglucose"


class DataSource(str, Enum):
    KAGGLE = "kaggle"
    LOCAL = "local"
    HUGGING_FACE = "hugging_face"


class DatasetType(str, Enum):
    TRAIN = "train"
    TEST = "test"


class DatasetConfig(BaseModel):
    source: DataSource
    description: str = "No description available"
    citation: str = "No citation available"
    required_files: list = []
    url: str = "No URL available"
    source_path: str | None = None
    # kaggle specific!
    competition_name: str | None = None
    # hugging face specific
    dataset_id: str | None = None

    @model_validator(mode="after")
    def check_huggingface_fields(self) -> "DatasetConfig":
        assert (
            self.source == DataSource.HUGGING_FACE and (self.dataset_id is not None)
        ), "HuggingFace dataset must have a dataset ID! Please include this in your `DatasetConfig`"
        return self

    @model_validator(mode="after")
    def check_kaggle_fields(self) -> "DatasetConfig":
        assert (
            self.source == DataSource.KAGGLE and (self.competition_name is not None)
        ), "Kaggle dataset must have a competition name! Please include this in your `DatasetConfig`"
        return self

    @model_validator(mode="after")
    def check_local_fields(self) -> "DatasetConfig":
        assert (
            self.source == DataSource.LOCAL and (self.source_path is not None)
        ), "Local dataset must have a source path! Please include this in your `DatasetConfig`"
        return self
