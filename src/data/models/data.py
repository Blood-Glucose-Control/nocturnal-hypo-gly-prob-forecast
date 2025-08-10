from pydantic import BaseModel, model_validator
from enum import Enum
import pandas as pd


class Dataset(str, Enum):
    KAGGLE_BRIS_T1D = "kaggle_brisT1D"
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


class DatasetStructure(str, Enum):
    """
    HIERARCHICAL: Dataset is stored in a nested directory structure where:
    - The top level contains patient ID directories
    - Each patient directory contains CSV files named by row_id

    SINGLE_FILE: Dataset is stored in a single CSV file (eg: train.csv / test.csv)
    """

    HIERARCHICAL = "hierarchical"
    SINGLE_FILE = "single_file"


class DataEnvelope(BaseModel):
    dataset_structure: DatasetStructure
    data: pd.DataFrame | dict

    @model_validator(mode="after")
    def check_valid_data(self) -> "DataEnvelope":
        if self.dataset_structure == DatasetStructure.HIERARCHICAL:
            assert isinstance(self.data, dict), (
                "For hierarchical structure, data must be a dictionary of the following format:"
                "{{patient_id: {{row_id: DataFrame, row_id2: DataFrame}}, patient_id2: ...}}"
            )
        if self.dataset_structure == DatasetStructure.SINGLE_FILE:
            assert isinstance(
                self.data, pd.DataFrame
            ), "For single file structure, data must be a DataFrame"
        return self


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
