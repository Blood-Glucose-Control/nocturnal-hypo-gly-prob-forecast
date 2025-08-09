from pydantic import BaseModel, create_model
from enum import Enum


class Dataset(str, Enum):
    BRIS_T1D = "bris_t1d"
    ALEPPO = "aleppo"
    ANDERSON = "anderson"
    KAGGLE_BRIS_T1D = "kaggle_bris_t1d"


class DataSource(str, Enum):
    KAGGLE = "kaggle"
    LOCAL = "local"
    HUGGING_FACE = "hugging_face"


class DatasetType(str, Enum):
    TRAIN = "train"
    TEST = "test"


class _DatasetConfigBase(BaseModel):
    source: DataSource
    description: str
    citation: str
    source_path: str = None
    required_files: list = []


def dataset_config_factory(model_name: str, **custom_fields):
    """Create a pydantic model, inherited from _DatasetConfigBase,
    for your own dataset"""
    return create_model(
        model_name,
        __base__=_DatasetConfigBase,
        **custom_fields,
    )


KaggleDatasetConfig = dataset_config_factory(
    "KaggleBrisT1DConfig",
    competition_name=str,  # contest name
    url=str,  # dataset / contest url
)

GlurooDatasetConfig = dataset_config_factory(
    "GlurooDatasetConfig",
)

SimglucoseConfig = dataset_config_factory("SimglucoseConfig", url=str)
