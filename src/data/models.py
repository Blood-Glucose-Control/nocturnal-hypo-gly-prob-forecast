# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

from pydantic import BaseModel, ConfigDict, Field
from enum import Enum


class DatasetSourceType(str, Enum):
    KAGGLE_BRIS_T1D = "kaggle_brisT1D"
    LOCAL = "local"
    GLUROO = "gluroo"
    HUGGING_FACE = "hugging_face"
    ALEPPO_2017 = "aleppo_2017"
    LYNCH_2022 = "lynch_2022"
    BROWN_2019 = "brown_2019"
    SIMGLUCOSE = "simglucose"
    TAMBORLANE_2008 = "tamborlane_2008"


# TODO: Add to the mean_cols list in sampling.py / ensure_regular_time_intervals_with_aggregation if the column is describing rate of change.
class ColumnNames(str, Enum):
    BG = "bg_mM"
    DATETIME = "datetime"  # This shoud be INDEX
    P_NUM = "p_num"
    DOSE_UNITS = "dose_units"
    BOLUS = "bolus"
    FOOD_G = "food_g"
    MSG_TYPE = "msg_type"
    COB = "cob"
    CARB_AVAILABILITY = "carb_availability"
    IOB = "iob"
    INSULIN_AVAILABILITY = "insulin_availability"
    CALS = "cals"
    STEPS = "steps"
    HR_BPM = "hr_bpm"
    ACTIVITY = "activity"
    RECORD_TYPE = (
        "record_type"  # Extra info for the record type (CGM, CALIBRATION, etc.)
    )
    RATE = "rate"  # Basal rate units/hr
    BASAL_DURATION_MINS = "basal_duration_mins"  # Basal duration in minutes
    SUPR_BASAL_TYPE = "supr_basal_type"  # Supr basal type
    SUPR_RATE = "supr_rate"  # Supr basal rate


class DatasetConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # No extra fields allowed
        validate_assignment=True,  # Validate on assignment
        frozen=True,  # Immutable
    )

    source: DatasetSourceType  # Source type of the dataset
    cache_path: str | None = (
        None  # Cache path relative to the cache root (should include raw and processed directories)
    )
    required_files: list[str] = Field(
        default_factory=list
    )  # Optional raw data files of the dataset (empty list if no files required)

    # Metadata of the dataset
    url: str = Field(pattern=r"^https?://")
    description: str = Field(min_length=1)
    citation: str = Field(min_length=1)

    # For Kaggle datasets
    competition_name: str | None = (
        None  # Competition name of the dataset (for Kaggle datasets)
    )

    hf_dataset_id: str | None = None
