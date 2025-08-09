import pandas as pd
from src.data.preprocessing.sampling import (
    grouped_ensure_regular_time_intervals_with_interpolation,
    InterpolationMethod,
)
from src.data.preprocessing.time_processing import ensure_datetime_index
from datetime import timedelta
from pydantic import BaseModel
from src.data.preprocessing.unit_conversion import mg_dl_to_mmol_l
from src.data.preprocessing.sampling import create_subpatients


class PreprocessConfig(BaseModel):
    day_start_time_hours: int = 4
    gap_threshold: timedelta = timedelta(hours=2)
    min_sample_size: int = 100
    sampling_frequency_min: int = 5
    sampling_interpolation: InterpolationMethod = InterpolationMethod.LINEAR


default_config = PreprocessConfig()


# Gluroo data one dataframe per patient
def clean_cgm_data(
    df_raw: pd.DataFrame,
    config: PreprocessConfig = default_config,
) -> pd.DataFrame:
    day_start_time: pd.Timedelta = pd.Timedelta(hours=config.day_start_time_hours)

    df = df_raw.copy()
    df = ensure_datetime_index(df)
    df.index = df.index.tz_localize(None)

    df["day_start_shift"] = (df.index - day_start_time).date
    df = create_subpatients(
        df.reset_index(),
        datetime_col="date",
        gap_threshold=config.gap_threshold,
        min_sample_size=config.min_sample_size,
    ).set_index("date")

    df = data_translation(df)

    df = grouped_ensure_regular_time_intervals_with_interpolation(
        df,
        datetime_col="datetime",
        patient_col="p_num",
        target_interval_minutes=config.sampling_frequency_min,
        interpolation_method=config.sampling_interpolation.value,
    )
    # quick fix: train validation splitter needs datetime to be a col
    df = df.reset_index()

    return df


def data_translation(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.rename(
        columns={
            "bgl": "bg-0:00",
        }
    )
    df["datetime"] = df.index

    # Convert blood glucose from mg/dL to mmol/L
    df["bg-0:00"] = mg_dl_to_mmol_l(df, bgl_col="bg-0:00")

    return df
