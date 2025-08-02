import pandas as pd
from src.data.preprocessing.sampling import ensure_regular_time_intervals_with_interpolation, InterpolationMethod
from src.data.preprocessing.time_processing import ensure_datetime_index
import polars as pl
from datetime import timedelta
from pydantic import BaseModel
from src.data.preprocessing.unit_conversion import mg_dl_to_mmol_l

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
        min_sample_size=config.min_sample_size
    )

    # Translate data to the correct format
    df = data_translation(df)
    df = ensure_regular_time_intervals_with_interpolation(
        df, 
        target_interval_minutes=config.sampling_frequency_min, 
        interpolation_method=config.sampling_interpolation.value
    )

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


def create_subpatients(    
    df: pd.DataFrame, 
    gap_threshold: timedelta = timedelta(hours=2), 
    min_sample_size: int = 100, 
    datetime_col: str = "datetime", 
    p_col: str = "p_num"
) -> pd.DataFrame:
    """If large gaps occur for a particular patient, then split them
    into subpatients"""
    pl_df = pl.DataFrame(df).sort(datetime_col, descending=False)

    pl_df = pl_df.with_columns(
        pl.col(datetime_col).diff().over(p_col).alias("diff")
    )

    pl_df = pl_df.with_columns(
        pl.when(pl.col("diff") > gap_threshold)
        .then(1)
        .otherwise(0)
        .cum_sum()
        .over(p_col)
        .alias("group_ids")
    )

    relevant_groups = (
        pl_df
        .group_by(p_col, "group_ids")
        .agg(pl.len())
        .filter(pl.col("len") > min_sample_size)
    )

    pl_df = (
        pl_df
        .join(relevant_groups, on=["p_num", "group_ids"])
        .with_columns(
            (pl.col("p_num").cast(str) + "_" + pl.col("group_ids").cast(str)).alias("p_num")
        )
        .drop(["len", "group_ids", "diff"])
    )

    return pl_df.to_pandas() # :(

    
