from pathlib import Path
import pandas as pd

# from src.data.preprocessing.sampling import (
#     grouped_ensure_regular_time_intervals_with_interpolation,
#     InterpolationMethod,
# )
# from src.data.preprocessing.time_processing import ensure_datetime_index
from datetime import timedelta
from pydantic import BaseModel
from src.data.models import ColumnNames
from src.data.preprocessing.pipeline import preprocessing_pipeline
from src.utils.unit import mg_dl_to_mmol_l
import os
import logging

# from src.data.preprocessing.sampling import create_subpatients
from src.utils.os_helper import get_project_root


logger = logging.getLogger(__name__)


class PreprocessConfig(BaseModel):
    day_start_time_hours: int = 4
    gap_threshold: timedelta = timedelta(hours=2)
    min_sample_size: int = 100
    sampling_frequency_min: int = 5
    # sampling_interpolation: InterpolationMethod = InterpolationMethod.LINEAR


default_config = PreprocessConfig()


# 	pid	date	tableType	bolusType	normalBolus	expectedNormalBolus	extendedBolus	expectedExtendedBolus	bgInput	foodG	iob	cr	isf	bgMgdl	rate	suprBasalType	suprRate
def data_translation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    'pid' -> p_num
    'date' -> datetime
    'tableType' -> msg_type (bolus, wizard, cgm). wizard contains information like carbs intake
    'eventType': This is bolus type
    'normal': Number of units of normal bolus
    'expectedNormal'
    'extended': Number of units for extended delivery
    'expectedExtended'
    'bgInput' -> Blood glucose as inputted into wizard in mg
    'carbInput' -> Carbohydrates as inputted into wizard in mg
    'iob': Units of insulin on board
    'cr': Number of mg carbs covered by unit of insulin. Not used yet but we can find a way to give model a hint about the slope of the glucose curve
    'isf': Number of bgs covered by unit of insulin. Same as above
    'recordType': CGM | CALIBRATION
    'glucoseValue' -> bg_mM
    """
    df = df_raw.copy()
    # TODO: Rename to the correct column names
    df = df.rename(
        columns={
            "pid": ColumnNames.P_NUM.value,
            "date": ColumnNames.DATETIME.value,
            "tableType": ColumnNames.MSG_TYPE.value,
            "normalBolus": ColumnNames.DOSE_UNITS.value,
            # "expectedNormalBolus": ColumnNames.EXPECTED_NORMAL.value,
            # "extendedBolus": ColumnNames.EXTENDED.value,
            # "expectedExtendedBolus": ColumnNames.EXPECTED_EXTENDED.value,
            # "bgInput": ColumnNames.BG.value, todo: Maybe tag the record type as bgInput
            "foodG": ColumnNames.FOOD_G.value,
            "iob": ColumnNames.IOB.value,
            # "cr": ColumnNames.INSULIN_CARB_RATIO.value,
            # "isf": ColumnNames.ISF.value,
            "recordType": ColumnNames.RECORD_TYPE.value,
            "bgMgdl": ColumnNames.BG.value,
            "rate": ColumnNames.RATE.value,
            "suprBasalType": ColumnNames.SUPR_BASAL_TYPE.value,
            "suprRate": ColumnNames.SUPR_RATE.value,
        }
    )
    df.drop(columns=["expectedNormalBolus", "expectedExtendedBolus"], inplace=True)
    df.set_index(ColumnNames.DATETIME.value, inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)

    # Convert blood glucose from mg/dL to mmol/L
    df[ColumnNames.BG.value] = mg_dl_to_mmol_l(df, bgl_col=ColumnNames.BG.value)

    return df


def keep_overlapping_data(patient_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Figure out how much data there is left for each patient after this step.
    """
    Args:
        patient_df: A dataframe that has been through data_translation (datetime is the index)
    Keep data that has overlapping time windows for all table types.
    Note that not all patients have data for all table types so we only consider the table types that are present.
    """
    table_types = patient_df[ColumnNames.MSG_TYPE.value].unique()
    start_datetime = None  # This should be the max of all the min datetimes
    end_datetime = None  # This should be the min of all the max datetimes

    for table_type in table_types:
        table_data = patient_df[patient_df[ColumnNames.MSG_TYPE.value] == table_type]
        if table_data.empty:
            continue

        min_datetime = table_data.index.min()
        max_datetime = table_data.index.max()

        if start_datetime is None or min_datetime > start_datetime:
            start_datetime = min_datetime
        if end_datetime is None or max_datetime < end_datetime:
            end_datetime = max_datetime

    if start_datetime is None or end_datetime is None:
        return None

    has_overlap = start_datetime < end_datetime
    if not has_overlap:
        return None

    # Filter using the index (datetime)
    return patient_df[
        (patient_df.index >= start_datetime) & (patient_df.index <= end_datetime)
    ]


def process_one_patient(
    df_raw: pd.DataFrame,
    to_csv: bool = False,
) -> pd.DataFrame:
    # TODO: Maybe need to drop days that don't have enough cgm readings
    """
    Process the raw data for one patient:
    1. Translate the data (columns and units)
    2. Keep overlapping data (for bolus, wizard, cgm and basal)
    3. Rollover basal rate to the next few rows if the rate is not null
    4. preprocessing_pipeline (This include resampling and deriving cob and iob)
    """
    df = df_raw.copy()

    # Translate the data
    df = data_translation(df)
    pid = df[ColumnNames.P_NUM.value].iloc[0]

    # Keep overlapping data
    df = keep_overlapping_data(df)
    if df is None:
        return None

    # Print data meta: span of datetime index and number of rows
    start_dt = df.index.min()
    end_dt = df.index.max()
    print(
        f"Patient {pid} processed data spans from {start_dt} to {end_dt} ({(end_dt - start_dt)})"
    )

    food_g = df[df[ColumnNames.FOOD_G.value].notna()]
    print(f"Number of rows with food intake: {len(food_g)}")

    bolus = df[df[ColumnNames.DOSE_UNITS.value].notna()]
    print(f"Number of rows with bolus: {len(bolus)}")

    # Let the pipeline handle the rest
    # This derives iob and cob which we is information we already have
    df = preprocessing_pipeline(pid, df, use_aggregation=True)

    # Debug only
    if to_csv:
        debug_dir = (
            get_project_root() / "cache" / "data" / "awesome_cgm" / "aleppo" / "debug"
        )
        os.makedirs(debug_dir, exist_ok=True)
        df.to_csv(debug_dir / f"p{pid}.csv", index=True)
    return df


def convert_basal_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rollover the basal rate to the next few rows if the rate is not null
    """
    df = df.copy()
    df[ColumnNames.RATE.value] = df[ColumnNames.RATE.value].astype(float)
    return df


# We can parallelize this by using multiprocessing because each patient is independent of each other.
def clean_all_patients(
    interim_path: Path,
    processed_path: Path,
) -> dict[str, pd.DataFrame]:
    """
    Clean all patients' data in the interim path and save the processed data to the processed path.
    """
    processed_data = {}
    for pid in os.listdir(interim_path):
        df = pd.read_csv(interim_path / pid)
        p_num = df[ColumnNames.P_NUM.value].iloc[0]

        df = process_one_patient(df)
        processed_data[p_num] = df

        df.to_csv(processed_path / pid, index=False)
        logger.info(f"{'-'*10} Done processing pid {pid} {'-'*10}")
    return processed_data
