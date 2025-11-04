from concurrent.futures import ProcessPoolExecutor, as_completed
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
from src.data.preprocessing.generic_cleaning import erase_consecutive_nan_values
from src.data.preprocessing.pipeline import preprocessing_pipeline
from src.data.preprocessing.time_processing import get_most_common_time_interval
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
            # "rate": ColumnNames.RATE.value,
            "basalDurationMins": ColumnNames.BASAL_DURATION_MINS.value,
            "suprBasalType": ColumnNames.SUPR_BASAL_TYPE.value,
            "suprRate": ColumnNames.SUPR_RATE.value,
        }
    )
    # Drop the expected columns for now
    df.drop(columns=["expectedNormalBolus", "expectedExtendedBolus"], inplace=True)
    df.set_index(ColumnNames.DATETIME.value, inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)

    # Convert blood glucose from mg/dL to mmol/L
    df[ColumnNames.BG.value] = mg_dl_to_mmol_l(df, bgl_col=ColumnNames.BG.value)

    return df


def keep_overlapping_data(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        patient_df: A dataframe that has been through data_translation (datetime is the index)

    Keep data that has overlapping time windows for all table types.
    Note that not all patients have data for all table types so we only consider the table types that are present.

    Don't think we are losing too much data here. Most patients still have at least 6 months worth of data after this step except for some patients.
    - p216, p019, p081, p289, p138 (check 3.18 notebook for more details)
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

        # Find the latest start time of all table types
        if start_datetime is None or min_datetime > start_datetime:
            start_datetime = min_datetime

        # Find the earliest end time of all table types
        if end_datetime is None or max_datetime < end_datetime:
            end_datetime = max_datetime

    if start_datetime is None or end_datetime is None:
        return None

    has_overlap = start_datetime < end_datetime
    if not has_overlap:
        # This shouldn't happen
        logger.warning(
            f"Patient {patient_df[ColumnNames.P_NUM.value].iloc[0]} has no overlapping data"
        )
        return None

    # Filter using the index (datetime)
    return patient_df[
        (patient_df.index >= start_datetime) & (patient_df.index <= end_datetime)
    ]


def process_one_patient(
    df_raw: pd.DataFrame, debug: bool = False, verbose: bool = False
) -> pd.DataFrame:
    """
    Process the raw data for one patient:
        1. Translate the data (columns and units)
        2. Keep overlapping data (for bolus, wizard, cgm and basal)
        3. Rollover basal rate to the next few rows if the rate is not null
        4. preprocessing_pipeline (This include resampling and deriving cob and iob)
    """
    HOURS_OF_CONSECUTIVE_NAN_VALUES = 6
    df = df_raw.copy()

    # Translate the data
    df = data_translation(df)
    pid = df[ColumnNames.P_NUM.value].iloc[0]

    # Keep overlapping data
    logger.info(f"Keeping overlapping data for patient {pid}")
    df = keep_overlapping_data(df)
    if df is None:
        return None

    # Print data meta: span of datetime index and number of rows
    if verbose:
        start_dt = df.index.min()
        end_dt = df.index.max()
        food_g = df[df[ColumnNames.FOOD_G.value].notna()]
        bolus = df[df[ColumnNames.DOSE_UNITS.value].notna()]
        logger.info(
            f"Patient {pid} processed data spans from {start_dt} to {end_dt} ({(end_dt - start_dt)})"
        )
        logger.info(f"Number of rows with food intake: {len(food_g)}")
        logger.info(f"Number of rows with bolus: {len(bolus)}")

    # Resampling to constant interval, rollover basal rate and derive cob and iob
    df = preprocessing_pipeline(pid, df, use_aggregation=True)

    # Drop days with more than 6 hours of consecutive NaN values
    freq_mins = get_most_common_time_interval(df)
    max_consecutive_nan_values_per_day = (
        HOURS_OF_CONSECUTIVE_NAN_VALUES * 60
    ) // freq_mins
    # Note that it is possible that we have bolus from the deleted days. because we rollover first then delete.
    df = erase_consecutive_nan_values(
        df, max_consecutive_nan_values_per_day=max_consecutive_nan_values_per_day
    )

    # Debug only
    if debug:
        debug_dir = (
            get_project_root() / "cache" / "data" / "awesome_cgm" / "aleppo" / "debug"
        )
        os.makedirs(debug_dir, exist_ok=True)
        df.to_csv(debug_dir / f"p{pid}.csv", index=True)
    return df


def process_single_patient_file(patient_file_tuple: tuple) -> tuple:
    """
    Process a single patient file for parallel execution.

    Note: Standalone function created to support multiprocessing, as corruption issues
    occur when using class methods with ProcessPoolExecutor.

    Args:
        patient_file_tuple (tuple): Tuple containing (filename, interim_path, processed_path)
            where filename is the CSV filename, interim_path is the source directory,
            and processed_path is the destination directory.

    Returns:
        tuple: Tuple containing (p_num, df) where p_num is the patient ID
            and df is the processed DataFrame.
    """
    filename, interim_path, processed_path = patient_file_tuple

    df = pd.read_csv(interim_path / filename)
    save_path = processed_path / filename

    # Don't process if already exists
    if save_path.exists():
        # Read the existing file and return it
        df = pd.read_csv(save_path, index_col=0)
        p_num = df[ColumnNames.P_NUM.value].iloc[0]
        logger.info(f"Skipping pid {filename} because {save_path} already exists.")
        return (p_num, df)

    # Process the patient
    df = process_one_patient(df)

    if df is None or df.empty:
        raise ValueError(f"Processed data is None or empty for patient {filename}")

    p_num = df[ColumnNames.P_NUM.value].iloc[0]

    # Save the processed data
    df.to_csv(save_path, index=True)
    logger.info(f"Done processing pid {filename}")

    return (p_num, df)


# We can parallelize this by using multiprocessing because each patient is independent of each other.
def clean_all_patients(
    interim_path: Path,
    processed_path: Path,
    parallel: bool = True,
    max_workers: int = None,
) -> dict[str, pd.DataFrame]:
    """
    Clean all patients' data in the interim path and save the processed data to the processed path.

    Args:
        interim_path: Path to directory containing interim patient data files
        processed_path: Path to directory where processed data will be saved
        parallel: Whether to use parallel processing (default: True)
        max_workers: Maximum number of worker processes (None = use default)

    Returns:
        Dictionary mapping patient IDs to processed DataFrames
    """
    processed_data = {}
    os.makedirs(processed_path, exist_ok=True)

    # Get all filenames
    filenames = list(os.listdir(interim_path))
    total_patients = len(filenames)

    if parallel:
        logger.info(
            f"Processing {total_patients} patients in parallel with {max_workers} workers"
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prepare data tuples for parallel processing
            patient_file_tuples = [
                (filename, interim_path, processed_path) for filename in filenames
            ]

            # Submit all tasks
            future_to_filename = {
                executor.submit(
                    process_single_patient_file, patient_tuple
                ): patient_tuple[0]
                for patient_tuple in patient_file_tuples
            }

            # Collect results
            for index, future in enumerate(as_completed(future_to_filename), 1):
                filename = future_to_filename[future]
                progress = f"({index}/{total_patients})"
                try:
                    p_num, df = future.result()
                    processed_data[p_num] = df
                    logger.info(
                        f"Successfully processed patient {p_num} from {filename} {progress}"
                    )
                    logger.info(
                        f"{'-'*10}Done processing pid {filename} {progress} {'-'*10}"
                    )
                except Exception as exc:
                    logger.error(f"Patient {filename} generated an exception: {exc}")
                    logger.error(f"Error processing {filename} {progress}: {exc}")
    else:
        # Sequential processing (original code)
        for index, filename in enumerate(filenames, 1):
            progress = f"({index}/{total_patients})"
            df = pd.read_csv(interim_path / filename)

            save_path = processed_path / filename
            # Don't process the patient if the processed data already exists
            if save_path.exists():
                logger.info(
                    f"Skipping pid {filename} because {save_path} already exists {progress}."
                )
                df = pd.read_csv(save_path, index_col=0)
                p_num = df[ColumnNames.P_NUM.value].iloc[0]
                processed_data[p_num] = df
                continue

            # Process the patient
            df = process_one_patient(df)

            if df is None or df.empty:
                raise ValueError(
                    f"Processed data is None or empty for patient {filename}"
                )

            p_num = str(df[ColumnNames.P_NUM.value].iloc[0]).split(".")[0]
            processed_data[p_num] = df

            df.to_csv(save_path, index=True)
            logger.info(f"{'-'*10}Done processing pid {filename} {progress} {'-'*10}")

    return processed_data
