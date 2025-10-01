"""
Data cleaning utilities for the Lynch 2022 IOBP2 RCT dataset.

Minimal, Kaggle-style helpers to:
- load SAS tables
- standardize to the common schema
- run preprocessing for train/test flows
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from src.data.cache_manager import get_cache_manager
from src.data.physiological.carb_model.carb_model import create_cob_and_carb_availability_cols
from src.data.physiological.insulin_model.insulin_model import create_iob_and_ins_availability_cols
from src.data.preprocessing.pipeline import preprocessing_pipeline
from src.data.preprocessing.sampling import ensure_regular_time_intervals

logger = logging.getLogger(__name__)

_RAW_SAS_FILENAMES = {
    "cgm": "iobp2devicecgm.sas7bdat",
    "demo": "iobp2diabscreening.sas7bdat",
    "roster": "iobp2ptroster.sas7bdat",
}


def load_lynch2022_raw_dataset(base_dir: Path) -> pd.DataFrame:
    """
    Load and merge Lynch 2022 SAS tables into a single dataframe.

    Args:
        base_dir: Path to the "Data Tables in SAS" directory.

    Returns:
        DataFrame with columns needed for cleaning.
    """
    base_dir = Path(base_dir)
    missing = [fname for fname in _RAW_SAS_FILENAMES.values() if not (base_dir / fname).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required SAS tables in {base_dir}: {missing}")

    # Load SAS files
    data, _ = pyreadstat.read_sas7bdat(str(base_dir / _RAW_SAS_FILENAMES["cgm"]))
    demo, _ = pyreadstat.read_sas7bdat(str(base_dir / _RAW_SAS_FILENAMES["demo"]))
    age, _ = pyreadstat.read_sas7bdat(str(base_dir / _RAW_SAS_FILENAMES["roster"]))

    # Select and merge
    data = data[["PtID", "DeviceDtTm", "Value"]]
    demo = demo[["PtID", "InsModPump", "Sex"]]
    age = age[["PtID", "AgeAsofEnrollDt"]]

    merged = (
        data.merge(demo, on="PtID", how="left")
        .merge(age, on="PtID", how="left")
        .rename(
            columns={
                "PtID": "id",
                "DeviceDtTm": "time",
                "Value": "gl",
                "AgeAsofEnrollDt": "age",
                "Sex": "sex",
                "InsModPump": "insulinModality",
            }
        )
    )

    # Clean
    merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
    merged = merged.dropna(subset=["time", "gl"]).sort_values(["id", "time"]).reset_index(drop=True)
    merged["type"] = 1
    merged["device"] = "Dexcom G6"
    merged["dataset"] = "lynch2022"
    merged["insulinModality"] = merged["insulinModality"].notna().astype(int)

    logger.info("Loaded Lynch 2022 raw dataset with %d rows across %d subjects", len(merged), merged["id"].nunique())
    return merged


def clean_lynch2022_train_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw Lynch data to the common training schema.
    """
    df = raw_data.copy()
    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Convert mg/dL to mmol/L
    df["bg_mM"] = pd.to_numeric(df["gl"], errors="coerce") / 18.0
    df = df.dropna(subset=["bg_mM"])

    # Patient ID and required columns for downstream pipeline
    df["p_num"] = df["id"].map(lambda pid: f"lynch_{int(pid)}")
    df["dose_units"] = 0.0
    df["food_g"] = 0.0
    df["hr_bpm"] = np.nan
    df["steps"] = np.nan
    df["cals"] = np.nan
    df["activity"] = np.nan
    df["msg_type"] = "bg"

    cols = [
        "p_num",
        "datetime",
        "bg_mM",
        "dose_units",
        "food_g",
        "hr_bpm",
        "steps",
        "cals",
        "activity",
        "msg_type",
        "age",
        "sex",
        "insulinModality",
        "type",
        "device",
        "dataset",
    ]
    df = df[cols].sort_values(["p_num", "datetime"]).reset_index(drop=True)

    # Optional debug cache
    cache_manager = get_cache_manager()
    cache_dir = cache_manager.get_cleaning_step_data_path("lynch_2022")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_dir / "train_after_cleaning.csv", index=False)

    logger.info("Prepared Lynch train data with %d patients", df["p_num"].nunique())
    return df


def clean_lynch2022_test_data(raw_data: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Create a simple test-style nested structure: one instance per patient.
    """
    cleaned = clean_lynch2022_train_data(raw_data)
    patient_groups = defaultdict(dict)
    for idx, (p_num, group) in enumerate(cleaned.groupby("p_num"), start=1):
        instance_id = f"instance_{idx:03d}"
        patient_groups[p_num][instance_id] = group.reset_index(drop=True)
    logger.info("Prepared Lynch test-style data for %d patients", len(patient_groups))
    return patient_groups


def process_single_patient_data(patient_data_tuple: tuple, store_in_between_data: bool = False) -> tuple[str, pd.DataFrame]:
    """
    Run preprocessing pipeline for a single patient's training data.
    Args:
        patient_data_tuple: (p_num, data, generic_patient_start_date)
    """
    p_num, data, _ = patient_data_tuple
    logger.info(f"Processing Lynch patient {p_num} (train)")
    data_copy = data.copy()
    data_copy["datetime"] = pd.to_datetime(data_copy["datetime"], errors="coerce")
    data_copy = data_copy.dropna(subset=["datetime"]).set_index("datetime", drop=True).sort_index()

    if store_in_between_data:
        cache_manager = get_cache_manager()
        cache_dir = cache_manager.get_cleaning_step_data_path("lynch_2022")
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_copy.to_csv(cache_dir / f"{p_num}_pre_pipeline.csv")

    processed_data = preprocessing_pipeline(p_num, data_copy)
    return p_num, processed_data


def process_patient_prediction_instances(
    patient_item: tuple,
    base_cache_path: Path,
    generic_patient_start_date: pd.Timestamp = pd.Timestamp("2024-01-01"),
    save_individual_files: bool = False,
) -> tuple[str, dict[str, pd.DataFrame]]:
    """
    Process Lynch test-style data per patient into regular intervals with COB/IOB.
    """
    pid, patient_data = patient_item
    processed_rows = {}

    patient_cache_dir = None
    if save_individual_files:
        patient_cache_dir = Path(base_cache_path) / pid
        patient_cache_dir.mkdir(parents=True, exist_ok=True)

    for instance_id, instance_df in patient_data.items():
        logger.info(f"Processing Lynch patient {pid}, instance {instance_id}")
        instance_copy = instance_df.copy()
        instance_copy["datetime"] = pd.to_datetime(instance_copy["datetime"], errors="coerce")
        instance_copy = instance_copy.dropna(subset=["datetime"]).set_index("datetime", drop=True).sort_index()

        instance_copy, freq = ensure_regular_time_intervals(instance_copy)
        instance_copy = instance_copy.pipe(create_cob_and_carb_availability_cols, freq).pipe(
            create_iob_and_ins_availability_cols, freq
        )

        if patient_cache_dir is not None:
            instance_copy.to_csv(patient_cache_dir / f"{instance_id}.csv")

        processed_rows[instance_id] = instance_copy

    return pid, processed_rows
    