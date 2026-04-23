# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

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

from src.data.cache_manager import get_cache_manager
from src.data.physiological.carb_model.carb_model import (
    create_cob_and_carb_availability_cols,
)
from src.data.physiological.insulin_model.insulin_model import (
    create_iob_and_ins_availability_cols,
)
from src.data.preprocessing.pipeline import preprocessing_pipeline
from src.data.preprocessing.sampling import ensure_regular_time_intervals
from src.data.utils.patient_id import format_patient_id

logger = logging.getLogger(__name__)

_RAW_TXT_FILENAMES = {
    "ilet": "IOBP2DeviceiLet.txt",
    "demo": "IOBP2DiabScreening.txt",
}


def load_lynch2022_raw_dataset(base_dir: Path) -> pd.DataFrame:
    """
    Load and process Lynch 2022 txt tables into a single dataframe.

    Reads from the pipe-separated flat files in the "Data Tables" directory,
    matching the BabelBetes IOBP2 approach. CGM and insulin are co-logged by
    the iLet device into a single file (IOBP2DeviceiLet.txt).

    Args:
        base_dir: Path to the "Data Tables" directory (pipe-separated txt files).

    Returns:
        DataFrame with columns needed for cleaning.
    """
    base_dir = Path(base_dir)
    missing = [
        fname
        for fname in _RAW_TXT_FILENAMES.values()
        if not (base_dir / fname).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required files in {base_dir}: {missing}")

    # Load pipe-separated txt files
    ilet_data = pd.read_csv(
        base_dir / _RAW_TXT_FILENAMES["ilet"], sep="|", low_memory=False
    )
    demo_data = pd.read_csv(
        base_dir / _RAW_TXT_FILENAMES["demo"], sep="|", low_memory=False
    )

    logger.info("Loaded iLet data with shape %s", ilet_data.shape)
    logger.info("Loaded Demographics data with shape %s", demo_data.shape)

    # Parse timestamps
    ilet_data["DeviceDtTm"] = pd.to_datetime(ilet_data["DeviceDtTm"], errors="coerce")

    # Preserve meal size text before any numeric coercion (Bug B fix)
    ilet_data["meal_size_text"] = (
        ilet_data["MealSize"].fillna("").astype(str).str.strip()
        if "MealSize" in ilet_data.columns
        else ""
    )

    # Extract CGM rows: only rows with a valid numeric CGM reading
    ilet_data["CGMVal"] = pd.to_numeric(ilet_data["CGMVal"], errors="coerce")
    cgm_rows = ilet_data.dropna(subset=["CGMVal"]).copy()

    # Apply BabelBetes sentinel clamping: sensor-low → 40, sensor-high → 400
    cgm_rows.loc[cgm_rows["CGMVal"] <= 39, "CGMVal"] = 40.0
    cgm_rows.loc[cgm_rows["CGMVal"] >= 401, "CGMVal"] = 400.0

    # Drop duplicate CGM readings for same patient+timestamp
    cgm_rows = cgm_rows.drop_duplicates(subset=["PtID", "DeviceDtTm"])

    # Build composite insulin dose from previous-step delivery components
    for col in ["BasalDelivPrev", "BolusDelivPrev", "MealBolusDelivPrev"]:
        if col in cgm_rows.columns:
            cgm_rows[col] = pd.to_numeric(cgm_rows[col], errors="coerce").fillna(0.0)
        else:
            cgm_rows[col] = 0.0
    cgm_rows["dose_units"] = (
        cgm_rows["BasalDelivPrev"]
        + cgm_rows["BolusDelivPrev"]
        + cgm_rows["MealBolusDelivPrev"]
    )

    # Note on timestamp shift: BabelBetes shifts insulin timestamps back 5 min
    # (delivery is for the previous step). In our pipeline, we keep the original
    # DeviceDtTm for both CGM and insulin because ensure_regular_time_intervals_with_aggregation
    # rounds to 5-min bins anyway, making a 5-min shift within the same bin immaterial.
    # Keeping the original timestamp also preserves perfect CGM alignment.
    cgm_rows["time"] = cgm_rows["DeviceDtTm"]

    # food_g = 0: MealSize is categorical text, kept separately as meal_size_text
    cgm_rows["food_g"] = 0.0

    # Extract demographics
    demo_cols = ["PtID", "DiagAge", "Sex"]
    available_demo_cols = [col for col in demo_cols if col in demo_data.columns]
    demo_subset = demo_data[available_demo_cols].copy()
    if "DiagAge" in demo_subset.columns:
        demo_subset["DiagAge"] = pd.to_numeric(demo_subset["DiagAge"], errors="coerce")

    # Build output frame
    out_cols = ["PtID", "time", "CGMVal", "dose_units", "food_g", "meal_size_text"]
    out = cgm_rows[out_cols].rename(columns={"PtID": "id", "CGMVal": "gl"})

    # Merge demographics
    out = pd.merge(
        out,
        demo_subset.rename(columns={"PtID": "id"}),
        on="id",
        how="left",
    )

    rename_dict = {}
    if "DiagAge" in out.columns:
        rename_dict["DiagAge"] = "age_at_diagnosis"
    if "Sex" in out.columns:
        rename_dict["Sex"] = "sex"
    out = out.rename(columns=rename_dict)

    # Remove rows without valid time or glucose
    out = (
        out.dropna(subset=["time", "gl"])
        .sort_values(["id", "time"])
        .reset_index(drop=True)
    )

    # Add metadata columns
    out["type"] = 1  # Type 1 diabetes
    out["device"] = "iLet CGM"
    out["dataset"] = "lynch2022"

    if "age_at_diagnosis" not in out.columns:
        out["age_at_diagnosis"] = np.nan
    if "sex" not in out.columns:
        out["sex"] = np.nan
    out["insulinModality"] = 1  # iLet is automated insulin delivery
    out["hr_bpm"] = np.nan
    out["steps"] = np.nan
    out["cals"] = np.nan
    out["activity"] = np.nan

    logger.info(
        "Loaded Lynch 2022 raw dataset with %d rows across %d subjects",
        len(out),
        out["id"].nunique(),
    )
    return out


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

    # Patient ID with standardized format: lyn_###
    df["p_num"] = df["id"].map(lambda pid: format_patient_id("lynch_2022", pid))

    # dose_units and food_g are already calculated in load_lynch2022_raw_dataset
    # Ensure they exist and have proper types
    if "dose_units" not in df.columns:
        df["dose_units"] = 0.0
    else:
        df["dose_units"] = df["dose_units"].fillna(0.0)

    if "food_g" not in df.columns:
        df["food_g"] = 0.0
    else:
        df["food_g"] = df["food_g"].fillna(0.0)

    # Ensure other physiological columns exist
    for col in ["hr_bpm", "steps", "cals", "activity"]:
        if col not in df.columns:
            df[col] = np.nan

    df["msg_type"] = "bg"

    if "meal_size_text" not in df.columns:
        df["meal_size_text"] = ""
    else:
        df["meal_size_text"] = df["meal_size_text"].fillna("").astype(str)

    cols = [
        "p_num",
        "datetime",
        "bg_mM",
        "dose_units",
        "food_g",
        "meal_size_text",
        "hr_bpm",
        "steps",
        "cals",
        "activity",
        "msg_type",
        "age_at_diagnosis",
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


def clean_lynch2022_test_data(
    raw_data: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
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


def process_single_patient_data(
    patient_data_tuple: tuple, store_in_between_data: bool = False
) -> tuple[str, pd.DataFrame]:
    """
    Run preprocessing pipeline for a single patient's training data.
    Args:
        patient_data_tuple: (p_num, data, generic_patient_start_date)
    """
    p_num, data, _ = patient_data_tuple
    logger.info(f"Processing Lynch patient {p_num} (train)")
    data_copy = data.copy()
    data_copy["datetime"] = pd.to_datetime(data_copy["datetime"], errors="coerce")
    data_copy = (
        data_copy.dropna(subset=["datetime"])
        .set_index("datetime", drop=True)
        .sort_index()
    )

    if store_in_between_data:
        cache_manager = get_cache_manager()
        cache_dir = cache_manager.get_cleaning_step_data_path("lynch_2022")
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_copy.to_csv(cache_dir / f"{p_num}_pre_pipeline.csv")

    processed_data = preprocessing_pipeline(p_num, data_copy, use_aggregation=True)
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
        instance_copy["datetime"] = pd.to_datetime(
            instance_copy["datetime"], errors="coerce"
        )
        instance_copy = (
            instance_copy.dropna(subset=["datetime"])
            .set_index("datetime", drop=True)
            .sort_index()
        )

        instance_copy, freq = ensure_regular_time_intervals(instance_copy)
        instance_copy = instance_copy.pipe(
            create_cob_and_carb_availability_cols, freq
        ).pipe(create_iob_and_ins_availability_cols, freq)

        if patient_cache_dir is not None:
            instance_copy.to_csv(patient_cache_dir / f"{instance_id}.csv")

        processed_rows[instance_id] = instance_copy

    return pid, processed_rows
