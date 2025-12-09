# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Data cleaning utilities for the Brown 2019 DCLP3 diabetes dataset.

Pipeline Steps:
1. Parse timestamps (use DataDtTm_adjusted for basal/bolus - corrects bad dates)
2. Minimal rename (PtID → p_num for groupby compatibility)
3. Floor timestamps to 5-min grid (preserves causality)
4. Aggregate collisions (CGM=mean, Bolus=sum, Basal=last)
5. Create regular 5-min grid for CGM
6. Merge insulin data onto CGM backbone
7. Fill missing values (bolus=0, basal=forward-fill)
8. Final rename to standard schema + select columns

Notes:
- 168 total patients, 125 have insulin pump data, 43 have CGM only
- Basal is event-based (rate changes), needs forward-fill
- Bolus is discrete events, sum when multiple in same bin
- ~81% of NaN basal values are from 43 patients without pump data
- ~19% are leading NaN before first recorded rate change
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.models import ColumnNames
from src.data.preprocessing.data_splitting import split_multipatient_dataframe
from src.data.preprocessing.sampling import (
    ensure_regular_time_intervals_with_aggregation,
)
from src.utils.os_helper import get_project_root
from src.utils.unit import mg_dl_to_mmol_l

logger = logging.getLogger(__name__)

CACHE_DIR = get_project_root() / "cache" / "data" / "brown_2019"
DATA_DIR = (
    CACHE_DIR / "raw" / "DCLP3 Public Dataset - Release 3 - 2022-08-04" / "Data Files"
)

# Raw column names from Brown 2019 dataset
RAW_COLS = {
    "patient_id": "PtID",
    "cgm_value": "CGM",
    "period": "Period",
    "basal_rate": "CommandedBasalRate",
    "bolus_amount": "BolusAmount",
    "bolus_type": "BolusType",
}


def load_raw_brown_2019_data(
    data_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data files from the Brown 2019 DCLP3 dataset.

    Args:
        data_dir: Optional path to the directory containing the data files.
                  Defaults to standard location if not provided.

    Returns:
        Tuple of (cgm_df, basal_df, bolus_df) DataFrames.

    Raises:
        FileNotFoundError: If data directory doesn't exist.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data tables directory not found: {data_dir}. "
            "Refer to README for download instructions."
        )

    # Load raw files
    cgm_df = pd.read_csv(data_dir / "cgm.txt", sep="|")
    basal_df = pd.read_csv(
        data_dir / "Pump_BasalRateChange.txt", sep="|", low_memory=False
    )
    bolus_df = pd.read_csv(data_dir / "Pump_BolusDelivered.txt", sep="|")

    # Fix CGM unnamed column (data format quirk - CGM values are in unnamed column)
    if "Unnamed: 3" in cgm_df.columns:
        cgm_df = cgm_df.rename(columns={"Unnamed: 3": RAW_COLS["cgm_value"]})

    # Sanity checks
    assert not cgm_df.empty, "CGM data file is empty"
    assert (
        RAW_COLS["patient_id"] in cgm_df.columns
    ), f"Expected {RAW_COLS['patient_id']} column in CGM data"
    assert "DataDtTm" in cgm_df.columns, "Expected DataDtTm column in CGM data"

    logger.info(
        f"Brown 2019 loaded raw CGM:   {len(cgm_df):,} rows, {cgm_df[RAW_COLS['patient_id']].nunique()} patients"
    )
    logger.info(
        f"Brown 2019 loaded raw Basal: {len(basal_df):,} rows, {basal_df[RAW_COLS['patient_id']].nunique()} patients"
    )
    logger.info(
        f"Brown 2019 loaded raw Bolus: {len(bolus_df):,} rows, {bolus_df[RAW_COLS['patient_id']].nunique()} patients"
    )

    return cgm_df, basal_df, bolus_df


def clean_brown_2019_data(
    cgm_df: pd.DataFrame,
    basal_df: pd.DataFrame,
    bolus_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean and merge the Brown 2019 DCLP3 dataset into a single DataFrame.

    Uses raw column names throughout the pipeline, then renames to standard
    schema at the end (Step 8).

    Args:
        cgm_df: Raw CGM data.
        basal_df: Raw Basal insulin data.
        bolus_df: Raw Bolus insulin data.

    Returns:
        Cleaned and merged DataFrame with columns:
        - datetime (index)
        - p_num: Patient ID
        - period: "1. Baseline" or "2. Post Randomization"
        - bg_mM: Blood glucose mmol/L
        - bg_mgdL: Blood glucose mg/dL
        - rate: Basal rate U/hr (NaN for patients without pump)
        - dose_units: Bolus insulin U
        - bolus_type: "Standard", "Extended", etc.
    """
    logger.info("Cleaning Brown 2019 DCLP3 dataset...")

    # Make copies to avoid modifying originals
    cgm_df = cgm_df.copy()
    basal_df = basal_df.copy()
    bolus_df = bolus_df.copy()

    # ========== STEP 1: Parse timestamps ==========
    # CGM has format: '11DEC17:23:59:25' (DDmmmYY:HH:MM:SS)
    dt_col = ColumnNames.DATETIME.value
    cgm_df[dt_col] = pd.to_datetime(cgm_df["DataDtTm"], format="%d%b%y:%H:%M:%S")

    # Basal/Bolus: use DataDtTm_adjusted if available (corrected timestamps)
    # Some patients (114, 165) have incorrect dates in DataDtTm (e.g., 2010 instead of 2018)
    basal_df[dt_col] = pd.to_datetime(
        basal_df["DataDtTm_adjusted"].fillna(basal_df["DataDtTm"])
    )
    bolus_df[dt_col] = pd.to_datetime(
        bolus_df["DataDtTm_adjusted"].fillna(bolus_df["DataDtTm"])
    )

    logger.info(
        f"CGM datetime range:   {cgm_df[dt_col].min()} to {cgm_df[dt_col].max()}"
    )
    logger.info(
        f"Basal datetime range: {basal_df[dt_col].min()} to {basal_df[dt_col].max()}"
    )
    logger.info(
        f"Bolus datetime range: {bolus_df[dt_col].min()} to {bolus_df[dt_col].max()}"
    )

    # ========== STEP 2: Minimal rename (only PtID → p_num for groupby) ==========
    # Keep all other raw column names until final step
    p_num_col = ColumnNames.P_NUM.value
    cgm_df = cgm_df.rename(columns={RAW_COLS["patient_id"]: p_num_col})
    basal_df = basal_df.rename(columns={RAW_COLS["patient_id"]: p_num_col})
    bolus_df = bolus_df.rename(columns={RAW_COLS["patient_id"]: p_num_col})

    # Drop raw datetime columns (we have 'datetime' now)
    cgm_df = cgm_df.drop(columns=["DataDtTm"])
    basal_df = basal_df.drop(columns=["DataDtTm", "DataDtTm_adjusted", "RecID"])
    bolus_df = bolus_df.drop(columns=["DataDtTm", "DataDtTm_adjusted", "RecID"])

    # ========== STEP 3: Floor timestamps to 5-min grid ==========
    cgm_df[dt_col] = cgm_df[dt_col].dt.floor("5min")
    basal_df[dt_col] = basal_df[dt_col].dt.floor("5min")
    bolus_df[dt_col] = bolus_df[dt_col].dt.floor("5min")

    # ========== STEP 4: Aggregate collisions ==========
    # When multiple readings fall in same 5-min bin:
    # - CGM: mean (duplicate readings are similar)
    # - Bolus: SUM (don't lose any insulin!)
    # - Basal: last (most recent rate is what's active)

    cgm_agg = (
        cgm_df.groupby([p_num_col, dt_col])
        .agg({RAW_COLS["cgm_value"]: "mean", RAW_COLS["period"]: "first"})
        .reset_index()
    )

    bolus_agg = (
        bolus_df.groupby([p_num_col, dt_col])
        .agg({RAW_COLS["bolus_amount"]: "sum", RAW_COLS["bolus_type"]: "first"})
        .reset_index()
    )

    basal_agg = (
        basal_df.groupby([p_num_col, dt_col])
        .agg({RAW_COLS["basal_rate"]: "last"})
        .reset_index()
    )

    logger.info(
        f"After collision aggregation - CGM: {len(cgm_agg):,}, Basal: {len(basal_agg):,}, Bolus: {len(bolus_agg):,}"
    )

    # ========== STEP 5: Create regular 5-min grid for CGM ==========
    cgm_indexed = cgm_agg.set_index(dt_col)
    patient_dict = split_multipatient_dataframe(cgm_indexed, patient_col=p_num_col)

    logger.info(f"Creating regular 5-min grid for {len(patient_dict)} patients...")

    processed_patients = {}
    for i, (pid, pdf) in enumerate(patient_dict.items()):
        processed_df, freq = ensure_regular_time_intervals_with_aggregation(pdf)
        processed_patients[pid] = processed_df

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(patient_dict)} patients...")

    cgm_regular = pd.concat(processed_patients.values()).reset_index()
    logger.info(
        f"Regular grid created: {len(cgm_regular):,} rows (added {len(cgm_regular) - len(cgm_agg):,} gap rows)"
    )

    # ========== STEP 6: Merge insulin data onto CGM backbone ==========
    merged = cgm_regular.merge(
        bolus_agg[
            [p_num_col, dt_col, RAW_COLS["bolus_amount"], RAW_COLS["bolus_type"]]
        ],
        on=[p_num_col, dt_col],
        how="left",
    )

    merged = merged.merge(
        basal_agg[[p_num_col, dt_col, RAW_COLS["basal_rate"]]],
        on=[p_num_col, dt_col],
        how="left",
    )

    logger.info(
        f"After merge: {len(merged):,} rows, "
        f"{merged[RAW_COLS['bolus_amount']].notna().sum():,} with bolus, "
        f"{merged[RAW_COLS['basal_rate']].notna().sum():,} with basal"
    )

    # ========== STEP 7: Fill missing values ==========
    # Bolus: no bolus = 0 units
    merged[RAW_COLS["bolus_amount"]] = merged[RAW_COLS["bolus_amount"]].fillna(0)

    # Basal: Keep SPARSE (only rate-change events)
    # The preprocessing pipeline's _rollover_basal_automated() will handle forward-fill
    # and conversion to dose_units during IOB calculation
    merged = merged.sort_values([p_num_col, dt_col])

    # Log NaN analysis
    patients_with_pump = set(basal_agg[p_num_col].unique())
    all_patients = set(merged[p_num_col].unique())
    patients_without_pump = all_patients - patients_with_pump

    logger.info(f"Patients with pump data: {len(patients_with_pump)}")
    logger.info(f"Patients without pump data: {len(patients_without_pump)}")

    total_nan = merged[RAW_COLS["basal_rate"]].isna().sum()
    rows_from_no_pump = merged[merged[p_num_col].isin(patients_without_pump)].shape[0]
    logger.info(
        f"Basal NaN breakdown: {rows_from_no_pump:,} from no-pump patients ({rows_from_no_pump/total_nan*100:.1f}%), "
        f"{total_nan - rows_from_no_pump:,} from leading NaN ({(total_nan - rows_from_no_pump)/total_nan*100:.1f}%)"
    )

    # ========== STEP 8: Final rename to standard schema ==========
    # Convert BG from mg/dL to mmol/L
    merged["bg_mgdL"] = merged[RAW_COLS["cgm_value"]]
    merged[ColumnNames.BG.value] = mg_dl_to_mmol_l(merged, "bg_mgdL")

    # Rename to standard column names
    merged = merged.rename(
        columns={
            RAW_COLS["basal_rate"]: ColumnNames.RATE.value,
            RAW_COLS["bolus_amount"]: ColumnNames.DOSE_UNITS.value,
            RAW_COLS["bolus_type"]: "bolus_type",
            RAW_COLS["period"]: "period",
        }
    )

    # Select and order final columns
    final_columns = [
        dt_col,
        p_num_col,
        "period",
        ColumnNames.BG.value,
        ColumnNames.RATE.value,
        ColumnNames.DOSE_UNITS.value,
        "bolus_type",
    ]

    output_df = merged[final_columns].copy()
    output_df = output_df.set_index(dt_col).sort_index()

    logger.info(
        f"Final output: {output_df.shape[0]:,} rows, {output_df[p_num_col].nunique()} patients"
    )

    return output_df


def process_single_patient(
    patient_df: pd.DataFrame,
    p_num: str,
) -> pd.DataFrame:
    """
    Process a single patient's data through the preprocessing pipeline.

    This applies COB/IOB calculations if the relevant columns exist.

    Args:
        patient_df: Single patient DataFrame (datetime indexed).
        p_num: Patient identifier for logging.

    Returns:
        Processed DataFrame with additional derived columns.
    """
    from src.data.preprocessing.pipeline import preprocessing_pipeline

    logger.info(f"Processing patient {p_num} through preprocessing pipeline...")

    dt_col = ColumnNames.DATETIME.value

    # preprocessing_pipeline expects datetime as index
    if patient_df.index.name != dt_col:
        if dt_col in patient_df.columns:
            patient_df = patient_df.set_index(dt_col)

    processed_df = preprocessing_pipeline(
        p_num,
        patient_df,
        use_aggregation=True,
        basal_delivery_type="automated",
    )

    return processed_df


if __name__ == "__main__":
    # Example usage / testing
    logging.basicConfig(level=logging.INFO)

    cgm_df, basal_df, bolus_df = load_raw_brown_2019_data(DATA_DIR)

    print("\n=== Raw Data Samples ===")
    print("CGM Data:")
    print(cgm_df.head())
    print("\nBasal Data:")
    print(basal_df.head())
    print("\nBolus Data:")
    print(bolus_df.head())

    print("\n=== Running Cleaning Pipeline ===")
    output_df = clean_brown_2019_data(cgm_df, basal_df, bolus_df)

    print("\n=== Cleaned Data ===")
    print(f"Shape: {output_df.shape}")
    print(f"Columns: {list(output_df.columns)}")
    print("\nSample:")
    print(output_df.head(20))

    print("\n=== Validation ===")
    print(
        f"All timestamps on 5-min grid: {((output_df.index.minute % 5 == 0) & (output_df.index.second == 0)).all()}"
    )
    print(f"No negative BG: {(output_df[ColumnNames.BG.value].dropna() >= 0).all()}")
    print(f"No negative bolus: {(output_df[ColumnNames.DOSE_UNITS.value] >= 0).all()}")
    print(
        f"No negative basal: {(output_df[ColumnNames.RATE.value].dropna() >= 0).all()}"
    )
