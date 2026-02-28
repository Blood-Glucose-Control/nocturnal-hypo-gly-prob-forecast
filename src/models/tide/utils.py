# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""
TiDE / AutoGluon data-format utilities.

These utilities convert between the project's data formats (flat DataFrames,
patient dicts, gap-handled segments) and AutoGluon's TimeSeriesDataFrame format.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def convert_to_patient_dict(
    flat_df: pd.DataFrame,
    patient_col: str = "p_num",
    time_col: str = "datetime",
) -> Dict[str, pd.DataFrame]:
    """Convert a flat DataFrame to per-patient dict with DatetimeIndex.

    The registry returns flat DataFrames (all patients concatenated).
    Downstream functions (segment_all_patients, build_midnight_episodes)
    expect Dict[patient_id, DataFrame] with DatetimeIndex.

    Args:
        flat_df: DataFrame with patient_col identifying patients and
            time_col for timestamps (or DatetimeIndex).
        patient_col: Column name for patient identifiers.
        time_col: Column name for timestamps.

    Returns:
        Dict mapping patient_id (str) to DataFrame with DatetimeIndex.
    """
    patient_dict = {}

    for pid, group in flat_df.groupby(patient_col):
        pdf = group.copy()

        # Set DatetimeIndex from column or use existing index
        if time_col in pdf.columns:
            pdf[time_col] = pd.to_datetime(pdf[time_col])
            pdf = pdf.set_index(time_col).sort_index()
        elif isinstance(pdf.index, pd.DatetimeIndex):
            pdf = pdf.sort_index()
        else:
            raise ValueError(
                f"Patient {pid!r}: expected '{time_col}' column or "
                f"DatetimeIndex, got {type(pdf.index).__name__}"
            )

        # Drop patient column (redundant — it's the dict key)
        if patient_col in pdf.columns:
            pdf = pdf.drop(columns=[patient_col])

        # Convert float IDs (e.g., 1.0) to clean strings ("1")
        key = str(int(pid)) if isinstance(pid, float) and pid == int(pid) else str(pid)
        patient_dict[key] = pdf

    logger.debug("Converted flat DataFrame to %d patient dicts", len(patient_dict))
    return patient_dict


def format_segments_for_autogluon(
    segments: Dict[str, pd.DataFrame],
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
) -> Any:
    """Convert gap-handled segments to AutoGluon TimeSeriesDataFrame.

    Each segment becomes one item_id with variable length. AutoGluon
    creates sliding windows internally during training, seeing covariate
    values within each segment.

    Args:
        segments: Dict mapping segment_id -> DataFrame with DatetimeIndex.
            Output from segment_all_patients().
        target_col: Column name for the target variable (e.g., "bg_mM").
        covariate_cols: Column names for covariates (e.g., ["iob"] or
            ["iob", "cob"]). Missing columns are filled with 0.

    Returns:
        TimeSeriesDataFrame with columns ["target", <covariates>],
        indexed by (item_id, timestamp).
    """
    from autogluon.timeseries import TimeSeriesDataFrame

    if covariate_cols is None:
        covariate_cols = ["iob"]

    data_list = []

    for seg_id, seg_df in segments.items():
        df = seg_df[[target_col]].copy()
        df = df.rename(columns={target_col: "target"})

        for cov_col in covariate_cols:
            has_cov = cov_col in seg_df.columns and seg_df[cov_col].notna().any()
            if has_cov:
                df[cov_col] = seg_df[cov_col].ffill().fillna(0)
            else:
                df[cov_col] = 0.0

        df["item_id"] = seg_id
        df["timestamp"] = df.index
        out_cols = ["item_id", "timestamp", "target"] + covariate_cols
        data_list.append(df[out_cols])

    if not data_list:
        raise ValueError(
            "No segments to format — gap handling discarded all data. "
            "Check imputation_threshold_mins and min_segment_length."
        )

    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.set_index(["item_id", "timestamp"])

    logger.debug(
        "Formatted %d segments for AutoGluon: %s", len(segments), combined.shape
    )
    return TimeSeriesDataFrame(combined)
