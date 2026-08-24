# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Shared data-format utilities for AutoGluon-backed model families."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def convert_to_patient_dict(
    flat_df: pd.DataFrame,
    patient_col: str = "p_num",
    time_col: str = "datetime",
) -> Dict[str, pd.DataFrame]:
    """Convert flat patient rows into per-patient DataFrames with DatetimeIndex."""
    patient_dict: Dict[str, pd.DataFrame] = {}

    for pid, group in flat_df.groupby(patient_col):
        pdf = group.copy()

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

        if patient_col in pdf.columns:
            pdf = pdf.drop(columns=[patient_col])

        if isinstance(pid, (int, np.integer, float, np.floating)) and pid == int(pid):
            key = str(int(pid))
        else:
            key = str(pid)
        patient_dict[key] = pdf

    logger.debug("Converted flat DataFrame to %d patient dicts", len(patient_dict))
    return patient_dict


def format_segments_for_autogluon(
    segments: Dict[str, pd.DataFrame],
    target_col: str = "bg_mM",
    covariate_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Any:
    """Convert gap-handled segments into AutoGluon TimeSeriesDataFrame format."""
    from autogluon.timeseries import TimeSeriesDataFrame

    if target_cols and len(target_cols) > 1:
        return _format_segments_multitarget(segments, target_cols)

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
        "Formatted %d segments for AutoGluon: %s",
        len(segments),
        combined.shape,
    )
    return TimeSeriesDataFrame(combined)


def _format_segments_multitarget(
    segments: Dict[str, pd.DataFrame],
    target_cols: List[str],
) -> Any:
    """Stack multiple target columns as independent AutoGluon item IDs."""
    from autogluon.timeseries import TimeSeriesDataFrame

    data_list = []
    for seg_id, seg_df in segments.items():
        for col in target_cols:
            if col not in seg_df.columns:
                logger.warning(
                    "Target column '%s' not in segment %s, skipping",
                    col,
                    seg_id,
                )
                continue

            vals = seg_df[col].ffill().fillna(0)
            df = pd.DataFrame(
                {
                    "item_id": f"{seg_id}__{col}",
                    "timestamp": seg_df.index,
                    "target": vals.values,
                }
            )
            data_list.append(df)

    if not data_list:
        raise ValueError(
            "No valid multi-target segments found. "
            "Check that target_cols columns exist in the data."
        )

    combined = pd.concat(data_list, ignore_index=True)
    combined = combined.set_index(["item_id", "timestamp"])

    logger.debug(
        "Formatted %d multi-target items (%d segments x %d targets): %s",
        len(data_list),
        len(segments),
        len(target_cols),
        combined.shape,
    )
    return TimeSeriesDataFrame(combined)
