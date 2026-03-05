# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Gap handling utilities for continuous glucose monitoring time series data.

This module provides functions for detecting and handling temporal gaps in CGM data.
Gaps occur due to sensor outages, sensor changes, or other interruptions. Without
proper handling, sliding window approaches (e.g., TSFM's ForecastDFDataset) will
create training windows that silently span these gaps, corrupting the data.

Assumes input data is already on a regular time grid (e.g., 5-min intervals).
This is the data loader's responsibility — Brown uses
ensure_regular_time_intervals_with_aggregation(), other loaders have equivalent steps.

Strategy (based on GlucoBench literature):
1. Small gaps (≤ threshold): Fill via linear interpolation
2. Large gaps (> threshold): Split into separate continuous segments
3. Short segments (< min length): Discard as unusable

Each resulting segment is a continuous time series with no temporal gaps,
suitable for sliding window approaches.

Functions:
    segment_all_patients: Main entry point for gap handling across all patients
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preprocessing.time_processing import get_most_common_time_interval

logger = logging.getLogger(__name__)

DEFAULT_IMPUTATION_THRESHOLD_MINS = 45
DEFAULT_MIN_SEGMENT_LENGTH = 608
DEFAULT_BG_COL = "bg_mM"
DEFAULT_FALLBACK_INTERVAL_MINS = 5


@dataclass
class GapStats:
    """Statistics from gap handling for a single patient."""

    patient_id: str
    total_rows_original: int = 0
    total_nan_before: int = 0
    total_nan_after_interp: int = 0
    num_gaps_interpolated: int = 0
    num_gaps_segmented: int = 0
    num_segments_created: int = 0
    num_segments_discarded: int = 0
    rows_retained: int = 0
    rows_discarded: int = 0
    detected_interval_mins: int = 0


def segment_all_patients(
    patients_data: dict[str, pd.DataFrame],
    imputation_threshold_mins: int = DEFAULT_IMPUTATION_THRESHOLD_MINS,
    min_segment_length: int = DEFAULT_MIN_SEGMENT_LENGTH,
    bg_col: str = DEFAULT_BG_COL,
    expected_interval_mins: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Segment multi-patient data at large temporal gaps.

    Assumes data is already on a regular time grid (data loader's responsibility).

    Pipeline per patient:
    1. Detect the sampling interval (auto-detect or use expected_interval_mins)
    2. Interpolate small gaps (consecutive NaN runs ≤ imputation_threshold_mins)
    3. Segment at remaining large gaps (> imputation_threshold_mins)
    4. Discard segments shorter than min_segment_length

    Args:
        patients_data: Dict mapping patient_id -> DataFrame with DatetimeIndex.
            Data must be on a regular time grid.
        imputation_threshold_mins: Gaps up to this duration (in minutes) are
            linearly interpolated. Gaps exceeding this trigger segmentation.
            Default 45 (per GlucoBench / CGM imputation literature).
        min_segment_length: Minimum number of rows for a segment to be kept.
            Segments shorter than this are discarded. Default 608
            (context_length=512 + prediction_length=96).
        bg_col: Name of the blood glucose column used for gap detection.
            Default "bg_mM".
        expected_interval_mins: Expected sampling interval in minutes.
            If None, auto-detected per patient via mode of time diffs.

    Returns:
        Dict mapping segment_id -> DataFrame with DatetimeIndex.
        Segment IDs are formatted as "{patient_id}_seg_{n}" where n
        is a zero-based counter per patient.
    """
    if not patients_data:
        logger.warning("Empty input: no patients to process")
        return {}

    logger.info(
        "Starting gap handling for %d patients " "(threshold=%d min, min_length=%d)",
        len(patients_data),
        imputation_threshold_mins,
        min_segment_length,
    )

    all_segments: dict[str, pd.DataFrame] = {}
    all_stats: list[GapStats] = []

    for patient_id, patient_df in patients_data.items():
        segments, stats = _segment_single_patient(
            patient_id=patient_id,
            patient_df=patient_df,
            imputation_threshold_mins=imputation_threshold_mins,
            min_segment_length=min_segment_length,
            bg_col=bg_col,
            expected_interval_mins=expected_interval_mins,
        )
        all_segments.update(segments)
        all_stats.append(stats)

        logger.debug(
            "Patient %s: %d segments created, %d discarded "
            "(%d rows retained, %d discarded)",
            patient_id,
            stats.num_segments_created,
            stats.num_segments_discarded,
            stats.rows_retained,
            stats.rows_discarded,
        )

        if stats.num_segments_created == 0:
            logger.warning(
                "Patient %s: 0 segments created (all data discarded)", patient_id
            )

    # Summary stats
    total_original = sum(s.total_rows_original for s in all_stats)
    total_retained = sum(s.rows_retained for s in all_stats)
    total_discarded_segs = sum(s.num_segments_discarded for s in all_stats)
    patients_with_no_segs = sum(1 for s in all_stats if s.num_segments_created == 0)

    pct_retained = (total_retained / total_original * 100) if total_original > 0 else 0

    logger.info(
        "Gap handling complete: %d patients -> %d segments",
        len(patients_data),
        len(all_segments),
    )
    logger.info(
        "  Rows retained: %d / %d (%.1f%%)",
        total_retained,
        total_original,
        pct_retained,
    )
    logger.info("  Segments discarded (too short): %d", total_discarded_segs)
    if patients_with_no_segs > 0:
        logger.warning(
            "  Patients with 0 segments (fully discarded): %d", patients_with_no_segs
        )

    return all_segments


def _segment_single_patient(
    patient_id: str,
    patient_df: pd.DataFrame,
    imputation_threshold_mins: int,
    min_segment_length: int,
    bg_col: str,
    expected_interval_mins: int | None,
) -> tuple[dict[str, pd.DataFrame], GapStats]:
    """Full gap-handling pipeline for a single patient."""
    stats = GapStats(patient_id=patient_id)

    # Validate input
    if not isinstance(patient_df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Patient {patient_id}: DataFrame must have DatetimeIndex, "
            f"got {type(patient_df.index).__name__}"
        )

    if bg_col not in patient_df.columns:
        raise ValueError(
            f"Patient {patient_id}: missing required column '{bg_col}'. "
            f"Available columns: {list(patient_df.columns)}"
        )

    df = patient_df.sort_index().copy()
    stats.total_rows_original = len(df)

    if len(df) == 0:
        return {}, stats

    # Detect sampling interval (to convert minutes threshold to row count)
    interval_mins = expected_interval_mins or _detect_interval(df)
    stats.detected_interval_mins = interval_mins

    # Count NaN before interpolation
    stats.total_nan_before = int(df[bg_col].isna().sum())

    # Step 1: Interpolate small gaps
    max_gap_rows = imputation_threshold_mins // interval_mins
    nan_runs_before = _find_nan_runs(df[bg_col])
    df = interpolate_small_gaps(df, max_gap_rows, bg_col)

    # Count NaN after interpolation and compute gap stats
    stats.total_nan_after_interp = int(df[bg_col].isna().sum())
    nan_runs_after = _find_nan_runs(df[bg_col])
    stats.num_gaps_interpolated = len(nan_runs_before) - len(nan_runs_after)
    stats.num_gaps_segmented = len(nan_runs_after)

    # Step 2: Segment at remaining gaps
    segments, num_discarded = _segment_at_remaining_gaps(
        df, patient_id, bg_col, min_segment_length
    )

    stats.num_segments_created = len(segments)
    stats.num_segments_discarded = num_discarded
    stats.rows_retained = sum(len(seg) for seg in segments.values())
    stats.rows_discarded = stats.total_rows_original - stats.rows_retained

    return segments, stats


def _detect_interval(df: pd.DataFrame) -> int:
    """
    Detect the most common sampling interval from a DatetimeIndex.

    Uses the existing get_most_common_time_interval utility. Falls back
    to 5 minutes if detection fails.
    """
    if len(df) < 2:
        return DEFAULT_FALLBACK_INTERVAL_MINS

    try:
        interval = get_most_common_time_interval(df)
        if interval > 0:
            return interval
    except (ValueError, IndexError) as exc:
        logger.warning(
            "Failed to detect sampling interval for DataFrame with %d rows; "
            "falling back to %d minutes. Error: %s",
            len(df),
            DEFAULT_FALLBACK_INTERVAL_MINS,
            exc,
        )

    return DEFAULT_FALLBACK_INTERVAL_MINS


def _find_nan_runs(series: pd.Series) -> list[tuple[int, int, int]]:
    """
    Find all runs of consecutive NaN values in a Series.

    Args:
        series: Pandas Series to scan for NaN runs.

    Returns:
        List of (start_idx, end_idx_exclusive, run_length) tuples.
    """
    is_nan = series.isna()
    if not is_nan.any():
        return []

    # Detect transitions: where NaN status changes
    diff = is_nan.ne(is_nan.shift())
    # Assign group IDs
    groups = diff.cumsum()

    runs = []
    for group_id, group in is_nan.groupby(groups):
        if group.iloc[0]:  # This group is NaN
            start = group.index[0]
            # Convert to positional index
            start_pos = series.index.get_loc(start)
            length = len(group)
            runs.append((start_pos, start_pos + length, length))

    return runs


def interpolate_small_gaps(
    df: pd.DataFrame,
    max_gap_rows: int,
    bg_col: str = DEFAULT_BG_COL,
) -> pd.DataFrame:
    """
    Linearly interpolate NaN runs that are <= max_gap_rows long (all-or-nothing).

    Only the blood glucose column is interpolated. Other numeric columns
    (e.g., bolus, food, steps) are left untouched — linear interpolation
    would create fractional values that are physiologically meaningless.

    Only gaps whose entire length fits within the threshold are interpolated.
    Gaps exceeding the threshold are left completely untouched — no partial
    filling from either side.

    Args:
        df: DataFrame (on regular grid) possibly containing NaN runs.
        max_gap_rows: Maximum consecutive NaN count to interpolate.
            Gaps with more NaN than this are left as-is.
        bg_col: Blood glucose column to interpolate. Only this column
            is modified; all other columns are preserved as-is.

    Returns:
        DataFrame with small BG gaps filled by linear interpolation.
    """
    if max_gap_rows <= 0 or bg_col not in df.columns:
        return df

    df = df.copy()

    nan_runs = _find_nan_runs(df[bg_col])
    if not nan_runs:
        return df

    # Mark positions belonging to large gaps — these must stay NaN
    large_gap_mask = np.zeros(len(df), dtype=bool)
    for start, end, length in nan_runs:
        if length > max_gap_rows:
            large_gap_mask[start:end] = True

    # Interpolate all internal NaN (small gaps get filled correctly
    # using their immediate non-NaN neighbors as anchors).
    # limit_area="inside" = only fill NaN that sit between two real values;
    # leading/trailing NaN with no anchor on one side are left untouched.
    interpolated = df[bg_col].interpolate(method="linear", limit_area="inside")

    # Restore NaN at large gap positions
    interpolated.values[large_gap_mask] = np.nan

    df[bg_col] = interpolated

    return df


def _segment_at_remaining_gaps(
    df: pd.DataFrame,
    patient_id: str,
    bg_col: str,
    min_segment_length: int,
) -> tuple[dict[str, pd.DataFrame], int]:
    """
    Split a DataFrame into continuous segments at remaining NaN boundaries.

    After interpolation, remaining NaN values in bg_col indicate gaps that
    exceeded the interpolation threshold. This function splits the data at
    those boundaries.

    Args:
        df: DataFrame with potential NaN gaps in bg_col.
        patient_id: Patient ID for constructing segment keys.
        bg_col: Column to check for NaN gaps.
        min_segment_length: Minimum rows to keep a segment.

    Returns:
        Tuple of (segments dict, count of discarded segments).
    """
    is_nan = df[bg_col].isna()

    if not is_nan.any():
        # No gaps - single segment
        if len(df) >= min_segment_length:
            return {f"{patient_id}_seg_0": df}, 0
        return {}, 1

    # Assign segment IDs: increment at each NaN-to-data transition
    # Non-NaN rows get grouped together between NaN boundaries
    group_ids = is_nan.ne(is_nan.shift()).cumsum()

    segments: dict[str, pd.DataFrame] = {}
    num_discarded = 0
    seg_counter = 0

    for group_id, group_df in df.groupby(group_ids):
        # Skip NaN groups (gap markers)
        if group_df[bg_col].isna().all():
            continue

        # Drop any stray NaN rows at edges
        group_df = group_df.dropna(subset=[bg_col])

        if len(group_df) < min_segment_length:
            num_discarded += 1
            continue

        segment_key = f"{patient_id}_seg_{seg_counter}"
        segments[segment_key] = group_df
        seg_counter += 1

    return segments, num_discarded
