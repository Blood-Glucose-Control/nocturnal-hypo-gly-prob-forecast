# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..dataset_base import ProcessedPatientDataFrames

PATIENT_ID_ALIASES = ("patient_id", "id")
DATETIME_ALIASES = ("datetime", "date", "timestamp")
SEGMENT_ID_ALIASES = ("row_id", "segment_id", "window_id")
BG_MGDL_ALIASES = ("bg_mg_dl", "CGM", "cgm")


def normalize_metabonet_dataframe(
    df: pd.DataFrame,
    *,
    split_name: str,
    require_bg: bool,
) -> pd.DataFrame:
    """Normalize a raw Metabonet split into canonical processed shape."""
    if df.empty:
        raise ValueError(f"Metabonet {split_name} split is empty.")

    normalized = df.copy()
    normalized = _ensure_datetime_column(normalized)
    normalized = _rename_first_present_column(
        normalized,
        aliases=PATIENT_ID_ALIASES,
        canonical_name="patient_id",
        split_name=split_name,
    )
    normalized = _rename_first_present_column(
        normalized,
        aliases=DATETIME_ALIASES,
        canonical_name="datetime",
        split_name=split_name,
    )

    if "bg_mM" not in normalized.columns:
        bg_mgdl_column = _resolve_optional_column(normalized.columns, BG_MGDL_ALIASES)
        if bg_mgdl_column is not None:
            normalized["bg_mM"] = (
                pd.to_numeric(normalized[bg_mgdl_column], errors="coerce") / 18.0
            )

    if require_bg and "bg_mM" not in normalized.columns:
        raise ValueError(
            "Metabonet train split must contain 'bg_mM' (or 'bg_mg_dl' for conversion)."
        )

    if "source_file" not in normalized.columns:
        raise ValueError(
            f"Metabonet {split_name} split must contain 'source_file' for unique patient identifiers."
        )
    if normalized["source_file"].isna().any():
        raise ValueError(
            f"Metabonet {split_name} split contains null source_file values."
        )

    source_values = normalized["source_file"].astype(str)
    source_key_map = {
        source_file: _build_source_file_key(source_file)
        for source_file in source_values.unique()
    }
    normalized["patient_id"] = (
        normalized["patient_id"].astype(str).str.strip()
        + "_"
        + source_values.map(source_key_map)
    )

    datetime_index = pd.to_datetime(
        normalized["datetime"],
        format="mixed",
        errors="raise",
    )
    normalized = normalized.drop(columns=["datetime"])
    normalized.index = pd.DatetimeIndex(datetime_index, name="datetime")
    return normalized.sort_index()


def split_by_patient_id(df: pd.DataFrame) -> ProcessedPatientDataFrames:
    """Split a canonical dataframe into a patient_id -> DataFrame dictionary."""
    return {
        str(patient_id): patient_df.copy()
        for patient_id, patient_df in df.groupby("patient_id", sort=False)
    }


def build_nested_test_data(
    test_df: pd.DataFrame,
    *,
    segment_column: str | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Create {patient_id: {segment_id: DataFrame}} structure for contest test data."""
    resolved_segment_column = segment_column or _resolve_optional_column(
        test_df.columns,
        SEGMENT_ID_ALIASES,
    )
    if resolved_segment_column is None:
        return {
            str(patient_id): {"full": patient_df.copy()}
            for patient_id, patient_df in test_df.groupby("patient_id", sort=False)
        }

    nested: dict[str, dict[str, pd.DataFrame]] = {}
    for patient_id, patient_df in test_df.groupby("patient_id", sort=False):
        nested[str(patient_id)] = {
            str(segment_id): segment_df.copy()
            for segment_id, segment_df in patient_df.groupby(
                resolved_segment_column, sort=False
            )
        }
    return nested


def _rename_first_present_column(
    df: pd.DataFrame,
    *,
    aliases: tuple[str, ...],
    canonical_name: str,
    split_name: str,
) -> pd.DataFrame:
    if canonical_name in df.columns:
        return df

    alias = _resolve_optional_column(df.columns, aliases)
    if alias is None:
        raise ValueError(
            f"Metabonet {split_name} split is missing required column aliases {aliases}."
        )
    if alias == canonical_name:
        return df
    return df.rename(columns={alias: canonical_name})


def _resolve_optional_column(
    columns: Iterable[str],
    candidate_names: tuple[str, ...],
) -> str | None:
    available = set(columns)
    for name in candidate_names:
        if name in available:
            return name
    return None


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        index_column_name = df.index.name or "index"
        return df.reset_index().rename(columns={index_column_name: "datetime"})
    return df


def _build_source_file_key(source_file: str) -> str:
    normalized_source = source_file.strip()
    source_name = Path(normalized_source).name
    source_stem = Path(source_name).stem or source_name
    readable_source = re.sub(r"[^0-9A-Za-z]+", "_", source_stem).strip("_").lower()
    if not readable_source:
        readable_source = "source"
    source_digest = hashlib.sha1(
        normalized_source.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:10]
    return f"{readable_source}-{source_digest}"
