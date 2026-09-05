#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.diabetes_datasets.metabonet.metabonet import (
    PIECEWISE_STATIC_COVARIATE_COLUMNS,
    PIECEWISE_STATIC_COVARIATES_FILE,
    PROCESSED_COMPLETE_MARKER,
    PROCESSED_PATIENT_PARQUET_DIR,
    STATIC_COVARIATE_COLUMNS,
    STATIC_COVARIATES_FILE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "cache/data/metabonet/processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Metabonet processed artifacts for global/piecewise split.",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help=f"Metabonet processed cache path (default: {DEFAULT_PROCESSED_PATH})",
    )
    parser.add_argument(
        "--sample-partition",
        type=str,
        default=None,
        help=(
            "Optional specific patient partition folder name to validate timeseries columns, "
            "for example: patient_id=1_azt1d-16e20824a5"
        ),
    )
    return parser.parse_args()


def _load_patient_partition_dirs(parts_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in parts_root.iterdir()
            if path.is_dir() and path.name.startswith("patient_id=")
        ],
        key=lambda path: path.name,
    )


def _ensure_artifacts_exist(processed_path: Path) -> tuple[Path, Path, Path, Path]:
    marker_path = processed_path / PROCESSED_COMPLETE_MARKER
    static_path = processed_path / STATIC_COVARIATES_FILE
    piecewise_path = processed_path / PIECEWISE_STATIC_COVARIATES_FILE
    parts_root = processed_path / PROCESSED_PATIENT_PARQUET_DIR

    if not marker_path.exists():
        raise FileNotFoundError(f"Missing marker file: {marker_path}")
    if not static_path.exists():
        raise FileNotFoundError(f"Missing static covariates file: {static_path}")
    if not piecewise_path.exists():
        raise FileNotFoundError(f"Missing piecewise covariates file: {piecewise_path}")
    if not parts_root.exists():
        raise FileNotFoundError(f"Missing patient partition root: {parts_root}")

    return marker_path, static_path, piecewise_path, parts_root


def _validate_patient_partition_count(
    marker_path: Path, parts_root: Path
) -> list[Path]:
    expected_patients = int(marker_path.read_text(encoding="utf-8").strip())
    patient_dirs = _load_patient_partition_dirs(parts_root)
    if len(patient_dirs) != expected_patients:
        raise ValueError(
            "Patient partition count mismatch: "
            f"dirs={len(patient_dirs)} marker={expected_patients}"
        )
    print(f"PASS patient partition count: {len(patient_dirs)}")
    return patient_dirs


def _validate_global_static_table(static_path: Path) -> pd.DataFrame:
    static_df = pd.read_csv(static_path, low_memory=False)
    if "patient_id" not in static_df.columns:
        raise ValueError("static_covariates.csv is missing patient_id column.")
    if static_df["patient_id"].astype(str).nunique() != len(static_df):
        raise ValueError("Duplicate patient_id rows found in static_covariates.csv.")
    print(f"PASS global static rows: {len(static_df)}")
    return static_df


def _validate_piecewise_table(piecewise_path: Path) -> pd.DataFrame:
    piecewise_df = pd.read_parquet(piecewise_path)
    required_cols = {
        "patient_id",
        "covariate",
        "start_datetime",
        "end_datetime",
        "value",
        "value_type",
    }
    missing_cols = required_cols - set(piecewise_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required piecewise columns: {sorted(missing_cols)}")
    print(f"PASS piecewise rows: {len(piecewise_df)}")
    return piecewise_df


def _validate_piecewise_overlap(piecewise_df: pd.DataFrame) -> None:
    if piecewise_df.empty:
        print("PASS piecewise overlap check (no piecewise rows)")
        return

    segments = piecewise_df.copy()
    segments["start_datetime"] = pd.to_datetime(
        segments["start_datetime"], errors="raise"
    )
    segments["end_datetime"] = pd.to_datetime(segments["end_datetime"], errors="coerce")
    segments = segments.sort_values(["patient_id", "covariate", "start_datetime"])
    segments["prev_end"] = segments.groupby(["patient_id", "covariate"])[
        "end_datetime"
    ].shift(1)
    overlap_rows = segments[
        (segments["prev_end"].notna())
        & (segments["prev_end"] > segments["start_datetime"])
    ]
    if not overlap_rows.empty:
        raise ValueError(
            f"Found overlapping piecewise segments: {len(overlap_rows)} rows"
        )
    print("PASS piecewise overlap check")


def _validate_timeseries_column_split(
    *,
    patient_dirs: list[Path],
    sample_partition: str | None,
) -> None:
    if not patient_dirs:
        raise ValueError("No patient partitions found to validate.")

    if sample_partition is None:
        chosen_partition = patient_dirs[0]
    else:
        matched = [path for path in patient_dirs if path.name == sample_partition]
        if not matched:
            raise ValueError(
                f"Requested sample partition {sample_partition!r} not found in patient partitions."
            )
        chosen_partition = matched[0]

    parquet_parts = sorted(chosen_partition.glob("*.parquet"))
    if not parquet_parts:
        raise ValueError(
            f"No parquet parts found in sample partition {chosen_partition}."
        )

    sample_df = pd.read_parquet(parquet_parts[0])
    unexpected_columns = (
        set(STATIC_COVARIATE_COLUMNS) | set(PIECEWISE_STATIC_COVARIATE_COLUMNS)
    ).intersection(sample_df.columns)
    if unexpected_columns:
        raise ValueError(
            "Unexpected static/piecewise covariate columns remain in timeseries: "
            f"{sorted(unexpected_columns)}"
        )
    print(
        f"PASS timeseries split check on: {chosen_partition.name}/{parquet_parts[0].name}"
    )


def main() -> None:
    args = parse_args()
    marker_path, static_path, piecewise_path, parts_root = _ensure_artifacts_exist(
        args.processed_path
    )

    patient_dirs = _validate_patient_partition_count(marker_path, parts_root)
    _validate_global_static_table(static_path)
    piecewise_df = _validate_piecewise_table(piecewise_path)
    _validate_piecewise_overlap(piecewise_df)
    _validate_timeseries_column_split(
        patient_dirs=patient_dirs,
        sample_partition=args.sample_partition,
    )

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
