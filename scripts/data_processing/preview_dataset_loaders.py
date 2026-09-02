#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""Preview processed output from one or more dataset loaders.

Usage examples:
    python scripts/data_processing/preview_dataset_loaders.py
    python scripts/data_processing/preview_dataset_loaders.py --datasets lynch_2022 brown_2019
    python scripts/data_processing/preview_dataset_loaders.py --head 10 --patients 2
    python scripts/data_processing/preview_dataset_loaders.py --no-cache
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable

import pandas as pd

from src.data.diabetes_datasets.data_loader import get_loader

logger = logging.getLogger(__name__)

CORE_DATASETS = (
    "aleppo_2017",
    "brown_2019",
    "lynch_2022",
    "tamborlane_2008",
)
ALL_FACTORY_DATASETS = CORE_DATASETS


def _resolve_patient_ids(
    processed_data: dict[str, pd.DataFrame], patients_to_show: int
) -> list[str]:
    """Return deterministic patient IDs for preview output."""
    return sorted(processed_data.keys())[:patients_to_show]


def _print_patient_preview(
    patient_id: str, patient_df: pd.DataFrame, head_rows: int
) -> None:
    """Print concise metadata and head rows for one patient DataFrame."""
    logger.info("  Patient: %s", patient_id)
    logger.info("  Shape: %s", patient_df.shape)
    logger.info("  Columns: %s", list(patient_df.columns))

    if isinstance(patient_df.index, pd.DatetimeIndex) and not patient_df.empty:
        logger.info(
            "  Datetime range: %s -> %s",
            patient_df.index.min(),
            patient_df.index.max(),
        )
    elif "datetime" in patient_df.columns and not patient_df.empty:
        dt_series = pd.to_datetime(patient_df["datetime"], errors="coerce").dropna()
        if not dt_series.empty:
            logger.info("  Datetime range: %s -> %s", dt_series.min(), dt_series.max())

    print(patient_df.head(head_rows).to_string())
    print()


def preview_loader(
    dataset_name: str,
    use_cached: bool,
    parallel: bool,
    max_workers: int,
    head_rows: int,
    patients_to_show: int,
) -> bool:
    """Instantiate one loader and print processed-data previews."""
    logger.info("=" * 80)
    logger.info("Dataset: %s", dataset_name)
    logger.info(
        "Options: use_cached=%s, parallel=%s, max_workers=%d",
        use_cached,
        parallel,
        max_workers,
    )
    logger.info("=" * 80)

    try:
        loader = get_loader(
            dataset_name,
            use_cached=use_cached,
            parallel=parallel,
            max_workers=max_workers,
        )
        processed_data = loader.processed_data
        if not isinstance(processed_data, dict):
            raise TypeError(
                f"Expected dict[str, DataFrame] processed_data, got {type(processed_data).__name__}"
            )
        if not processed_data:
            raise ValueError("processed_data is empty.")

        logger.info("Loaded %d patients", len(processed_data))

        patient_ids = _resolve_patient_ids(processed_data, patients_to_show)
        for patient_id in patient_ids:
            patient_df = processed_data[patient_id]
            _print_patient_preview(patient_id, patient_df, head_rows)

        return True
    except Exception:
        logger.exception("Failed to preview dataset loader: %s", dataset_name)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview processed outputs from diabetes dataset loaders."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(CORE_DATASETS),
        choices=list(ALL_FACTORY_DATASETS),
        help="Dataset loaders to run (default: core four).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Number of rows to print from each selected patient DataFrame.",
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=3,
        help="Number of patients to preview per dataset (deterministic order).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache usage and force processing from raw data.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel preprocessing when processing from raw data.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=14,
        help="Worker count for parallel preprocessing paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.head <= 0:
        raise ValueError("--head must be a positive integer.")
    if args.patients <= 0:
        raise ValueError("--patients must be a positive integer.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be a positive integer.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    successful: list[str] = []
    failed: list[str] = []

    for dataset_name in _ordered_unique(args.datasets):
        ok = preview_loader(
            dataset_name=dataset_name,
            use_cached=not args.no_cache,
            parallel=args.parallel,
            max_workers=args.max_workers,
            head_rows=args.head,
            patients_to_show=args.patients,
        )
        if ok:
            successful.append(dataset_name)
        else:
            failed.append(dataset_name)

    logger.info("-" * 80)
    logger.info("Completed loader preview run")
    logger.info("Succeeded: %s", successful if successful else "none")
    logger.info("Failed: %s", failed if failed else "none")


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


if __name__ == "__main__":
    main()
