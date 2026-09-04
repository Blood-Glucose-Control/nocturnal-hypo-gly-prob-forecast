#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Load Metabonet and open an interactive terminal session with the loader object.

Examples:
    python scripts/scratch/metabonet_loader_example.py
    python scripts/scratch/metabonet_loader_example.py --no-cache --no-repl
    python -i scripts/scratch/metabonet_loader_example.py
"""

from __future__ import annotations

import argparse
import code
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Allow direct script execution from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.diabetes_datasets.metabonet.metabonet import (  # noqa: E402
    PATIENT_PARTITION_PREFIX,
    PROCESSED_PATIENT_PARQUET_DIR,
    MetabonetDataLoader,
)

logger = logging.getLogger(__name__)
DEFAULT_MAX_WORKER_UTILIZATION = 0.8

# Exposed global for `python -i scripts/scratch/metabonet_loader_example.py`
metabonet_loader: MetabonetDataLoader | None = None


def get_default_max_workers() -> int:
    detected_cores = os.cpu_count() or 1
    return max(1, int(detected_cores * DEFAULT_MAX_WORKER_UTILIZATION))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instantiate MetabonetDataLoader and optionally drop into a REPL.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cached loading and force processing from raw files.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing when rebuilding from raw files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=get_default_max_workers(),
        help=(
            "Worker count used when parallel processing is enabled "
            f"(default: {int(DEFAULT_MAX_WORKER_UTILIZATION * 100)}%% of detected cores)."
        ),
    )
    parser.add_argument(
        "--keep-columns",
        type=str,
        default=None,
        help="Comma-separated list of columns to keep (required columns are auto-kept).",
    )
    parser.add_argument(
        "--load-all",
        action="store_true",
        help="Load all processed patient data into memory after processing/loading cache.",
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Do not open an interactive shell after loading.",
    )
    return parser.parse_args()


def parse_keep_columns(raw_keep_columns: str | None) -> list[str] | None:
    if raw_keep_columns is None:
        return None
    keep_columns = [
        value.strip() for value in raw_keep_columns.split(",") if value.strip()
    ]
    if not keep_columns:
        raise ValueError(
            "--keep-columns was provided but no valid columns were parsed."
        )
    return keep_columns


def build_loader(args: argparse.Namespace) -> MetabonetDataLoader:
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be a positive integer.")

    return MetabonetDataLoader(
        keep_columns=parse_keep_columns(args.keep_columns),
        use_cached=not args.no_cache,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        load_all=args.load_all,
    )


def _log_loader_summary(loader: MetabonetDataLoader) -> None:
    processed_data = loader.processed_data or {}
    processed_path = loader.cache_manager.get_absolute_path_by_type(
        loader.dataset_name, "processed"
    )
    patient_cache_path = processed_path / PROCESSED_PATIENT_PARQUET_DIR
    if patient_cache_path.exists():
        cached_patient_files = [
            partition_path
            for partition_path in patient_cache_path.iterdir()
            if partition_path.is_dir()
            and partition_path.name.startswith(PATIENT_PARTITION_PREFIX)
        ]
    else:
        cached_patient_files = []
    logger.info("Loaded MetabonetDataLoader")
    logger.info("Detected CPU cores: %d", os.cpu_count() or 1)
    logger.info("Configured max_workers: %d", loader.max_workers)
    logger.info("Patients in processed_data: %d", len(processed_data))
    logger.info(
        "Processed patient parquet partitions in cache: %d", len(cached_patient_files)
    )
    logger.info("Patients in test_data: %d", len(loader.test_data))

    if processed_data:
        first_patient_id = sorted(processed_data.keys())[0]
        first_patient_df = processed_data[first_patient_id]
        logger.info("First patient id: %s", first_patient_id)
        logger.info("First patient shape: %s", first_patient_df.shape)
        logger.info("First patient columns: %s", list(first_patient_df.columns))
    else:
        logger.info(
            "processed_data is empty because load_all=False. Set --load-all to materialize in memory."
        )


def launch_repl(loader: MetabonetDataLoader) -> None:
    banner = (
        "Metabonet loader shell is ready.\n"
        "Available objects:\n"
        "  - metabonet_loader (same as loader)\n"
        "  - loader\n"
        "  - pd (pandas)\n"
        "Example: loader.processed_data.keys() or loader.load_test_data()"
    )
    code.interact(
        banner=banner,
        local={
            "metabonet_loader": loader,
            "loader": loader,
            "pd": pd,
        },
    )


def main() -> MetabonetDataLoader:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    loader = build_loader(args)
    _log_loader_summary(loader)

    global metabonet_loader
    metabonet_loader = loader

    if not args.no_repl:
        launch_repl(loader)

    return loader


if __name__ == "__main__":
    main()
