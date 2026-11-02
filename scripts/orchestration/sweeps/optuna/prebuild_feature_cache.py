# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Pre-build the MetaboNet per-patient engineered-feature cache.

Engineers every patient's covariate frame (renamed dose channels + IOB/COB
kinetics for the exponential/Hovorka/Wilinska models) once and writes it to the
feature cache, so Optuna workers start instantly and never race to compute it.

Patients are processed one at a time and the engineered frame is discarded after
writing, so peak host-RAM stays small (the earlier full-population load held all
frames at once and OOM-killed workers).

Usage:
    .venvs/autogluon/bin/python \
        scripts/orchestration/sweeps/optuna/prebuild_feature_cache.py \
        --processed-path cache/data/metabonet/processed \
        --cache-dir results/optuna/feature_cache \
        --workers 8
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.workflows.sweeps.optuna_chronos2.data import (
    PATIENT_PARQUET_DIR,
    PATIENT_PARTITION_PREFIX,
    MetabonetData,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prebuild_feature_cache")


def _build_one(
    processed_path: str, cache_dir: str, interval_mins: int, pid: str
) -> str:
    """Engineer + cache a single patient; returns a short status string."""
    data = MetabonetData(
        processed_path,
        interval_mins=interval_mins,
        feature_cache_dir=cache_dir,
    )
    data._load_static_lookup()
    part_dir = (
        Path(processed_path) / PATIENT_PARQUET_DIR / f"{PATIENT_PARTITION_PREFIX}{pid}"
    )
    parquet_parts = sorted(part_dir.glob("*.parquet"))
    if not parquet_parts:
        return f"skip {pid} (no parquet)"
    frame = data._engineer_or_cache(pid, parquet_parts)
    n = len(frame)
    del frame
    return f"ok {pid} ({n} rows)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-path", default="cache/data/metabonet/processed")
    parser.add_argument("--cache-dir", default="results/optuna/feature_cache")
    parser.add_argument("--interval-mins", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Cap patients (debug).")
    args = parser.parse_args()

    parts_root = Path(args.processed_path) / PATIENT_PARQUET_DIR
    pids = [
        p.name[len(PATIENT_PARTITION_PREFIX) :]
        for p in sorted(parts_root.iterdir())
        if p.is_dir() and p.name.startswith(PATIENT_PARTITION_PREFIX)
    ]
    if args.limit:
        pids = pids[: args.limit]
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Pre-building feature cache for %d patients -> %s", len(pids), args.cache_dir
    )

    start = time.time()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _build_one, args.processed_path, args.cache_dir, args.interval_mins, pid
            ): pid
            for pid in pids
        }
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                logger.warning("FAILED %s: %s", pid, exc)
            if (done + failed) % 100 == 0:
                logger.info(
                    "progress %d/%d (%d failed) %.1fs",
                    done + failed,
                    len(pids),
                    failed,
                    time.time() - start,
                )

    logger.info(
        "Done: %d cached, %d failed in %.1fs -> %s",
        done,
        failed,
        time.time() - start,
        args.cache_dir,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
