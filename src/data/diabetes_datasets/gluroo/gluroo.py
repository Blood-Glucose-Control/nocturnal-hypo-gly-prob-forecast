# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
⚠️  IMPORTANT NOTICE - INTERNAL USE ONLY ⚠️
This dataset is for INTERNAL USE ONLY and will NOT be released to the public.
"""

import logging
import queue
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import IterableDataset, load_dataset
from sqlalchemy import bindparam, create_engine, text

from src.data.cache_manager import get_cache_manager
from src.data.diabetes_datasets.dataset_base import DatasetBase
from src.data.diabetes_datasets.gluroo.data_cleaner import data_translation
from src.data.dataset_configs import get_dataset_config
from src.data.models import ColumnNames
from src.data.preprocessing.pipeline import preprocessing_pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

PROCESSING_CHECKPOINT_FILENAME = "processing_checkpoint.json"
PROCESSED_PATIENTS_LOG_FILENAME = "processed_patients.csv"
BATCH_TIMINGS_FILENAME = "batch_timings.csv"
SKIPPED_PATIENT_IDS_FILENAME = "skipped_patient_ids.csv"

# Maximum number of individual patient DataFrames held in the queue at once.
# This will need to be tuned based on the available RAM, the number of workers, and the time it takes to process a patient.
_QUEUE_MAXSIZE = 80

# Sentinel value placed on the queue by the producer when all patients are enqueued.
_QUEUE_SENTINEL = None


class Gluroo2026DataLoader(DatasetBase):
    """
    Loader for Gluroo diabetes dataset with preprocessing and feature engineering.

    Processing uses a producer-consumer pipeline:
    - Producer thread: issues batched SQL queries (patients_per_batch at a time),
      splits each batch by patient, and enqueues individual (p_num, DataFrame) items.
    - Consumer pool (ProcessPoolExecutor): workers pop one patient at a time, run
      the full preprocessing pipeline, write staging/gluroo_N.parquet, and append
      the p_num to processed_patients.csv atomically.

    Storage layout:
        parquet/staging/   — one file per patient; permanent source of truth
        parquet/merged/    — ~250 consolidated files for training; rebuilt on demand
                             via merge_staging_to_merged()

    Resumability: processed_patients.csv tracks completed patients.  On restart the
    producer skips all already-done patients so no work is ever repeated.

    Note:
    - gid is the base64 encoded group key (primary key in TimescaleDB groups table).
    - p_num is the internal string identifier (e.g., gluroo_1) ordered by gid.
    """

    def __init__(
        self,
        keep_columns: list[str] | None = None,
        use_cached: bool = True,
        max_workers: int = 10,
        patients_per_batch: int = 100,
        patients_per_file: int = 400,
        min_date_span_days: int = 30,
        load_all: bool = False,
        max_batches_per_run: int | None = None,
    ):
        """
        Initialize the Gluroo data loader.

        To force a full reprocess from scratch, delete these files in the cache dir:
            processed_patients.csv, processing_checkpoint.json,
            batch_timings.csv, skipped_patient_ids.csv, parquet/

        Args:
            keep_columns: Columns to retain from the raw data. None keeps all.
            use_cached: If True, skip processing and use existing parquet/merged/.
            max_workers: Worker processes for parallel patient preprocessing.
                1 worker is reserved for the producer thread.
            patients_per_batch: Patients per DB query (I/O efficiency knob).
            patients_per_file: Patients per merged Parquet file (~400 → ~250 files
                for 100k patients).  Must be constant across runs on the same dataset.
            min_date_span_days: Minimum date span for a patient to be included.
            load_all: Load all processed data into processed_data dict (testing only).
            max_batches_per_run: If set, process at most this many batches this run.
                i,e max_batches_per_run * patients_per_batch patients this run.
        """
        self.keep_columns = keep_columns
        self.cache_manager = get_cache_manager()
        self.dataset_config = get_dataset_config(self.dataset_name)
        self.use_cached = use_cached
        self.max_workers = max_workers
        self.patients_per_batch = patients_per_batch
        self.max_batches_per_run = max_batches_per_run
        self.patients_per_file = patients_per_file
        self.min_date_span_days = min_date_span_days
        self.load_all = load_all
        self.db_connection_string = (
            "postgresql://postgres:password@127.0.0.1:5432/gluroo_datasets"
        )

        self.processed_data: dict[str, pd.DataFrame] = {}
        self.train_data: dict[str, pd.DataFrame] = {}
        self.validation_data: dict[str, pd.DataFrame] | None = None

        logger.info(f"Initializing GlurooDataLoader with use_cached={use_cached}")
        self.load_data()

    @property
    def dataset_name(self):
        return "gluroo"

    @property
    def description(self):
        return """
        Gluroo diabetes dataset loaded from TimescaleDB.
        Contains continuous glucose monitoring data, meal announcements, and insulin dosing
        information for multiple patients.
        This is for internal use only. The dataset will NOT be released to the public as they are collected within Gluroo and therefore confidential.
        """

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def load_data(self):
        """
        Load processed data from cache or process raw data from database.

        If use_cached is True and merged Parquet files exist, skips processing.
        Otherwise, runs the producer-consumer processing pipeline.

        Note: processed_data is empty by default to avoid loading all data into
        memory.  Use load_all=True (testing only) or get_hf_streaming_dataset()
        for streaming access.
        """
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        merged_path = processed_path / "parquet" / "merged"

        need_to_process = True
        if self.use_cached:
            if merged_path.exists() and any(merged_path.glob("*.parquet")):
                need_to_process = False
                logger.info("Found cached merged Parquet data; skipping processing.")
                if self.load_all:
                    logger.warning(
                        "WARNING: Loading all data into processed_data for testing."
                    )
                    self._load_all_into_processed_data()
            else:
                logger.warning(
                    "use_cached=True but no merged Parquet found. Will process from source."
                )

        if need_to_process:
            logger.warning(
                "This is a very long running operation. Make sure to run it as a job."
            )
            self._process_and_cache_data_pipeline()
            logger.info(
                "Data processing completed. "
                "Call merge_staging_to_merged() to build training files, "
                "then reload with use_cached=True."
            )
            if self.load_all:
                logger.warning(
                    "WARNING: Loading all data into processed_data for testing."
                )
                # TODO: This is bugged. We need to fix it.
                self._load_all_into_processed_data()

    def merge_staging_to_merged(self) -> None:
        """
        This is meant to be run after the processing pipeline has finished (which might just take forever).
        Consolidate per-patient staging files into grouped merged Parquet files.
        Check ARCHITECTURE.md for more details.

        Reads processed_patients.log for the exact list of completed patients (no
        directory glob over 100k staging files).  Groups them by
            file_id = p_num_numeric // patients_per_file
        and writes parquet/merged/file_NNNNNN.parquet.

        Idempotent: re-running overwrites merged files with up-to-date content.
        Never touches staging files.

        patients_per_file must be the same value used during processing; it is
        read from processing_checkpoint.json if available.
        """
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        staging_path = processed_path / "parquet" / "staging"
        merged_path = processed_path / "parquet" / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)

        patients_per_file: int = (
            self._get_saved_patients_per_file(processed_path) or self.patients_per_file
        )

        processed_p_nums = self._load_processed_set(processed_path)
        if not processed_p_nums:
            logger.warning("processed_patients.csv is empty; nothing to merge.")
            return

        logger.info(
            f"Merging {len(processed_p_nums)} staged patients into "
            f"merged/ (patients_per_file={patients_per_file})..."
        )

        # Group p_nums by target merged file without globbing staging/
        file_groups: dict[int, list[str]] = defaultdict(list)
        for p_num in processed_p_nums:
            file_id = self._p_num_to_int(p_num) // patients_per_file
            file_groups[file_id].append(p_num)

        total_files = len(file_groups)
        for idx, (file_id, p_nums) in enumerate(sorted(file_groups.items()), 1):
            dfs = []
            for p_num in sorted(p_nums, key=self._p_num_to_int):
                staging_file = staging_path / f"{p_num}.parquet"
                if not staging_file.exists():
                    logger.warning(
                        f"Staging file missing for {p_num}; skipping in merge."
                    )
                    continue
                dfs.append(pd.read_parquet(staging_file))

            if not dfs:
                continue

            merged_df = pd.concat(dfs, ignore_index=True)
            out_path = merged_path / f"file_{file_id:06d}.parquet"
            merged_df.to_parquet(
                out_path, engine="pyarrow", compression="snappy", index=False
            )
            del merged_df
            del dfs
            logger.info(
                f"Merged file {idx}/{total_files}: {out_path.name} "
                f"({len(p_nums)} patients)"
            )

        logger.info(f"merge_staging_to_merged() complete: {total_files} files written.")

    def get_hf_streaming_dataset(
        self,
        columns: list[str] | None = None,
        patient_ids: Iterable[str] | None = None,
        batch_size: int | None = None,
        validate_non_empty: bool = True,
    ):
        """
        Return a HuggingFace IterableDataset streaming from parquet/merged/.

        Args:
            columns: Optional column subset (passed to load_dataset).
            patient_ids: Optional p_num filter.
            batch_size: Optional on-the-fly batching.
            validate_non_empty: If True, peek one element to fail fast on empty streams.

        Returns:
            datasets.IterableDataset configured for streaming.
        """
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        merged_path = processed_path / "parquet" / "merged"
        if not merged_path.exists() or not any(merged_path.glob("*.parquet")):
            raise FileNotFoundError(
                f"No merged Parquet files found at {merged_path}. "
                "Run merge_staging_to_merged() first."
            )

        data_files = str(merged_path / "*.parquet")
        load_kwargs: dict = {
            "path": "parquet",
            "data_files": data_files,
            "split": "train",
            "streaming": True,
        }
        if columns:
            load_kwargs["columns"] = columns

        dataset = load_dataset(**load_kwargs)
        if isinstance(dataset, dict):
            dataset = dataset["train"]

        if patient_ids:
            patient_ids_set = {str(pid) for pid in patient_ids}
            dataset = dataset.filter(
                lambda example: (
                    str(example.get(ColumnNames.P_NUM.value, "")) in patient_ids_set
                )
            )

        if batch_size:
            dataset = dataset.batch(batch_size=batch_size)

        if validate_non_empty:
            try:
                first_item = next(iter(dataset))
                dataset = IterableDataset.from_generator(
                    lambda: chain([first_item], dataset)
                )
            except StopIteration as e:
                raise ValueError(
                    f"Hugging Face streaming dataset is empty. "
                    f"Merged path: {merged_path}. "
                    f"Filters -> columns={columns}, p_num filter={patient_ids}."
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Failed to initialize streaming dataset from {merged_path}: {e}"
                ) from e

        return dataset

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _process_and_cache_data_pipeline(self) -> None:
        """
        Producer-consumer pipeline: overlaps DB I/O with CPU preprocessing.

        Producer thread:
            - Queries the DB in batches of patients_per_batch.
            - Splits each batch by patient and enqueues (p_num, df) items.
            - Blocks when the patient queue is full (back-pressure).

        Consumer ProcessPoolExecutor (max_workers processes):
            - Each worker pops one (p_num, df) item.
            - Runs data_translation + preprocessing_pipeline.
            - Writes parquet/staging/<p_num>.parquet.
            - Atomically appends p_num to processed_patients.csv.

        Resumability:
            - On startup, processed_patients.csv is read to build the set of
              already-done patients.  The producer skips them entirely.
            - A crash mid-run is safe: done patients stay checkpointed; failed
              ones are retried on the next run.
        """
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.mkdir(parents=True, exist_ok=True)
        staging_path = processed_path / "parquet" / "staging"
        staging_path.mkdir(parents=True, exist_ok=True)

        # Validate patients_per_file consistency across runs.
        saved_ppf = self._get_saved_patients_per_file(processed_path)
        if saved_ppf is not None and saved_ppf != self.patients_per_file:
            raise ValueError(
                f"patients_per_file mismatch: checkpoint has {saved_ppf}, "
                f"loader has {self.patients_per_file}. "
                "Use the same value or delete the checkpoint to start fresh."
            )

        # Load run metadata checkpoint.
        checkpoint = self._load_run_metadata(processed_path)
        current_run_number = checkpoint.get("run_number", 0) + 1
        run_start_date = datetime.now(timezone.utc).isoformat()
        first_run_date = checkpoint.get("first_run_date") or run_start_date

        # Determine which patients still need processing.
        full_gids_p_nums = self._get_all_patient_ids()
        total_patients = len(full_gids_p_nums)

        already_done = self._load_processed_set(processed_path)
        remaining_gids_p_nums = [
            (gid, p_num) for gid, p_num in full_gids_p_nums if p_num not in already_done
        ]
        # Apply run limit as a multiple of patients_per_batch (full batches only).)
        if self.max_batches_per_run is not None:
            run_limit = self.max_batches_per_run * self.patients_per_batch
            if len(remaining_gids_p_nums) > run_limit:
                remaining_gids_p_nums = remaining_gids_p_nums[:run_limit]
                logger.info(
                    f"Run limit: processing at most {run_limit} patients "
                    f"({self.max_batches_per_run} batches) this run."
                )
        logger.info(
            f"Run {current_run_number} started {run_start_date}. "
            f"Total patients: {total_patients}, "
            f"already done: {len(already_done)}, "
            f"remaining (this run): {len(remaining_gids_p_nums)}."
        )

        if not remaining_gids_p_nums:
            logger.info("All patients already processed. Nothing to do.")
            return

        # Save run metadata before starting so it reflects this run even on crash.
        self._save_run_metadata(
            processed_path,
            patients_per_file=self.patients_per_file,
            run_number=current_run_number,
            first_run_date=first_run_date,
            last_run_date=run_start_date,
        )

        # Initialise the CSV checkpoint with a header if it doesn't exist yet.
        log_path = processed_path / PROCESSED_PATIENTS_LOG_FILENAME
        if not log_path.exists():
            log_path.write_text("p_num\n")

        # Initialise the queue for the producer and consumer threads to communicate.
        patient_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        stop_event = threading.Event()

        # Launch producer thread.
        producer_thread = threading.Thread(
            target=self._producer,
            args=(remaining_gids_p_nums, patient_queue, stop_event, processed_path),
            daemon=True,
            name="gluroo-producer",
        )
        producer_thread.start()

        # Consumer pool: submit work as items arrive from the queue.
        run_start_time = time.perf_counter()
        patients_done_this_run = 0
        patients_failed_this_run = 0

        # use max_workers - 1 cores for data processing. Leave 1 core for the producer thread and the main thread.
        with ProcessPoolExecutor(max_workers=(self.max_workers - 1)) as executor:
            # future -> p_num
            pending_futures: dict = {}

            while True:
                try:
                    item = patient_queue.get(timeout=5)
                except queue.Empty:
                    # Check if producer is done and no futures remain.
                    if not producer_thread.is_alive() and not pending_futures:
                        logger.info(
                            "Consumer: producer done and no futures remaining; exiting."
                        )
                        break
                    # Reap any completed futures while waiting.
                    self._reap_futures(pending_futures)
                    continue

                if item is _QUEUE_SENTINEL:
                    # Producer finished enqueuing; drain remaining futures.
                    break

                p_num, df_raw = item
                fut = executor.submit(
                    process_and_save_patient,
                    p_num,
                    df_raw,
                    staging_path,
                    processed_path / PROCESSED_PATIENTS_LOG_FILENAME,
                )
                pending_futures[fut] = p_num

                # Reap completed futures periodically to bound memory.
                if len(pending_futures) >= self.max_workers * 2:
                    done, failed = self._reap_futures(pending_futures)
                    patients_done_this_run += done
                    patients_failed_this_run += failed

            # Wait for all remaining workers.
            logger.info(
                f"Producer done. Waiting for {len(pending_futures)} in-flight workers..."
            )
            done, failed = self._reap_futures(pending_futures, wait_all=True)
            patients_done_this_run += done
            patients_failed_this_run += failed

        producer_thread.join()

        elapsed = time.perf_counter() - run_start_time
        total_done = len(already_done) + patients_done_this_run
        logger.info(
            f"Run {current_run_number} complete in {elapsed:.0f}s. "
            f"This run: {patients_done_this_run} done, {patients_failed_this_run} failed. "
            f"All-time total done: {total_done}/{total_patients}."
        )
        self._append_batch_timing(
            processed_path,
            run_number=current_run_number,
            elapsed_sec=elapsed,
            patients_done=patients_done_this_run,
            patients_failed=patients_failed_this_run,
            total_done=total_done,
            total_patients=total_patients,
        )

    # TODO: We need to make this fast enough to keep up with the consumer.
    # If conusmer is waiting too much (which I doubt woudl be the case), we need to increase patients_per_batch
    def _producer(
        self,
        remaining_gids_p_nums: list[tuple[str, str]],
        patient_queue: queue.Queue,
        stop_event: threading.Event,
        processed_path: Path,
    ) -> None:
        """
        Producer thread: queries DB in batches, splits by patient, enqueues items.

        Queries patients_per_batch patients per DB call for efficiency, then puts
        individual (p_num, DataFrame) tuples onto the queue.  Blocks on queue.put()
        when the queue is full — the OS suspends the thread (zero CPU) until a
        worker pops an item.

        Skipped patients (no valid data / too short) are appended to
        skipped_patient_ids.csv and are not enqueued.
        """
        batch_size = self.patients_per_batch
        total_remaining = len(remaining_gids_p_nums)
        # Need to add one more batch if not a multiple of batch_size.
        total_batches = (total_remaining + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            if stop_event.is_set():
                logger.warning("Producer: stop_event set; aborting.")
                break

            batch_start = batch_idx * batch_size
            batch = remaining_gids_p_nums[batch_start : batch_start + batch_size]
            batch_num = batch_idx + 1
            p_nums_in_batch = [p_num for _, p_num in batch]
            logger.info(
                f"Producer: fetching batch {batch_num}/{total_batches} "
                f"({len(batch)} patients from DB): {p_nums_in_batch}"
            )

            raw_data = self.load_raw(batch, processed_path)

            for p_num, df in raw_data.items():
                if stop_event.is_set():
                    break
                patient_queue.put(item=(p_num, df), block=True)

            logger.info(
                f"Producer: batch {batch_num}/{total_batches} enqueued "
                f"({len(raw_data)} valid patients)."
            )

        # No more patients to enqueue
        patient_queue.put(_QUEUE_SENTINEL)
        logger.info("Producer: sentinel enqueued; exiting.")

    @staticmethod
    def _reap_futures(pending: dict, wait_all: bool = False) -> tuple[int, int]:
        """
        Collect completed futures from pending dict (modified in place).
        If wait_all is True, wait for all futures to complete.
        Returns (done_count, failed_count).
        """
        done_count = 0
        failed_count = 0
        futures_to_check = list(
            as_completed(pending)
            if wait_all
            else [f for f in list(pending) if f.done()]
        )
        for fut in futures_to_check:
            p_num = pending.pop(fut)
            try:
                success = fut.result()
                if success:
                    done_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Worker returned failure for p_num={p_num}.")
            except Exception as exc:
                failed_count += 1
                logger.error(f"Worker exception for p_num={p_num}: {exc}")
        return done_count, failed_count

    # ------------------------------------------------------------------
    # Database access
    # ------------------------------------------------------------------

    def _get_all_patient_ids(self) -> list[tuple[str, str]]:
        """
        Return all (gid, p_num) pairs sorted by gid (deterministic order).

        gid: base64-encoded primary key in the groups table.
        p_num: string like "gluroo_0" assigned in order of gid.
        """
        engine = create_engine(self.db_connection_string)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT DISTINCT gid, p_num FROM groups ORDER BY gid")
            )
            return [(row[0], str(row[1])) for row in result.fetchall()]

    def load_raw(
        self,
        gids_p_nums: list[tuple[str, str]] | None = None,
        processed_path: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Query TimescaleDB for the given patients and apply validity filters.

        Patients are skipped (and appended to skipped_patient_ids.csv with a
        reason column) if they have no rows, no BGL readings, or fewer than
        min_date_span_days of data.

        Args:
            gids_p_nums: List of (gid, p_num) pairs to load.
            processed_path: Path for writing skipped_patient_ids.csv. If None,
                skipped IDs are logged but not written to disk.

        Returns:
            dict mapping p_num → raw DataFrame (gid column dropped).
        """
        engine = create_engine(self.db_connection_string)
        raw_data: dict[str, pd.DataFrame] = {}
        if not gids_p_nums:
            logger.info("load_raw: no patients requested.")
            return raw_data

        try:
            gids = [gid for gid, _ in gids_p_nums]
            all_df = self._load_all_patients_data_from_db(engine, gids)
            logger.info(
                f"load_raw: fetched {len(all_df)} rows for {len(gids)} patients."
            )
        except Exception as e:
            logger.warning(f"load_raw: DB error: {e}")
            return raw_data

        # (patient_id, skip_reason) for skipped_patient_ids.csv
        skipped: list[tuple[str, str]] = []

        if all_df.empty:
            skipped.extend((p_num, "empty_db_batch") for _, p_num in gids_p_nums)
            if processed_path:
                self._append_skipped_patient_ids_to_file(processed_path, skipped)
            logger.info("load_raw: no data returned from DB for any patient.")
            return raw_data

        for gid, p_num in gids_p_nums:
            df = all_df.loc[all_df["gid"] == gid].copy()

            if df.empty:
                skipped.append((p_num, "no_rows_for_gid"))
                continue

            # No readings
            if "bgl" not in df.columns or df["bgl"].notna().sum() == 0:
                skipped.append((p_num, "no_bgl_readings"))
                continue

            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            date_span = df["date"].max() - df["date"].min()
            if pd.isnull(date_span):
                skipped.append((p_num, "invalid_date_span"))
                continue
            if date_span.days < self.min_date_span_days:
                skipped.append(
                    (
                        p_num,
                        f"date_span_below_minimum_{date_span.days}d_lt_{self.min_date_span_days}d",
                    )
                )
                continue

            df = df.drop(columns=["gid"])
            df[ColumnNames.P_NUM.value] = df["p_num"]
            raw_data[p_num] = df

        if skipped:
            logger.info(f"load_raw: skipped {len(skipped)} patients.")
            if processed_path:
                self._append_skipped_patient_ids_to_file(processed_path, skipped)

        logger.info(f"load_raw: returning {len(raw_data)} valid patients.")
        return raw_data

    def _load_all_patients_data_from_db(self, engine, gids: list[str]) -> pd.DataFrame:
        """
        Single SQL query combining readings + messages for all given gids.

        Returns a DataFrame with a 'gid' column; callers split by gid.
        Rows are ordered by date, then gid (deterministic).
        """
        query = text("""
            SELECT
                r.gid,
                r.date,
                r.bgl,
                r.trend,
                r.source,
                NULL::TEXT as msg_type,
                NULL::FLOAT as food_g,
                NULL::FLOAT as dose_units,
                NULL::TEXT as dose_type,
                NULL::INT as exercise_mins,
                NULL::TEXT as exercise_level,
                g.p_num
            FROM readings r
            JOIN groups g ON r.gid = g.gid
            WHERE r.gid IN :gids
            UNION ALL
            SELECT
                m.gid,
                m.date,
                NULL::INT,
                NULL::TEXT as trend,
                NULL::TEXT as source,
                m.type as msg_type,
                m.food_g,
                m.dose_units,
                m.dose_type,
                m.exercise_mins,
                m.exercise_level,
                g.p_num
            FROM messages m
            JOIN groups g ON m.gid = g.gid
            WHERE m.gid IN :gids
            ORDER BY date, gid
        """).bindparams(bindparam("gids", expanding=True))

        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"gids": gids})

    # ------------------------------------------------------------------
    # Loading cached data
    # ------------------------------------------------------------------

    def _load_all_into_processed_data(self) -> None:
        """
        Load all patients from merged Parquet cache into processed_data (testing only).
        """
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        merged_path = processed_path / "parquet" / "merged"
        if not merged_path.exists():
            raise FileNotFoundError(
                f"Merged Parquet path not found: {merged_path}. "
                "Run merge_staging_to_merged() first."
            )

        parquet_files = sorted(merged_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning(
                "No merged Parquet files found; processed_data remains empty."
            )
            return

        logger.info(
            f"Loading all patients from {len(parquet_files)} merged Parquet file(s)..."
        )
        for path in parquet_files:
            df = pd.read_parquet(path)
            if ColumnNames.P_NUM.value not in df.columns:
                logger.warning(f"File {path} missing p_num column; skipping.")
                continue
            for p_num, group in df.groupby(ColumnNames.P_NUM.value, sort=False):
                p_num = str(p_num)
                if p_num not in self.processed_data:
                    self.processed_data[p_num] = group.copy()
                else:
                    self.processed_data[p_num] = pd.concat(
                        [self.processed_data[p_num], group], ignore_index=True
                    )

        logger.info(f"Loaded {len(self.processed_data)} patients into processed_data.")

    def _process_raw_data(self):
        """Required by DatasetBase. Not used in this loader."""
        raise NotImplementedError(
            "GlurooDataLoader does not implement _process_raw_data(). "
            "Use _process_and_cache_data_pipeline() + merge_staging_to_merged()."
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_processed_set(self, processed_path: Path) -> set[str]:
        """
        Read processed_patients.csv and return the set of completed p_nums.
        File must have a header row "p_num". If the file exists without a header
        (legacy), it is rewritten once with a header.
        Returns an empty set if the file does not exist.
        """
        log_path = processed_path / PROCESSED_PATIENTS_LOG_FILENAME
        if not log_path.exists():
            return set()
        try:
            df = pd.read_csv(log_path, dtype=str)
            if "p_num" in df.columns:
                return set(df["p_num"].dropna())
            # Legacy headless format: migrate to header format and return
            df = pd.read_csv(log_path, dtype=str, header=None, names=["p_num"])
            df = df[df["p_num"].notna() & (df["p_num"] != "p_num")].drop_duplicates()
            df.to_csv(log_path, index=False)
            return set(df["p_num"])
        except Exception as e:
            logger.warning(f"Could not read {log_path}: {e}; treating as empty.")
            return set()

    def _get_saved_patients_per_file(self, processed_path: Path) -> int | None:
        """
        Return the patients_per_file stored in processing_checkpoint.json, or None.
        """
        import json

        path = processed_path / PROCESSING_CHECKPOINT_FILENAME
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("patients_per_file")
        except (OSError, ValueError):
            return None

    def _load_run_metadata(self, processed_path: Path) -> dict:
        """Load run metadata from processing_checkpoint.json."""
        import json

        path = processed_path / PROCESSING_CHECKPOINT_FILENAME
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read run metadata: {e}")
            return {}

    def _save_run_metadata(
        self,
        processed_path: Path,
        patients_per_file: int,
        run_number: int,
        first_run_date: str,
        last_run_date: str,
    ) -> None:
        """Persist run metadata to processing_checkpoint.json."""
        import json

        path = processed_path / PROCESSING_CHECKPOINT_FILENAME
        try:
            with open(path, "w") as f:
                json.dump(
                    {
                        "patients_per_file": patients_per_file,
                        "run_number": run_number,
                        "first_run_date": first_run_date,
                        "last_run_date": last_run_date,
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            logger.warning(f"Could not save run metadata to {path}: {e}")

    def _append_skipped_patient_ids_to_file(
        self,
        processed_path: Path,
        entries: list[tuple[str, str]],
    ) -> None:
        """
        Append skipped p_nums to skipped_patient_ids.csv with a reason column.

        Columns: patient_id, reason. If an existing file has only patient_id
        (legacy), it is rewritten once with reason ``legacy_unknown`` for old
        rows, then new rows are appended.
        """
        if not entries:
            return
        path = processed_path / SKIPPED_PATIENT_IDS_FILENAME
        new_df = pd.DataFrame(entries, columns=["patient_id", "reason"])
        try:
            if not path.exists() or path.stat().st_size == 0:
                new_df.to_csv(path, index=False)
                return
            existing = pd.read_csv(path, dtype=str)
            if "reason" not in existing.columns:
                existing["reason"] = "legacy_unknown"
                existing.to_csv(path, index=False)
            new_df.to_csv(path, mode="a", header=False, index=False)
        except OSError as e:
            logger.warning(f"Could not append skipped patient IDs to {path}: {e}")

    def _append_batch_timing(
        self,
        processed_path: Path,
        run_number: int,
        elapsed_sec: float,
        patients_done: int,
        patients_failed: int,
        total_done: int,
        total_patients: int,
    ) -> None:
        """Append one row per run to batch_timings.csv for performance monitoring."""
        path = processed_path / BATCH_TIMINGS_FILENAME
        row = pd.DataFrame(
            [
                {
                    "run_number": run_number,
                    "elapsed_sec": f"{elapsed_sec:.2f}",
                    "patients_done_this_run": patients_done,
                    "patients_failed_this_run": patients_failed,
                    "total_done": total_done,
                    "total_patients": total_patients,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )
        try:
            row.to_csv(path, mode="a", header=not path.exists(), index=False)
        except OSError as e:
            logger.warning(f"Could not append run timing to {path}: {e}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _p_num_to_int(self, p_num: int | str) -> int:
        """Extract integer from p_num string (e.g., 'gluroo_123' → 123)."""
        if isinstance(p_num, int):
            return p_num
        match = re.search(r"(\d+)$", str(p_num))
        if match:
            return int(match.group(1))
        raise ValueError(f"Invalid p_num format: {p_num}")


# ---------------------------------------------------------------------------
# Standalone worker function (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def process_and_save_patient(
    p_num: str,
    df_raw: pd.DataFrame,
    staging_path: Path,
    log_path: Path,
) -> bool:
    """
    Process one patient and persist the result.  Designed for ProcessPoolExecutor.

    Steps:
        1. data_translation(df_raw)
        2. preprocessing_pipeline(p_num, df)
        3. Write staging/<p_num>.parquet
        4. Atomically append p_num to processed_patients.csv

    The CSV append uses O_APPEND which is atomic on Linux for writes ≤ 4096 bytes,
    so multiple workers can append concurrently without a lock.

    Args:
        p_num: Patient identifier string (e.g., 'gluroo_42').
        df_raw: Raw DataFrame from load_raw().
        staging_path: Directory to write <p_num>.parquet.
        log_path: Path to processed_patients.csv.

    Returns:
        True on success, False on failure (logged internally; caller counts it).
    """
    logger.info(f"Worker: processing p_num={p_num}")

    if df_raw.empty:
        logger.warning(f"Worker: empty raw DataFrame for p_num={p_num}; skipping.")
        return False

    if "p_num" not in df_raw.columns:
        logger.error(
            f"Worker: DataFrame missing p_num column for p_num={p_num}; skipping."
        )
        return False

    df_p_num = str(df_raw["p_num"].iloc[0])
    if df_p_num != p_num:
        logger.warning(f"Worker: p_num mismatch — expected {p_num}, got {df_p_num}.")

    try:
        df = data_translation(df_raw)
        df = preprocessing_pipeline(str(p_num), df, use_aggregation=True)

        if df.empty:
            logger.warning(
                f"Worker: preprocessing produced empty DataFrame for p_num={p_num}."
            )
            return False

        # Prepare for Parquet: drop plain 'date' column (datetime index is used),
        # ensure p_num column is present, reset index so datetime becomes a column.
        df_out = df.copy()
        if "date" in df_out.columns:
            df_out = df_out.drop(columns=["date"])
        df_out[ColumnNames.P_NUM.value] = p_num
        df_out = df_out.reset_index()

        out_path = staging_path / f"{p_num}.parquet"
        df_out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

        # Atomic append checkpoint (O_APPEND, ≤ 4096 bytes per write on Linux).
        with open(log_path, "a") as f:
            f.write(p_num + "\n")

        logger.debug(f"Worker: done p_num={p_num} → {out_path.name}")
        return True

    except Exception as e:
        logger.error(f"Worker: exception for p_num={p_num}: {e}")
        return False
