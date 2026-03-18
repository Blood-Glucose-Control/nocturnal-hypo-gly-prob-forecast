# Gluroo Dataset Storage Architecture

## Overview

The Gluroo dataset uses a **Parquet** storage system optimized for large-scale time series data
(100K+ patients, estimated 500 GB–1 TB). Processing runs as a long-running resumable job with a
producer-consumer pipeline that overlaps database I/O with CPU-intensive preprocessing.

## Problem
- **Scale**: 100K+ patients. Even with timescaleDB, things can still be slow.
- **Size**: a few TB total dataset size
- **Bottleneck**: Per-batch timing shows DB query ~6 min, preprocessing ~30–50 min — processing
  dominates, but DB time is currently wasted (idle while workers run). On top of that, we need to always keep worker busy. We can' dispatch them in batches because some might finish early and just idle until all finished.
- **Requirement**: Efficient streaming for training; no need for random patient lookup
- **Constraint**: Data must remain local (sensitive, cannot leave the machine)
- **Resumability**: Job runs over multiple days; must resume from exactly where it left off
  without reprocessing completed patients

## Solution: Producer-Consumer Pipeline with Staged Parquet

### Core Idea

Overlap the slow DB query with the CPU-intensive preprocessing by running them concurrently,
with the queue holding **individual patients** (not whole batches):

- **Producer**: one background thread issues batched SQL queries against TimescaleDB (100
  patients per query for efficiency), splits the result by patient, and enqueues individual
  `(p_num, DataFrame)` items. Blocks when the queue is full (back-pressure).
- **Consumer pool**: `ProcessPoolExecutor` with `max_workers` processes. Each worker pops one
  patient from the queue, runs the full preprocessing pipeline, writes one staging Parquet file,
  then atomically appends its `p_num` to `processed_patients.log`. Workers that finish early
  immediately pick up the next patient — no synchronization barrier at batch boundaries.
- **Queue size**: 60–80 patients. At ~2 min average per patient with 20 workers, the pool
  consumes ~10 patients/min. A 6-minute DB query gap needs ~60 patients buffered to keep all
  workers fed. Tune `QUEUE_MAXSIZE` against available RAM (~5–20 MB per patient raw DataFrame).

**Why patient-level, not batch-level?**
Patients vary enormously in size (8k rows to 186k rows observed). With batch-level queuing,
slow "straggler" patients at the end of a batch hold up all idle workers until the whole batch
drains before the next one is enqueued. With patient-level queuing, a worker that finishes early
immediately picks up the next patient — the pool stays fully saturated at all times.

### Storage Structure

```
cache/data/gluroo_2026/
├── raw/                              # Empty (raw data lives in TimescaleDB)
└── processed/
    ├── processed_patients.log        # Append-only checkpoint: one p_num per line
    ├── skipped_patient_ids.csv       # patient_id + reason (append-only)
    ├── processing_checkpoint.json    # Run metadata (run count, dates, patients_per_file)
    ├── batch_timings.csv             # Per-batch elapsed time for monitoring
    └── parquet/
        ├── staging/                  # Permanent; one file per patient (~100k files total)
        │   ├── gluroo_0.parquet
        │   ├── gluroo_1.parquet
        │   └── ...
        └── merged/                   # Derived; ~250 files; rebuilt from staging on demand
            ├── file_000000.parquet   # gluroo_0 … gluroo_399
            ├── file_000001.parquet   # gluroo_400 … gluroo_799
            └── ...
```

**Why two layers?**

| Layer | Files | Written by | Purpose |
|-------|-------|------------|---------|
| `staging/` | ~100k (permanent) | Consumer workers (concurrent, no conflicts) | Source of truth; durable recovery point. Easy to work with in parallel processing |
| `merged/` | ~250 (disposable) | Merge step (sequential, post-processing) | Training reads; fast glob; rebuildable from staging |

`staging/` is never deleted. Workers write one file each — no read-modify-write, no conflicts.
`merged/` can be discarded and reconstructed at any time without reprocessing any patient.

**Note:**
We intentionally don't write multiple patients into a file during processing because:
- multi-threading: multiple threads can try to write to the same file at the same time (which is gonna be a nightmare to debug)
- Read performance during training: We can merge different number of patients into a file given the IO efficiency of the file system. -> Tunable parameter for training woohoo!
- I think it is worth it to sacrifice some storage space for the sake of readability and maintainability. Hard drive is dirt cheap to have serveral duplicated copies anyway.


### Key Design Decisions

#### 1. Per-patient checkpointing via `processed_patients.log`

The checkpoint unit is an individual patient, not a batch.

- On startup: read `processed_patients.log` → build a `set` of completed `p_num` strings.
  O(n) in the number of processed patients; runs in milliseconds even at 100k entries.
- After a worker saves `staging/gluroo_N.parquet` successfully: atomically append `gluroo_N\n`
  to `processed_patients.log`. On Linux, `O_APPEND` writes ≤ 4096 bytes are atomic at the OS
  level, so multiple workers can append concurrently without a lock.
- Producer filters the full patient list against this set before querying the DB — already-done
  patients are never re-queried or re-processed. There is also a run limit parameter for the producer to limit the number of patients to process in a single run. For example, we might just want 500 patients/batch * 10 batches = 5000 patients this run.

A crash mid-batch is safe: completed workers have already checkpointed; failed/incomplete ones
have not, so they are retried automatically on the next run.

#### 2. Staging files are permanent

Processing 100k patients takes multiple days. Staging files are kept indefinitely as the durable
record of work done. Benefits:

- **Recovery**: if `merged/` is lost or corrupted, reconstruct from staging without any DB query
- **Selective reprocessing**: to reprocess a single patient, delete
  `staging/gluroo_N.parquet`, remove its line from `processed_patients.log`, and re-run —
  all other patients are untouched
- **Pipeline versioning**: if the preprocessing logic changes, staging captures exactly what
  version produced each file (combine with a pipeline version tag in parquet metadata if needed)

#### 3. Merge step is separate and incremental

The merge step reads `processed_patients.log` (the exact list of completed patients — no glob
needed over the 100k staging directory) and consolidates staging files into the `merged/` layout.

Ideally, we would want to tune this. More patients per files = more efficient I/O during training, less granularity for train/test/validation splits.

```
file_id = p_num_numeric // patients_per_file
```

The merge can be run:
- Once at the end of a full processing run
- Periodically (e.g. every 5,000 new patients) to keep `merged/` fresh for training
- Never touches staging; idempotent; fast (pure I/O, no CPU preprocessing)

`patients_per_file` (default 400) gives ~250 merged files for 100k patients — a good balance
between file count and read granularity.

#### 4. Producer is a thread; workers are processes

- DB query (SQLAlchemy + `pd.read_sql`) is I/O-bound. The GIL is released during network I/O,
  so a **thread** is sufficient for the producer. No separate process needed.
- `create_physiological_features` (COB/IOB/aggregation) is CPU-bound. Workers run in a
  **ProcessPoolExecutor** to bypass the GIL.
- Might need to crank up the number of patients per batch for the producer to keep up with the consumer (which I doubt would be a concern)

#### 5. Parquet compression: Snappy

Fast encode/decode with good compression ratio. Appropriate for the read-heavy training workload.

## Data Flow

### Writing (Processing)

```
Startup
  └─ read processed_patients.log → already_done set
  └─ full_list = _get_all_patient_ids() from DB
  └─ remaining = full_list - already_done

Producer thread:
1. Issues batched SQL queries against TimescaleDB
2. Splits each batch by patient and enqueues (p_num, df) items.
3. Once all patiens of this batch are enqueued, fetch the next batch.


--- Queue  (size is tunable) ---


Consumer workers:
1. Pop one patient at a time, run the full preprocessing pipeline, write one staging Parquet file, then atomically append its `p_num` to `processed_patients.log`.

```

### Reading (Training) - WIP

```
merged/*.parquet  (250 files)
    ↓
HuggingFace load_dataset(..., streaming=True)   or   PyArrow Dataset
    ↓
PyTorch DataLoader (batching, num_workers, pin_memory, prefetch)
    ↓
Training loop
```

Training always reads from `merged/`, never from `staging/`.

## Checkpoint Files Reference

| File | Format | Written by | Content |
|------|--------|------------|---------|
| `processed_patients.log` | plain text, one `p_num` per line | Consumer workers (atomic append) | Primary resume state |
| `skipped_patient_ids.csv` | CSV, append-only | Producer thread after `load_raw` | `patient_id`, `reason` (e.g. `no_bgl_readings`, `date_span_below_minimum_*`) |
| `processing_checkpoint.json` | JSON | Orchestrator | `run_number`, `first_run_date`, `last_run_date`, `patients_per_file` (must be constant across runs) |
| `batch_timings.csv` | CSV, append-only | Orchestrator | Per-batch elapsed seconds for monitoring |

`processing_checkpoint.json` no longer stores `next_batch_start` — that is fully replaced by
`processed_patients.log`. It retains metadata fields for observability only.

## Implementation

### Key Methods (current → proposed)

| Current method | Proposed replacement | Notes |
|---------------|---------------------|-------|
| `_process_and_cache_data()` | `_process_and_cache_data_pipeline()` | Launches producer thread + consumer pool |
| `_process_raw_data_batch()` | `_consumer_worker()` (standalone fn) | Process + save + checkpoint in one step |
| `_save_batch_incremental()` | `_save_single_patient_parquet()` | Writes to `staging/`; no read-modify-write |
| `_save_checkpoint()` | `_append_to_processed_log()` | Atomic per-patient append |
| *(new)* | `merge_staging_to_merged()` | Consolidates staging → merged; run on demand |

### Usage

```python
# Process (long-running job — run via nohup or SLURM)
loader = GlurooDataLoader(use_cached=False, max_workers=10, patients_per_batch=100)

# Merge staging → merged (run after processing, or periodically)
loader.merge_staging_to_merged()

# Training (reads merged/ only)
loader = GlurooDataLoader(use_cached=True)
dataset = loader.get_hf_streaming_dataset()
```
