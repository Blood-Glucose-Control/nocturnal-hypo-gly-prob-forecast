# Gluroo Dataset Storage Architecture

## Overview

The Gluroo dataset uses a **Parquet** storage system optimized for large-scale time series data (100K+ patients, 500GB-1TB). This architecture enables efficient streaming for training.

## Problem Statement

- **Scale**: 100K+ patients with time series data
- **Size**: 500GB-1TB total dataset size
- **Requirement**: Efficient streaming for training (no need for random patient lookup)
- **Constraint**: Data must remain local (sensitive, cannot leave computer)

TODO:
1. Figure out the size per patient on average.
2. Figure out number of patients per file we need
3. Figure out number of patients per batch we need

## Solution: Parquet Architecture

### Storage Structure

```
cache/data/gluroo/
├── raw/                    # Empty (raw data stored in TimescaleDB)
└── processed/
    └── parquet/            # Parquet dataset
        ├── file_000000.parquet
        ├── file_000001.parquet
        └── ...
```

### Key Design Decisions

1. **Partitioning by patient ID**: Patients are assigned to files by numeric patient ID
   - e.g. gluroo_0..gluroo_399 → file_000000.parquet, gluroo_400..gluroo_799 → file_000001.parquet
   - Simple rule: `file_id = p_num_numeric // patients_per_file`
   - No global ordering required; each patient always maps to the same file
   - Enables parallel reads across files

2. **Batched Storage**: Multiple patients per Parquet file (~400 patients per file by default, configurable)
   - Reduces file count (manageable for 100K+ patients)
   - Better compression ratios
   - Efficient for streaming

3. **Compression**: Snappy compression (fast, good compression ratio)

## Data Flow

### Writing (Processing)

```
TimescaleDB (Raw Data)
    ↓
Load patients in batches
    ↓
Process each patient (preprocessing pipeline)
    ↓
Assign each patient to file by p_num (file_id = p_num // patients_per_file)
    ↓
Save to Parquet files (read existing if present → concat → write)
```
### Below is WIP. Will be updated later.

### Reading (Training) - WIP

```
Parquet Files
    ↓
PyArrow Dataset (lazy loading, column pruning)
    ↓
Load into memory (or stream)
    ↓
PyTorch DataLoader (batching, multiprocessing, GPU transfer)
    ↓
Training Loop
```

## PyArrow vs PyTorch DataLoader

### PyArrow Dataset (File I/O Layer)
- **Purpose**: Efficient disk I/O
- **Features**:
  - Column pruning (only read needed columns)
  - Predicate pushdown (filter at disk level)
  - Zero-copy reads
  - Parallel reads across files
- **Returns**: Arrow Tables/RecordBatches

### PyTorch DataLoader (Training Layer)
- **Purpose**: GPU training interface
- **Features**:
  - Batching samples
  - Multiprocessing (`num_workers`)
  - Prefetching (overlap I/O with training)
  - GPU memory transfer (`pin_memory`)
  - Shuffling and collation
- **Returns**: PyTorch Tensors

**They work together**: PyArrow handles efficient disk reads, PyTorch DataLoader handles training batching.

## Implementation

### Key Methods

- `_save_batch_incremental()`: Saves each batch to Parquet; assigns patients to files by p_num, read-if-exists → concat → write per file
- `_load_all_into_processed_data()`: Loads processed data (e.g. from Parquet/cache) into memory
- Streaming: Hugging Face `load_dataset(..., streaming=True)` or similar over the parquet directory

### Usage

```python
# Standard usage (loads into memory)
loader = GlurooDataLoader(use_cached=True)
train_data = loader.train_data  # dict[patient_id, DataFrame]

# Streaming access (for large datasets)
# Use load_dataset or IterableDataset over cache/data/gluroo/processed/parquet/
```

## Benefits

- ✅ Scales to 100K+ patients
- ✅ Efficient streaming for training
- ✅ Parallel reads (multiple CPU cores)
- ✅ Reduced storage (compression)
- ✅ Works with existing PyTorch DataLoader

## Future Enhancements

1. **Lazy Loading**: PyTorch Dataset wrapper for true streaming (don't load all into memory)
2. **Metadata Index**: Fast lookup files for dataset statistics
3. **Optimized Batch Sizes**: Tune based on actual data characteristics
