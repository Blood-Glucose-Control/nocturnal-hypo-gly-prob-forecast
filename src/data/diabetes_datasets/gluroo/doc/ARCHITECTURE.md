# Gluroo Dataset Storage Architecture

## Overview

The Gluroo dataset uses a **partitioned Parquet** storage system optimized for large-scale time series data (100K+ patients, 500GB-1TB). This architecture enables efficient streaming for training.

## Problem Statement

- **Scale**: 100K+ patients with time series data
- **Size**: 500GB-1TB total dataset size
- **Requirement**: Efficient streaming for training (no need for random patient lookup)
- **Constraint**: Data must remain local (sensitive, cannot leave computer)

## Solution: Partitioned Parquet Architecture

### Storage Structure

```
cache/data/gluroo/
├── raw/                    # Empty (raw data stored in TimescaleDB)
└── processed/
    └── parquet/            # Partitioned Parquet dataset
        ├── partition=000/
        │   ├── batch_0000.parquet  # ~500MB-2GB, contains ~100-500 patients
        │   ├── batch_0001.parquet
        │   └── ...
        ├── partition=001/
        │   └── ...
        └── partition=255/   # Sequential partitions (0-255)
            └── ...
```

### Key Design Decisions

1. **Sequential Partitioning**: Patients batched sequentially into partitions
   - Since all patients have similar size (~1 year of data), simple sequential batching works well
   - Patients are assigned to partitions in order (partition 0 = first ~400 patients, partition 1 = next ~400, etc.)
   - Avoids filesystem limitations (keeps files per directory manageable)
   - Enables parallel reads across partitions
   - No hash calculation needed - simpler and faster than hash-based partitioning

2. **Batched Storage**: Multiple patients per Parquet file (~100-500 patients, ~500MB-2GB per file)
   - Reduces file count (manageable for 100K+ patients)
   - Better compression ratios
   - Efficient for streaming

3. **Compression**: Snappy compression (fast, good compression ratio)

## Data Flow

### Writing (Processing)

```
TimescaleDB (Raw Data)
    ↓
Load patients sequentially/parallel
    ↓
Process each patient (preprocessing pipeline)
    ↓
Batch patients sequentially into partitions
    ↓
Save to partitioned Parquet files
```

### Reading (Training)

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
  - Parallel reads across partitions
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

- `_save_as_partitioned_parquet()`: Batches patients and saves to partitioned structure
- `_load_from_parquet()`: Loads all data from Parquet (for compatibility)
- `get_parquet_dataset()`: Returns PyArrow Dataset for streaming access

### Usage

```python
# Standard usage (loads into memory)
loader = GlurooDataLoader(use_cached=True)
train_data = loader.train_data  # dict[patient_id, DataFrame]

# Streaming access (for large datasets)
parquet_dataset = loader.get_parquet_dataset()
# Use with PyTorch Dataset wrapper for lazy loading
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
