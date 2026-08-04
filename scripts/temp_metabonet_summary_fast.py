#!/usr/bin/env python3
"""
Fast MetaboNet dataset summary - uses efficient pandas operations.
"""

import pandas as pd
from pathlib import Path

data_dir = Path("cache/data/metabonet_2026")

print("=" * 80)
print("METABONET DATASET SUMMARY (FAST MODE)")
print("=" * 80)

# Read just metadata first
print("\nReading file metadata...")
train_file = data_dir / "train.parquet"
test_file = data_dir / "test.parquet"

# Get row counts efficiently using parquet metadata
import pyarrow.parquet as pq  # noqa: E402

train_pf = pq.ParquetFile(train_file)
test_pf = pq.ParquetFile(test_file)

train_rows = train_pf.metadata.num_rows
test_rows = test_pf.metadata.num_rows
total_rows = train_rows + test_rows

print("\n" + "=" * 80)
print("DATASET SIZES")
print("=" * 80)
print(f"Train set: {train_rows:,} rows")
print(f"Test set: {test_rows:,} rows")
print(f"Total: {total_rows:,} rows")
print(f"Train %: {train_rows / total_rows * 100:.1f}%")
print(f"Test %: {test_rows / total_rows * 100:.1f}%")

# Get column info from schema
print("\n" + "=" * 80)
print("COLUMN INFORMATION")
print("=" * 80)
schema = train_pf.schema_arrow
print(f"Total columns: {len(schema)}")
print("\nColumns:")
for i, field in enumerate(schema, 1):
    print(f"  {i:2d}. {field.name:30s} ({field.type})")

# Read small samples efficiently
print("\n" + "=" * 80)
print("PATIENT COUNTS")
print("=" * 80)

# Read only 'id' column (full dataset - should still be reasonable)
print("Counting unique patients from id column...")
train_ids = pd.read_parquet(train_file, columns=["id"])
test_ids = pd.read_parquet(test_file, columns=["id"])

train_patients = train_ids["id"].nunique()
test_patients = test_ids["id"].nunique()
overlap = len(set(train_ids["id"].unique()) & set(test_ids["id"].unique()))

print(f"  Train patients: {train_patients:,}")
print(f"  Test patients: {test_patients:,}")
print(f"  Total unique: {train_patients + test_patients - overlap:,}")
print(f"  Overlap: {overlap}")

# Calculate rows per patient
print("\nAverage rows per patient:")
print(f"  Train: {train_rows / train_patients:.0f} rows/patient")
print(f"  Test: {test_rows / test_patients:.0f} rows/patient")

# Read a small sample of full data for statistics
print("\n" + "=" * 80)
print("DATA SAMPLE (first 5 rows)")
print("=" * 80)
# Use filters to limit rows read
sample_df = pq.read_table(train_file).slice(0, 10000).to_pandas()
print(sample_df.head())

print("\n" + "=" * 80)
print("KEY STATISTICS (from 10K row sample)")
print("=" * 80)

# CGM statistics
if "CGM" in sample_df.columns:
    cgm_data = sample_df["CGM"].dropna()
    print("\nCGM (blood glucose):")
    print(
        f"  Non-null in sample: {len(cgm_data)} ({len(cgm_data)/len(sample_df)*100:.1f}%)"
    )
    if len(cgm_data) > 0:
        print(f"  Mean: {cgm_data.mean():.1f} mg/dL")
        print(f"  Std: {cgm_data.std():.1f} mg/dL")
        print(f"  Min: {cgm_data.min():.1f} mg/dL")
        print(f"  Max: {cgm_data.max():.1f} mg/dL")

# Insulin
if "insulin" in sample_df.columns:
    insulin_events = sample_df[sample_df["insulin"] > 0]
    print(f"\nInsulin events in sample: {len(insulin_events)}")
    if len(insulin_events) > 0:
        print(f"  Mean dose: {insulin_events['insulin'].mean():.2f} IU")
        print(f"  Max dose: {insulin_events['insulin'].max():.2f} IU")

# Carbs
if "carbs" in sample_df.columns:
    carb_events = sample_df[sample_df["carbs"] > 0]
    print(f"\nCarb events in sample: {len(carb_events)}")
    if len(carb_events) > 0:
        print(f"  Mean: {carb_events['carbs'].mean():.1f} g")
        print(f"  Max: {carb_events['carbs'].max():.1f} g")

# Source distribution
if "source_file" in sample_df.columns:
    print("\nSource file distribution (in 10K sample):")
    sources = sample_df["source_file"].value_counts().head(10)
    for source, count in sources.items():
        print(f"  {source:25s}: {count:5,}")

# Date range (from sample)
if "date" in sample_df.columns:
    sample_df["date"] = pd.to_datetime(sample_df["date"])
    print("\nDate range (in sample):")
    print(f"  Min: {sample_df['date'].min()}")
    print(f"  Max: {sample_df['date'].max()}")

print("\n" + "=" * 80)
print("SUMMARY COMPLETE")
print("=" * 80)
print("\nNote: Statistics based on samples for speed.")
print("Full dataset analysis would take longer with 132M+ rows.")
