#!/usr/bin/env python3
"""
Temporary script to get MetaboNet dataset summary information.
Provides row counts, unique patient counts, and basic statistics.
"""

import pandas as pd
from pathlib import Path

# Path to MetaboNet data
data_dir = Path("cache/data/metabonet_2026")

print("=" * 80)
print("METABONET DATASET SUMMARY")
print("=" * 80)

# Load parquet files
print("\nLoading parquet files...")
train_df = pd.read_parquet(data_dir / "train.parquet")
test_df = pd.read_parquet(data_dir / "test.parquet")

print("\n" + "=" * 80)
print("DATASET SIZES")
print("=" * 80)
print("Train set:")
print(f"  Rows: {len(train_df):,}")
print(f"  Columns: {len(train_df.columns)}")
print(f"  Memory: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nTest set:")
print(f"  Rows: {len(test_df):,}")
print(f"  Columns: {len(test_df.columns)}")
print(f"  Memory: {test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

total_rows = len(train_df) + len(test_df)
print("\nTotal:")
print(f"  Rows: {total_rows:,}")
print(f"  Train %: {len(train_df) / total_rows * 100:.1f}%")
print(f"  Test %: {len(test_df) / total_rows * 100:.1f}%")

print("\n" + "=" * 80)
print("PATIENT COUNTS")
print("=" * 80)
train_patients = train_df["id"].nunique()
test_patients = test_df["id"].nunique()
overlap = len(set(train_df["id"].unique()) & set(test_df["id"].unique()))

print(f"Train patients: {train_patients:,}")
print(f"Test patients: {test_patients:,}")
print(f"Total unique patients: {train_patients + test_patients - overlap:,}")
print(f"Overlap (patients in both): {overlap}")

print("\n" + "=" * 80)
print("COLUMN INFORMATION")
print("=" * 80)
print(f"Total columns: {len(train_df.columns)}")
print("\nColumns:")
for i, col in enumerate(train_df.columns, 1):
    dtype = train_df[col].dtype
    non_null = train_df[col].notna().sum()
    pct = non_null / len(train_df) * 100
    print(f"  {i:2d}. {col:30s} ({dtype:10s}) - {pct:5.1f}% non-null")

print("\n" + "=" * 80)
print("DATE RANGE")
print("=" * 80)
if "date" in train_df.columns:
    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    print("Train set:")
    print(f"  Start: {train_df['date'].min()}")
    print(f"  End: {train_df['date'].max()}")
    print(f"  Span: {(train_df['date'].max() - train_df['date'].min()).days} days")

    print("\nTest set:")
    print(f"  Start: {test_df['date'].min()}")
    print(f"  End: {test_df['date'].max()}")
    print(f"  Span: {(test_df['date'].max() - test_df['date'].min()).days} days")

print("\n" + "=" * 80)
print("SAMPLE DATA (First 5 rows of train set)")
print("=" * 80)
print(train_df.head())

print("\n" + "=" * 80)
print("KEY COLUMN STATISTICS (Train set)")
print("=" * 80)

# CGM statistics
if "CGM" in train_df.columns:
    cgm_stats = train_df["CGM"].describe()
    print("\nCGM (blood glucose):")
    print(
        f"  Non-null: {train_df['CGM'].notna().sum():,} ({train_df['CGM'].notna().sum() / len(train_df) * 100:.1f}%)"
    )
    print(f"  Mean: {cgm_stats['mean']:.1f} mg/dL")
    print(f"  Std: {cgm_stats['std']:.1f} mg/dL")
    print(f"  Min: {cgm_stats['min']:.1f} mg/dL")
    print(f"  Max: {cgm_stats['max']:.1f} mg/dL")

# Insulin statistics
if "insulin" in train_df.columns:
    insulin_events = train_df[train_df["insulin"] > 0]
    print("\nInsulin:")
    print(f"  Total events: {len(insulin_events):,}")
    print(f"  Events per patient: {len(insulin_events) / train_patients:.1f}")
    if len(insulin_events) > 0:
        print(f"  Mean dose: {insulin_events['insulin'].mean():.2f} IU")
        print(f"  Max dose: {insulin_events['insulin'].max():.2f} IU")

# Carbs statistics
if "carbs" in train_df.columns:
    carb_events = train_df[train_df["carbs"] > 0]
    print("\nCarbs:")
    print(f"  Total events: {len(carb_events):,}")
    print(f"  Events per patient: {len(carb_events) / train_patients:.1f}")
    if len(carb_events) > 0:
        print(f"  Mean intake: {carb_events['carbs'].mean():.1f} g")
        print(f"  Max intake: {carb_events['carbs'].max():.1f} g")

# Source file distribution
if "source_file" in train_df.columns:
    print("\nSource file distribution (top 10):")
    source_counts = train_df["source_file"].value_counts().head(10)
    for source, count in source_counts.items():
        pct = count / len(train_df) * 100
        print(f"  {source:25s}: {count:8,} ({pct:5.1f}%)")

print("\n" + "=" * 80)
print("SUMMARY COMPLETE")
print("=" * 80)
