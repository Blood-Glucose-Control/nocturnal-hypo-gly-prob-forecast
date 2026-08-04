#!/usr/bin/env python3
"""Count non-null CGM values in MetaboNet dataset."""

import pandas as pd
from pathlib import Path

data_dir = Path("cache/data/metabonet_2026")

print("Counting non-null CGM values...")
print("(This may take a moment for 154M rows)")
print()

# Read only CGM column from both files
train_cgm = pd.read_parquet(data_dir / "train.parquet", columns=["CGM"])
test_cgm = pd.read_parquet(data_dir / "test.parquet", columns=["CGM"])

# Count non-null values
train_non_null = train_cgm["CGM"].notna().sum()
test_non_null = test_cgm["CGM"].notna().sum()
total_non_null = train_non_null + test_non_null

# Total rows
train_total = len(train_cgm)
test_total = len(test_cgm)
total_rows = train_total + test_total

print("=" * 80)
print("CGM NON-NULL COUNT")
print("=" * 80)
print("\nTrain set:")
print(f"  Non-null CGM: {train_non_null:,}")
print(f"  Total rows:   {train_total:,}")
print(f"  Coverage:     {train_non_null/train_total*100:.2f}%")

print("\nTest set:")
print(f"  Non-null CGM: {test_non_null:,}")
print(f"  Total rows:   {test_total:,}")
print(f"  Coverage:     {test_non_null/test_total*100:.2f}%")

print("\nCombined:")
print(f"  Non-null CGM: {total_non_null:,}")
print(f"  Total rows:   {total_rows:,}")
print(f"  Coverage:     {total_non_null/total_rows*100:.2f}%")
print()
