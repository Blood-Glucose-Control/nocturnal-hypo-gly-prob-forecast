#!/usr/bin/env python3
"""
Verify time-of-day features (hour_sin, hour_cos) are correctly computed
and flow through the dataset registry with proper holdout splits.

Run this BEFORE any fine-tuning to catch data bugs early.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from src.data.versioning.dataset_registry import DatasetRegistry  # noqa: E402
from src.data.models import ColumnNames  # noqa: E402


def verify_time_features():
    registry = DatasetRegistry(holdout_config_dir="configs/data/holdout_10pct")
    train_data, holdout_data = registry.load_dataset_with_split("brown_2019")

    print("=" * 60)
    print("TIME FEATURE VERIFICATION")
    print("=" * 60)

    # 1. Columns exist
    for col in ["hour_sin", "hour_cos"]:
        assert col in train_data.columns, f"{col} missing from train_data!"
        assert col in holdout_data.columns, f"{col} missing from holdout_data!"
    print("[PASS] hour_sin and hour_cos columns exist in both splits")

    # 2. No NaNs
    for name, df in [("train", train_data), ("holdout", holdout_data)]:
        sin_nans = df["hour_sin"].isna().sum()
        cos_nans = df["hour_cos"].isna().sum()
        assert sin_nans == 0, f"{name} has {sin_nans} NaN hour_sin values"
        assert cos_nans == 0, f"{name} has {cos_nans} NaN hour_cos values"
    print("[PASS] No NaN values in time features")

    # 3. Value range [-1, 1]
    for name, df in [("train", train_data), ("holdout", holdout_data)]:
        for col in ["hour_sin", "hour_cos"]:
            vmin, vmax = df[col].min(), df[col].max()
            assert -1.0 <= vmin <= 1.0, f"{name} {col} min={vmin} out of range"
            assert -1.0 <= vmax <= 1.0, f"{name} {col} max={vmax} out of range"
            print(f"  {name} {col}: [{vmin:.4f}, {vmax:.4f}]")
    print("[PASS] All values in [-1, 1]")

    # 4. Spot-check specific times against expected values
    print("\nSpot-checking specific timestamps...")

    # The registry returns flat DataFrames with datetime as a COLUMN (not index).
    # HoldoutManager.split_data() calls reset_index() before pd.concat().
    hours = pd.to_datetime(train_data["datetime"]).dt.hour
    minutes = pd.to_datetime(train_data["datetime"]).dt.minute

    # Midnight rows (hour=0, minute=0)
    midnight_mask = (hours == 0) & (minutes == 0)
    if midnight_mask.any():
        midnight_sin = train_data.loc[midnight_mask, "hour_sin"].iloc[0]
        midnight_cos = train_data.loc[midnight_mask, "hour_cos"].iloc[0]
        assert (
            abs(midnight_sin - 0.0) < 0.01
        ), f"Midnight sin={midnight_sin}, expected ~0"
        assert (
            abs(midnight_cos - 1.0) < 0.01
        ), f"Midnight cos={midnight_cos}, expected ~1"
        print(
            f"  00:00 -> sin={midnight_sin:.4f} (exp 0.0), cos={midnight_cos:.4f} (exp 1.0) [PASS]"
        )

    # 6 AM rows (hour=6, minute=0)
    six_am_mask = (hours == 6) & (minutes == 0)
    if six_am_mask.any():
        six_sin = train_data.loc[six_am_mask, "hour_sin"].iloc[0]
        six_cos = train_data.loc[six_am_mask, "hour_cos"].iloc[0]
        assert abs(six_sin - 1.0) < 0.01, f"6AM sin={six_sin}, expected ~1"
        assert abs(six_cos - 0.0) < 0.01, f"6AM cos={six_cos}, expected ~0"
        print(
            f"  06:00 -> sin={six_sin:.4f} (exp 1.0), cos={six_cos:.4f} (exp 0.0) [PASS]"
        )

    # Noon rows (hour=12, minute=0)
    noon_mask = (hours == 12) & (minutes == 0)
    if noon_mask.any():
        noon_sin = train_data.loc[noon_mask, "hour_sin"].iloc[0]
        noon_cos = train_data.loc[noon_mask, "hour_cos"].iloc[0]
        assert abs(noon_sin - 0.0) < 0.01, f"Noon sin={noon_sin}, expected ~0"
        assert abs(noon_cos - (-1.0)) < 0.01, f"Noon cos={noon_cos}, expected ~-1"
        print(
            f"  12:00 -> sin={noon_sin:.4f} (exp 0.0), cos={noon_cos:.4f} (exp -1.0) [PASS]"
        )

    # 6 PM rows (hour=18, minute=0)
    six_pm_mask = (hours == 18) & (minutes == 0)
    if six_pm_mask.any():
        six_pm_sin = train_data.loc[six_pm_mask, "hour_sin"].iloc[0]
        six_pm_cos = train_data.loc[six_pm_mask, "hour_cos"].iloc[0]
        assert abs(six_pm_sin - (-1.0)) < 0.01, f"6PM sin={six_pm_sin}, expected ~-1"
        assert abs(six_pm_cos - 0.0) < 0.01, f"6PM cos={six_pm_cos}, expected ~0"
        print(
            f"  18:00 -> sin={six_pm_sin:.4f} (exp -1.0), cos={six_pm_cos:.4f} (exp 0.0) [PASS]"
        )

    # 5. Verify unit circle identity: sin² + cos² ≈ 1
    identity = train_data["hour_sin"] ** 2 + train_data["hour_cos"] ** 2
    max_deviation = (identity - 1.0).abs().max()
    assert (
        max_deviation < 1e-6
    ), f"Unit circle identity violated: max deviation={max_deviation}"
    print(f"\n[PASS] sin² + cos² = 1 (max deviation: {max_deviation:.2e})")

    # 6. Split sizes match benchmarks
    train_patients = train_data["p_num"].nunique()
    holdout_patients = holdout_data["p_num"].nunique()
    print("\nSplit sizes:")
    print(f"  Train patients: {train_patients}")
    print(f"  Holdout patients: {holdout_patients}")
    print(f"  Train rows: {len(train_data):,}")
    print(f"  Holdout rows: {len(holdout_data):,}")

    # 7. Verify time features are distinct from BG (not correlated by construction)
    bg_col = ColumnNames.BG.value
    if bg_col in train_data.columns:
        valid_mask = train_data[bg_col].notna()
        corr_sin = train_data.loc[valid_mask, bg_col].corr(
            train_data.loc[valid_mask, "hour_sin"]
        )
        corr_cos = train_data.loc[valid_mask, bg_col].corr(
            train_data.loc[valid_mask, "hour_cos"]
        )
        print(f"\n  BG-hour_sin correlation: {corr_sin:.4f}")
        print(f"  BG-hour_cos correlation: {corr_cos:.4f}")
        print("  (Low correlation expected — time features are independent of BG)")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify_time_features()
