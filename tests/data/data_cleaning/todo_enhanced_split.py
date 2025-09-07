#!/usr/bin/env python3
"""
Test script for the enhanced get_train_validation_split function.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.preprocessing.time_processing import (
    create_datetime_index,
    get_train_validation_split,
)


def create_test_data():
    """Create sample test data with multiple patients."""

    # Create data for 3 patients with different characteristics
    patients = ["p01", "p02", "p03"]
    all_data = []

    for i, patient in enumerate(patients):
        # Create 30 days of data with 5-minute intervals
        start_time = datetime(2025, 1, 1, 6, 0, 0)  # Start at 6am
        end_time = start_time + timedelta(days=30)

        # Create time series
        times = []
        current_time = start_time
        while current_time < end_time:
            times.append(current_time.strftime("%H:%M:%S"))
            current_time += timedelta(minutes=5)

        # Create glucose values (random but realistic)
        np.random.seed(42 + i)  # Different seed for each patient
        n_points = len(times)
        glucose_values = (
            100
            + 50 * np.sin(np.linspace(0, 60 * np.pi, n_points))
            + 20 * np.random.randn(n_points)
        )
        glucose_values = np.clip(glucose_values, 50, 300)  # Realistic glucose range

        # Create patient dataframe
        patient_data = pd.DataFrame(
            {
                "id": range(i * n_points, (i + 1) * n_points),
                "p_num": patient,
                "time": times,
                "gl": glucose_values,
            }
        )

        all_data.append(patient_data)

    # Combine all patient data
    combined_data = pd.concat(all_data, ignore_index=True)

    # Create datetime index
    combined_data = create_datetime_index(combined_data)
    
    # Set datetime as index for the new function requirement
    combined_data = combined_data.set_index('datetime')

    return combined_data


def test_enhanced_function():
    """Test the enhanced get_train_validation_split function."""

    print("Creating test data...")
    df = create_test_data()
    print(
        f"Created test data with {len(df)} records for {df['p_num'].nunique()} patients"
    )
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"DataFrame has DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")

    print("\n" + "=" * 50)
    print("TEST 1: Basic functionality with default parameters")
    print("=" * 50)

    try:
        train_dict, val_dict, info = get_train_validation_split(
            df, num_validation_days=5
        )

        print("✅ Split successful!")
        print(f"   Train patients: {list(train_dict.keys())}")
        print(f"   Validation patients: {list(val_dict.keys())}")
        print(f"   Total train records: {sum(len(df) for df in train_dict.values())}")
        print(
            f"   Total validation records: {sum(len(df) for df in val_dict.values())}"
        )
        print(f"   Patients included: {info['patients_included']}")
        print(f"   Split ratio: {info['split_ratio']:.3f}")

        if info["patients_excluded"]:
            print(f"   Patients excluded: {info['patients_excluded']}")
            print(f"   Exclusion reasons: {info['exclusion_reasons']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    print("\n" + "=" * 50)
    print("TEST 2: Different day start hour (8am)")
    print("=" * 50)

    try:
        train_dict, val_dict, info = get_train_validation_split(
            df, num_validation_days=3, day_start_hour=8
        )

        print("✅ Split with 8am start successful!")
        print(f"   Total train records: {sum(len(df) for df in train_dict.values())}")
        print(
            f"   Total validation records: {sum(len(df) for df in val_dict.values())}"
        )
        print(f"   Patients included: {info['patients_included']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    print("\n" + "=" * 50)
    print("TEST 3: Include partial days")
    print("=" * 50)

    try:
        train_dict, val_dict, info = get_train_validation_split(
            df, num_validation_days=3, include_partial_days=True
        )

        print("✅ Split with partial days successful!")
        print(f"   Total train records: {sum(len(df) for df in train_dict.values())}")
        print(
            f"   Total validation records: {sum(len(df) for df in val_dict.values())}"
        )
        print(f"   Patients included: {info['patients_included']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    print("\n" + "=" * 50)
    print("TEST 4: Error handling - DataFrame without DatetimeIndex")
    print("=" * 50)

    try:
        # Test with DataFrame that doesn't have DatetimeIndex
        df_no_index = df.reset_index()  # This removes the DatetimeIndex
        train_dict, val_dict, info = get_train_validation_split(df_no_index)
        print("❌ Should have failed but didn't!")

    except TypeError as e:
        print(f"✅ Correctly caught DatetimeIndex error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print("\n" + "=" * 50)
    print("TEST 5: Error handling - invalid patient column")
    print("=" * 50)

    try:
        train_dict, val_dict, info = get_train_validation_split(
            df, patient_col="invalid_column"
        )
        print("❌ Should have failed but didn't!")

    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print("\n" + "=" * 50)
    print("TEST 6: Error handling - too many validation days")
    print("=" * 50)

    try:
        train_dict, val_dict, info = get_train_validation_split(
            df,
            num_validation_days=50,  # More than available data
        )
        print("❌ Should have failed but didn't!")

    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print("\n" + "=" * 50)
    print("TEST 7: Detailed split info examination")
    print("=" * 50)

    try:
        train_dict, val_dict, info = get_train_validation_split(
            df, num_validation_days=5
        )

        print("Split Information:")
        print(f"   Train patients and record counts:")
        for patient_id, patient_df in train_dict.items():
            print(f"      {patient_id}: {len(patient_df)} records")

        print(f"   Validation patients and record counts:")
        for patient_id, patient_df in val_dict.items():
            print(f"      {patient_id}: {len(patient_df)} records")

        print(f"   Metadata:")
        for key, value in info.items():
            if key in [
                "train_data_range",
                "validation_data_range",
                "validation_days_actual",
            ]:
                print(f"      {key}:")
                for patient, patient_value in value.items():
                    print(f"         {patient}: {patient_value}")
            else:
                print(f"      {key}: {value}")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_enhanced_function()
