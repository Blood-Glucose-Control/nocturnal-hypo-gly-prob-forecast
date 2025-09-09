import numpy as np
import pandas as pd

from src.data.preprocessing.time_processing import get_train_validation_split

# Create sample data for a single patient
np.random.seed(42)
dates = pd.date_range("2024-01-01 06:00:00", periods=1000, freq="5T")
single_patient_data = pd.DataFrame(
    {
        "glucose": np.random.normal(120, 20, 1000),
        "some_other_col": np.random.randn(1000),
    },
    index=dates,
)

print("Testing enhanced get_train_validation_split() with single patient...")

# TEST 1: Basic functionality
print("\n=== TEST 1: Basic functionality ===")
try:
    train, val, info = get_train_validation_split(
        single_patient_data, num_validation_days=2
    )
    print(f"✅ Success! Train: {len(train)} records, Validation: {len(val)} records")
    print(f"   Split ratio: {info['split_ratio']:.3f}")
    print(f"   Validation days actual: {info['validation_days_actual']}")
except Exception as e:
    print(f"❌ Error: {e}")

# TEST 2: Different day start hour
print("\n=== TEST 2: Different day start hour (midnight) ===")
try:
    train, val, info = get_train_validation_split(
        single_patient_data, num_validation_days=1, day_start_hour=0
    )
    print(f"✅ Success! Train: {len(train)} records, Validation: {len(val)} records")
    print(f"   Day start hour: {info['day_start_hour']}")
except Exception as e:
    print(f"❌ Error: {e}")

# TEST 3: Include partial days
print("\n=== TEST 3: Include partial days ===")
try:
    train, val, info = get_train_validation_split(
        single_patient_data, num_validation_days=1, include_partial_days=True
    )
    print(f"✅ Success! Train: {len(train)} records, Validation: {len(val)} records")
    print(f"   Include partial days: {info['include_partial_days']}")
except Exception as e:
    print(f"❌ Error: {e}")

# TEST 4: Error handling - no DatetimeIndex
print("\n=== TEST 4: Error handling - no DatetimeIndex ===")
try:
    df_no_datetime_index = single_patient_data.reset_index()
    train, val, info = get_train_validation_split(df_no_datetime_index)
    print("❌ Should have failed but didn't!")
except TypeError as e:
    print(f"✅ Correctly caught TypeError: {e}")
except Exception as e:
    print(f"❌ Wrong error type: {e}")

# TEST 5: Error handling - too many validation days
print("\n=== TEST 5: Error handling - too many validation days ===")
try:
    train, val, info = get_train_validation_split(
        single_patient_data, num_validation_days=100
    )
    print("❌ Should have failed but didn't!")
except ValueError as e:
    print(f"✅ Correctly caught ValueError: {e}")

# TEST 6: Detailed split info
print("\n=== TEST 6: Detailed split info ===")
try:
    train, val, info = get_train_validation_split(
        single_patient_data, num_validation_days=3
    )
    print("✅ Split info details:")
    for key, value in info.items():
        print(f"   {key}: {value}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🎉 All tests completed!")
