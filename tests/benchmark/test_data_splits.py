"""
pytest tests/benchmark/test_data_splits.py -v -s
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data.preprocessing.time_processing import get_train_validation_split


@pytest.fixture
def dataset():
    """Create an artificial dataset for testing data splits."""
    start_date = datetime(2023, 1, 1, 6, 0)  # Start at 6am
    end_date = start_date + timedelta(days=30)

    # Create timestamps every 5 minutes
    timestamps = pd.date_range(start=start_date, end=end_date, freq="15min")

    # Create data for 1 patient
    data_rows = []
    for patient_id in [1]:
        for ts in timestamps:
            data_rows.append(
                {
                    "datetime": ts,
                    "p_num": patient_id,
                    "bg_mM": 5.5
                    + (patient_id * 0.5),  # Some dummy blood glucose values
                    "hr_bpm": 70 + (patient_id * 5),  # Some dummy heart rate values
                }
            )

    df = pd.DataFrame(data_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    return df


@pytest.fixture
def partial_day_dataset():
    """Create an artificial dataset for testing data splits."""
    start_date = datetime(2023, 1, 1, 6, 0)  # Start at 6am
    end_date = start_date + timedelta(
        days=30, hours=12
    )  # End at 6pm to create partial day

    # Create timestamps every 5 minutes
    timestamps = pd.date_range(start=start_date, end=end_date, freq="15min")

    # Create data for 1 patient
    data_rows = []
    for patient_id in [1]:
        for ts in timestamps:
            data_rows.append(
                {
                    "datetime": ts,
                    "p_num": patient_id,
                    "bg_mM": 5.5
                    + (patient_id * 0.5),  # Some dummy blood glucose values
                    "hr_bpm": 70 + (patient_id * 5),  # Some dummy heart rate values
                }
            )

    df = pd.DataFrame(data_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    return df


def test_get_train_validation_split_complete_days(dataset):
    """Test splitting with complete days only."""
    train_data, validation_data, meta_data = get_train_validation_split(
        dataset, num_validation_days=10, include_partial_days=False
    )

    assert len(train_data) > 0
    assert len(validation_data) > 0
    assert meta_data is not None

    # Check for no duplicate timestamps between train and validation
    train_timestamps = set(train_data.index)
    validation_timestamps = set(validation_data.index)
    overlap = train_timestamps.intersection(validation_timestamps)

    assert len(overlap) == 0, (
        f"Found {len(overlap)} overlapping timestamps between train and validation"
    )

    # Total should equal original (no data loss)
    total_split_length = len(train_data) + len(validation_data)
    assert total_split_length == len(dataset), (
        f"Expected {len(dataset)} total rows, got {total_split_length}"
    )

    print(f"Train data: {len(train_data)} rows")
    print(f"Validation data: {len(validation_data)} rows")
    print(f"Metadata: {meta_data}")


def test_get_train_validation_split_partial_days(partial_day_dataset):
    """Test splitting with partial days included."""
    # Test where partial days are included
    train_data, validation_data, meta_data = get_train_validation_split(
        partial_day_dataset, num_validation_days=10, include_partial_days=True
    )

    assert len(train_data) > 0
    assert len(validation_data) > 0

    # Check for no duplicate timestamps
    train_timestamps = set(train_data.index)
    validation_timestamps = set(validation_data.index)
    overlap = train_timestamps.intersection(validation_timestamps)

    assert len(overlap) == 0, f"Found {len(overlap)} overlapping timestamps"

    # Total should equal original
    total_split_length = len(train_data) + len(validation_data)
    assert total_split_length == len(partial_day_dataset)

    # Test where partial days are excluded
    train_data, validation_data, meta_data = get_train_validation_split(
        partial_day_dataset, num_validation_days=10, include_partial_days=False
    )
    assert len(train_data) > 0
    assert len(validation_data) > 0

    # Check for no duplicate timestamps
    train_timestamps = set(train_data.index)
    validation_timestamps = set(validation_data.index)
    overlap = train_timestamps.intersection(validation_timestamps)

    assert len(overlap) == 0, f"Found {len(overlap)} overlapping timestamps"

    total_split_length = (
        len(train_data)
        + len(validation_data)
        + meta_data.get("dropped_partial_day_records", 0)
    )
    assert total_split_length == len(partial_day_dataset)


def test_num_validation_days(dataset):
    """Test that validation period spans the correct number of days."""
    train_data, validation_data, meta_data = get_train_validation_split(
        dataset, num_validation_days=10
    )

    # Check validation data spans approximately 10 days
    validation_span_days = (
        validation_data.index.max() - validation_data.index.min()
    ).days
    assert 9 <= validation_span_days <= 10, (
        f"Validation data spans {validation_span_days} days, expected around 10"
    )

    # Test that train data comes before validation data chronologically
    max_train_time = train_data.index.max()
    min_validation_time = validation_data.index.min()

    assert max_train_time < min_validation_time, (
        "Train data should come before validation data chronologically"
    )

    # Check train data spans expected time
    train_span_days = (train_data.index.max() - train_data.index.min()).days
    expected_train_days = 30 - 10  # Total days minus validation days

    assert expected_train_days - 2 <= train_span_days <= expected_train_days + 2, (
        f"Train data spans {train_span_days} days, expected around {expected_train_days}"
    )


@pytest.mark.parametrize("dataset_fixture", ["dataset", "partial_day_dataset"])
def test_no_duplicate_timestamps(dataset_fixture, request):
    """Test that there are no duplicate timestamps between train and validation sets."""
    # Get the fixture by name
    dataset = request.getfixturevalue(dataset_fixture)

    train_data, validation_data, _ = get_train_validation_split(
        dataset, num_validation_days=5
    )

    # Check for overlapping timestamps
    train_timestamps = set(train_data.index)
    validation_timestamps = set(validation_data.index)
    overlap = train_timestamps.intersection(validation_timestamps)

    assert len(overlap) == 0, (
        f"Found {len(overlap)} duplicate timestamps between train and validation sets"
    )


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Create minimal dataset
    start_date = datetime(2023, 1, 1, 6, 0)
    timestamps = pd.date_range(start=start_date, periods=100, freq="15min")

    df = pd.DataFrame(
        {
            "p_num": [1] * 100,
            "bg_mM": [5.5] * 100,
        },
        index=timestamps,
    )

    # Test requesting too many validation days
    with pytest.raises(ValueError, match="Insufficient data"):
        get_train_validation_split(df, num_validation_days=30)

    # Test with non-DatetimeIndex
    df_no_datetime_index = df.reset_index()
    with pytest.raises(TypeError, match="DatetimeIndex"):
        get_train_validation_split(df_no_datetime_index, num_validation_days=1)
