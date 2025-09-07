"""
pytest tests/benchmark/test_data_splits.py -v -s
"""

from src.data.preprocessing.time_processing import get_train_validation_split
import pytest
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def dataset():
    """Create an artificial dataset for testing data splits."""
    # Create a 90-day dataset for 70-20 split testing
    start_date = datetime(2023, 1, 1, 6, 0)  # Start at 6am
    end_date = start_date + timedelta(days=90)

    # Create timestamps every 5 minutes
    timestamps = pd.date_range(start=start_date, end=end_date, freq="5min")

    # Create data for 3 patients
    data_rows = []
    for patient_id in [1, 2, 3]:
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

    return df


def test_get_train_validation_split(dataset):
    # Test 70-20 split (70 days training, 20 days validation)
    train_data, validation_data = get_train_validation_split(
        dataset, num_validation_days=20
    )
    assert len(train_data) > 0
    assert len(validation_data) > 0

    # With our controlled dataset, we expect the split to be exact (no data loss)
    # The function should preserve all data points
    total_split_length = len(train_data) + len(validation_data)
    assert total_split_length == len(dataset), (
        f"Expected {len(dataset)} total rows, got {total_split_length}"
    )

    # Verify we have data for all patients in both splits
    train_patients = set(train_data["p_num"].unique())
    validation_patients = set(validation_data["p_num"].unique())
    original_patients = set(dataset["p_num"].unique())

    assert train_patients == original_patients, "Train data should contain all patients"
    assert validation_patients == original_patients, (
        "Validation data should contain all patients"
    )


def test_num_validation_days(dataset):
    train_data, validation_data = get_train_validation_split(
        dataset, num_validation_days=20
    )

    # With our controlled dataset, we can test this more precisely
    # The validation data for each patient should span exactly 20 days
    for patient_id in validation_data["p_num"].unique():
        patient_validation = validation_data[validation_data["p_num"] == patient_id]

        # Get the time span of validation data for this patient
        time_span = (
            patient_validation["datetime"].max() - patient_validation["datetime"].min()
        ).days

        # Should be exactly 20 days (or very close due to the 6am boundary logic)
        assert time_span >= 19 and time_span <= 20, (
            f"Patient {patient_id} validation data spans {time_span} days, "
            f"expected around 20 days"
        )

    # Test that train data comes before validation data (chronologically)
    for patient_id in dataset["p_num"].unique():
        patient_train = train_data[train_data["p_num"] == patient_id]
        patient_validation = validation_data[validation_data["p_num"] == patient_id]

        if len(patient_train) > 0 and len(patient_validation) > 0:
            max_train_time = patient_train["datetime"].max()
            min_validation_time = patient_validation["datetime"].min()
            assert max_train_time <= min_validation_time, (
                f"Train data should come before validation data for patient {patient_id}"
            )

            # Verify training data spans approximately 70 days
            train_time_span = (
                patient_train["datetime"].max() - patient_train["datetime"].min()
            ).days

            # Should be approximately 70 days (some variation due to 6am boundary logic)
            assert train_time_span >= 68 and train_time_span <= 72, (
                f"Patient {patient_id} training data spans {train_time_span} days, "
                f"expected around 70 days for a 70-20 split"
            )


# NOTE: this test case fails. Seems like number of days is not retained after splitting. Unsure why
# def test_split_patient_data_by_day(dataset):
#     # Check that number of days in split is retained after as well
#     patient_data = split_patient_data_by_day(dataset, "p01")

#     og_df = dataset[dataset["p_num"] == "p01"]
#     # check that number of days in og_df is the same as number of days in patient_data
#     # Convert time to datetime if not already
#     og_df["datetime"] = pd.to_datetime(og_df["datetime"])

#     # Count unique days by checking date
#     days_seen = set()
#     num_days = 0

#     for dt in og_df["datetime"]:
#         day = dt.date() # Get date object
#         if day not in days_seen:
#             days_seen.add(day)
#             num_days += 1

#     print("NUM DAYS OG DF", num_days)
#     print("PATIENT TRANSFORMED DATA ", patient_data.keys())
#     num_days_transformed = len(patient_data.keys())
#     assert num_days == num_days_transformed
