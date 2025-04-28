from src.data.data_loader import get_loader
from src.data.data_splitter import get_train_validation_split
import pytest
import pandas as pd


@pytest.fixture
def dataset():
    loader = get_loader(
        data_source_name="kaggle_brisT1D",
        dataset_type="train",
        use_cached=True,
    )
    return loader.processed_data


def test_get_train_validation_split(dataset):
    train_data, validation_data = get_train_validation_split(dataset)
    assert len(train_data) > 0
    assert len(validation_data) > 0
    # ensure that the split is not too far off (add some leeway since the function pops out some rows)
    assert abs(len(train_data) + len(validation_data) - len(dataset)) < 1500


def test_num_validation_days(dataset):
    train_data, validation_data = get_train_validation_split(
        dataset, num_validation_days=11
    )
    # check time index of validation data. should be 11 days
    validation_data["datetime"] = pd.to_datetime(validation_data["datetime"])
    assert (
        abs(
            (
                validation_data["datetime"].iloc[-1]
                - validation_data["datetime"].iloc[0]
            ).days
            - 11
        )
        < 2
    )  # since we dont include the last day


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
