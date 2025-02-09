import pandas as pd
from data_loader import load_data
from data_cleaner import clean_data


def split_patient_data_by_day(patients_dfs: pd.DataFrame, patient_id: str) -> dict:
    """
    Split patient data into daily dataframes based on 6am transitions.

    Args:
        df: Input DataFrame containing patient data
        patient: Patient ID (e.g. 'p01')

    Returns:
        Dictionary of DataFrames, with keys as '{patient_id}_{day}' and values as daily DataFrames
    """
    # Convert time column to datetime and extract time and hour components
    patient_copy = patients_dfs[patients_dfs["p_num"] == patient_id].copy()

    patient_copy["time"] = pd.to_datetime(patient_copy["time"]).dt.strftime("%H:%M:%S")
    patient_copy["hour"] = pd.to_datetime(patient_copy["time"]).dt.hour

    # Initialize variables
    day_count = 0
    prev_hour = None
    patient_copy["day"] = 0

    # Traverse rows one by one
    for idx, row in patient_copy.iterrows():
        # Check if we transitioned from 5:xx to 6:xx
        if prev_hour == 5 and row["hour"] == 6:
            day_count += 1

        patient_copy.at[idx, "day"] = day_count
        prev_hour = row["hour"]

    patient_copy.drop(columns=["hour"], inplace=True)

    # Split data into dictionary of daily dataframes
    daily_dfs = {}
    for day in range(patient_copy["day"].max() + 1):
        daily_df = patient_copy[patient_copy["day"] == day].copy()
        # Drop id and p_num columns since they're in the key
        daily_df = daily_df.drop(columns=["id", "p_num", "day"])
        key = f"{patient_id}_{day}"
        daily_dfs[key] = daily_df

    return daily_dfs


if __name__ == "__main__":
    patient_id = "p01"
    df = clean_data(load_data(data_source_name="kaggle_brisT1D", dataset_type="train"))

    # Test the function with sample data
    daily_dfs = split_patient_data_by_day(df, "p01")

    # Print results
    print(f"Number of days: {len(daily_dfs)}")
    n = 3
    key = f"p01_{n}"
    print(f"\nDay {n} data:")
    print(daily_dfs[key])
