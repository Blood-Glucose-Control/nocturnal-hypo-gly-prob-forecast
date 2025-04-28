import pandas as pd


# TODO: Move this to somewhere else (Don't think it is being used anywhere now but can be useful in the future)
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


# TODO: Remove the dependency of p_num. Kaggle data is the very few dataset where there are multiple patients in the same file.
def get_train_validation_split(
    df: pd.DataFrame,
    num_validation_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and validation sets based on complete days,
    where a day is defined as 6am-6am.

    Args:
        df (pd.DataFrame): Input dataframe with datetime column
        num_validation_days (int): Number of complete 6am-6am days to use for validation

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_data, validation_data) where validation
            contains exactly num_validation_days of complete 6am-6am days
    """
    if "datetime" not in df.columns:
        raise ValueError(
            "datetime column not found in data. Please run create_datetime_index first."
        )

    df["datetime"] = pd.to_datetime(df["datetime"])

    # For each patient:
    # 1. Find the last 6am timestamp
    # 2. Go back num_validation_days to get the start of validation
    # 3. Trim any data after the last 6am
    validation_data_list = []
    train_data_list = []

    for _, patient_df in df.groupby("p_num"):
        # Get timestamps where hour is 6 (6am)
        six_am_times = patient_df[patient_df["datetime"].dt.hour == 6]["datetime"]

        if len(six_am_times) == 0:
            continue

        # Get the last 6am timestamp
        last_six_am = six_am_times.max()

        # Calculate the start of validation period (num_validation_days before last_six_am)
        validation_start = last_six_am - pd.Timedelta(days=num_validation_days)

        # Split the patient's data
        patient_validation = patient_df[
            (patient_df["datetime"] >= validation_start)
            & (patient_df["datetime"] <= last_six_am)
        ]
        patient_train = patient_df[patient_df["datetime"] < validation_start]

        validation_data_list.append(patient_validation)
        train_data_list.append(patient_train)

    # Combine all patients' data
    validation_data = pd.concat(validation_data_list)
    train_data = pd.concat(train_data_list)

    return train_data, validation_data
