import pandas as pd
from src.utils.kaggle_util import create_time_variable_lists
from collections import defaultdict


def clean_brist1d_test_data(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Cleans the test data for the Bris1TD dataset by removing columns of historic data
    """
    patient_ids = df["p_num"].unique().tolist()
    all_value_var_lists = create_time_variable_lists()
    patient_dfs = defaultdict(dict)
    for patient_id in patient_ids:
        for _, row in df[df["p_num"] == patient_id].iterrows():
            row_df = pd.DataFrame([row])  # Convert single row to DataFrame
            df_list = []
            for val_var in all_value_var_lists:
                temp_df = pd.melt(
                    row_df, id_vars=["id", "p_num", "time"], value_vars=val_var
                )
                temp_df = temp_df.rename(
                    columns={
                        "variable": val_var[0][:-4] + "time",
                        "value": val_var[0][:-4] + "0:00",
                    }
                )
                df_list.append(temp_df)

            bg_df = df_list[0]
            insulin_df = df_list[1]
            carbs_df = df_list[2]
            hr_df = df_list[3]
            steps_df = df_list[4]
            cals_df = df_list[5]
            activity_df = df_list[6]

            new_df = pd.concat(
                [
                    bg_df,
                    insulin_df.iloc[:, -1:],
                    carbs_df.iloc[:, -1:],
                    hr_df.iloc[:, -1:],
                    steps_df.iloc[:, -1:],
                    cals_df.iloc[:, -1:],
                    activity_df.iloc[:, -1:],
                ],
                axis=1,
            )

            # Convert time to datetime
            new_df["time"] = pd.to_datetime(new_df["time"])

            # Extract hours and minutes separately
            time_parts = new_df["bg-time"].str.extract(r"bg-(\d+):(\d+)")

            hours = pd.to_timedelta(time_parts[0].astype(int), unit="h")
            minutes = pd.to_timedelta(time_parts[1].astype(int), unit="m")
            total_hours = hours + minutes

            # Subtract offset from time and format to HH:MM:SS
            new_df["time"] = (new_df["time"] - total_hours).dt.strftime("%H:%M:%S")

            row_id = new_df["id"].iloc[0]
            # Drop the bg-time column
            new_df = new_df.drop("bg-time", axis=1)
            new_df = new_df.drop("p_num", axis=1)
            new_df = new_df.drop("id", axis=1)

            new_df["p_num"] = patient_id

            patient_dfs[patient_id][row_id] = new_df

    return patient_dfs

#TODO: Rename this funciton to something more descriptive
def clean_brist1d_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the brisT1D Kaggle data with the following transformations:
        1. Deletes columns of historic data (eg: bg-5:55, ..., activity-5:55, ...) --> but does not remove -0:00 timestamp

    Args:
        df: Raw DataFrame to clean

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    prefixes_to_check = ["activity", "bg", "cals", "insulin", "steps", "carbs", "hr"]

    # Create a copy to avoid modifying the original
    data = df.copy()

    # Create the list of columns to drop
    # Identify columns to drop based on the following conditions:
    # - The column name contains any of the specified prefixes.
    # - The column name does not end with "-0:00".
    columns_to_drop = [
        col
        for col in data.columns
        if any(prefix in col for prefix in prefixes_to_check)
        and not col.endswith("-0:00")
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    return data
