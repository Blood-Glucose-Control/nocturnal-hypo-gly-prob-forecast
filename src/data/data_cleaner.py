"""Data cleaning functions for the various datasets."""

from collections import defaultdict
import pandas as pd
from sktime.split import temporal_train_test_split
from sktime.transformations.series.impute import Imputer


def clean_data(
    data: pd.DataFrame, data_source_name="kaggle_brisT1D", data_type="train"
) -> pd.DataFrame:
    """
    Cleans the input data based on the specified data source name.

    Args:
        data (pd.DataFrame): The input data to be cleaned.
        data_source_name (str): The name of the data source. Default is "kaggle_brisT1D".

    Returns:
        pd.DataFrame: The cleaned data.
    """
    # leaving data_type in for now, because we may use test/train in the near future.
    if data_source_name == "kaggle_brisT1D":
        if data_type == "train":
            _clean_bris_data(data)
        else:
            raise NotImplementedError(
                "Use method clean_bris1d_test_data for test brist1d data"
            )

    return data


def clean_brist1d_test_data(df: pd.DataFrame):
    """
    Cleans the test data for the Bris1TD dataset by removing columns of historic data
    """
    patient_ids = df["p_num"].unique().tolist()
    all_value_var_lists = create_time_variable_lists()
    patient_dfs = defaultdict(dict)
    for patient_id in patient_ids:
        print("Patient ID: ", patient_id)
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
                        "value": val_var[0][:-4] + "value",
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

            patient_dfs[patient_id][row_id] = new_df

    return patient_dfs


def _transform_rows_to_timeseries(df: pd.DataFrame, patient_ids: list[str]) -> dict:
    """Given a patient id, transform each row into a timeseries dataframe

    Args:
        df (pd.DataFrame): Input dataframe in wide format
        patient_id (list): ID of patient to transform

    Returns:
        dict: dict of dicts containing dataframes, each containing timeseries data for a single patient
    """

    all_value_var_lists = create_time_variable_lists()
    patient_dfs = {}
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
                        "value": val_var[0][:-4] + "value",
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

            patient_dfs[patient_id][row_id] = new_df

    return patient_dfs


def create_time_variable_lists():
    """Creates lists of time variables for each measurement type (bg, insulin, etc)

    Returns:
        list: List of lists containing time variables for each measurement type
    """

    all_value_var_lists = []
    var_strs = ["bg-", "insulin-", "carbs-", "hr-", "steps-", "cals-", "activity-"]

    for var in var_strs:
        var_str = var
        var_list = []
        for hour in range(5, -1, -1):
            time_hour = var_str + str(hour)
            for minutes in range(55, -1, -5):
                if minutes < 10:
                    time = time_hour + ":0" + str(minutes)
                else:
                    time = time_hour + ":" + str(minutes)
                var_list.append(time)
        all_value_var_lists.append(var_list)
    return all_value_var_lists


def _clean_bris_data(data: pd.DataFrame):
    """
    Cleans the bris1TD Kaggle data with the following transformations:
        1. Deletes columns of historic data (eg: bg-5:55, ..., activity-5:55, ...) --> but does not remove -0:00 timestamp
    Args:
        data: the df for the Bris1TD dataset
    Mutations:
        Modifies the data in place
    """
    prefixes_to_check = ["activity", "bg", "cals", "insulin", "steps", "carbs", "hr"]

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


# this is not an appropriate helper fuction for this module, one it should be
# put into a sktime pipeline, but also this is being applied to
# sparse data. This does not make sense.
def handle_missing_values(data: pd.DataFrame, strategy="mean") -> pd.DataFrame:
    """
    Handles missing values in the DataFrame using sktime's Imputer.

    Args:
        data: Input DataFrame with possible missing values.
        strategy: Imputation strategy ('mean', 'median', 'constant', etc.).
    Returns:
        DataFrame with missing values handled.
    """
    # Create an Imputer instance
    imputer = Imputer(method=strategy)

    # Apply the imputer to all columns
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = imputer.fit_transform(data[col])

    return data


## AI GENERATED NOT TESTED
def downsample_missing_values(data: pd.DataFrame, freq="5T") -> pd.DataFrame:
    """
    Downsamples the data to the specified frequency and fills missing values.

    Args:
        data: Input DataFrame.
        freq: Frequency string ('5T', '10T', etc.).
    Returns:
        DataFrame with missing values handled.
    """
    # Convert the index to datetime
    data.index = pd.to_datetime(data.index)

    # Downsample the data
    data = data.resample(freq).mean()

    # Fill missing values with the mean
    data = data.fillna(data.mean())

    return data


# This will be deleted, not necesary, and hides import function.
def perform_train_test_split(df: pd.DataFrame, target_col="bg-0:00", test_size=0.2):
    """
    Splits the data into training and testing sets
    Args:
        df: the dataframe to split
        target_col: the column that you are trying to predict (i.e the "y" column)
        test_size: the size of the test data (0.0 - 1.0)
    Returns:
        y_train, y_test, X_train, X_test
    """
    y = df[target_col]
    x = df.drop(columns=[target_col])
    return temporal_train_test_split(y, x, test_size=test_size)


def melt_data(df: pd.DataFrame, id_vars: list, value_vars: list) -> pd.DataFrame:
    """
    Transforms column data into row data using pandas melt.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    id_vars (list): Column(s) to use as identifier variables.
    value_vars (list): Column(s) to unpivot.

    Returns:
    pd.DataFrame: The melted DataFrame.
    """
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
    return melted_df


# Example usage
if __name__ == "__main__":
    # Sample data
    example_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "p_num": [101, 102, 103],
            "bg": [120, 130, 140],
            "insulin": [10, 20, 30],
            "carbs": [50, 60, 70],
        }
    )

    id_variables = ["id", "p_num"]
    value_variables = ["bg", "insulin", "carbs"]

    melted_data = melt_data(example_df, id_variables, value_variables)
    print(melted_data)
