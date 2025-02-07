from src.data.data_loader import load_data
import pandas as pd


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


def transform_rows_to_timeseries(df: pd.DataFrame, patient_id: str):
    """Given a patient id, transform each row into a timeseries dataframe

    Args:
        df (pd.DataFrame): Input dataframe in wide format
        patient_id (str): ID of patient to transform

    Returns:
        list: List of dataframes, each containing timeseries data for a single patient
    """

    all_value_var_lists = create_time_variable_lists()
    patient_dfs = []
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
        patient_dfs.append(new_df)

    return patient_dfs


def main():
    df = load_data(dataset_type="test")
    dfs = transform_rows_to_timeseries(df, "p01")
    print(dfs[0])


if __name__ == "__main__":
    main()
