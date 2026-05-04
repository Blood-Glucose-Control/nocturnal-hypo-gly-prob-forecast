"""Functions for splitting or combining patient dataframes."""

import pandas as pd
from src.data.preprocessing.imputation import impute_missing_values
from src.utils.time_series_helper import get_interval_minutes
from src.utils.logging_helper import info_print, debug_print


def reduce_features_multi_patient(patients_dict, resolution_min, x_features, y_feature):
    """
    1. Select patients with the correct resolution
    2. Remove all unnecessary columns
    3. Impute missing values
    4. Add patient id
    5. Concatenate all patients
    """
    processed_patients = []

    for patient_id, df in patients_dict.items():
        # Check if patient has the correct interval
        interval_minutes = get_interval_minutes(df)
        debug_print(
            f"Checking patient {patient_id} with interval {interval_minutes} minutes..."
        )
        if interval_minutes == resolution_min:
            info_print(f"Processing patient {patient_id}...")
            # Process each patient individually
            p_df = df.iloc[:]
            p_df = p_df[x_features + y_feature]
            # Impute missing values for this patient
            p_df = impute_missing_values(p_df, columns=x_features)
            p_df = impute_missing_values(p_df, columns=y_feature)
            p_df["id"] = patient_id
            processed_patients.append(p_df)
        else:
            info_print(
                f"Skipping patient {patient_id} due to incorrect interval: {interval_minutes} minutes\n Needed: {resolution_min} minutes"
            )
    return pd.concat(processed_patients)
