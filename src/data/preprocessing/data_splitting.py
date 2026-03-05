# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Module for splitting multi-patient data into single-patient data.
"""

import pandas as pd


def split_multipatient_dataframe(
    data: pd.DataFrame, patient_col: str = "p_num"
) -> dict[str, pd.DataFrame]:
    """
    Convert multi-patient data to single patient data.

    Args:
        data (pd.DataFrame): Multi-patient data, with p_num column.
        patient_col (str): Column name for patient IDs. Default is "p_num".

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their data.
    """
    unique_patient_ids = data[patient_col].unique()
    patient_data = {
        patient_id: data[data[patient_col] == patient_id].copy()
        for patient_id in unique_patient_ids
    }
    return patient_data
