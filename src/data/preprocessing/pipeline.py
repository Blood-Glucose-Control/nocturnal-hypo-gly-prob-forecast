import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.data.preprocessing.feature_engineering import derive_features
from src.data.preprocessing.generic_cleaning import ensure_datetime_index

logger = logging.getLogger(__name__)

required_columns = [
    "datetime",  # Datetime of the data (not the index)
    "p_num",  # Patient number (id)
    "bg_mM",  # Blood glucose in mmol/L
    "msg_type",  # Message type: ANNOUNCE_MEAL | ''
    "food_g",  # Carbs in grams
    "steps",  # Steps - TODO: probably not a required column... rmv
    "dose_units",  # Insulin units
]


def process_single_patient(patient_data: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: NOT DELETING UNTIL NEW PROCESS IS VERIFIED
    Process a single patient's data through the pipeline.
    """
    # Single patient processing
    p_num = patient_data["p_num"].iloc[0]
    logger.info("==============================")
    logger.info(f"Processing single patient {p_num}")
    logger.info("==============================")
    patient_df = patient_data.copy(deep=True)
    return derive_features(patient_df)
    # return patient_df


def preprocessing_pipeline_parallel(
    df: pd.DataFrame, max_workers: int = 9
) -> dict[str, pd.DataFrame]:
    """
    DEPRECATED: NOT DELETING UNTIL NEW PROCESS IS VERIFIED
    Parallel processing version of the preprocessing pipeline.
    Splits data by patient and processes each patient in parallel.
    Returns a dictionary mapping patient IDs to their processed DataFrames.
    """
    # Group by patient
    patient_groups = df.groupby("p_num")

    # Prepare data for parallel processing
    patient_data_list = [(p_num, group.copy()) for p_num, group in patient_groups]
    logger.info(f"Processing {len(patient_data_list)} patients in parallel:")
    for patient, _ in patient_data_list:
        # TODO: Make verbose=False shut this off in lager patient contexts.
        logger.info(f"\tProcessing patient: {patient}")

    processed_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_patient = {
            executor.submit(process_single_patient, data): p_num
            for p_num, data in patient_data_list
        }

        # Collect results
        for future in as_completed(future_to_patient):
            p_num = future_to_patient[future]
            try:
                result = future.result()
                processed_results[p_num] = result
            except Exception as exc:
                logger.error(f"Patient {p_num} generated an exception: {exc}")

    return processed_results


def preprocessing_pipeline(p_num: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    The entry point for the preprocessing pipeline.
    This function does the following:
    1. Ensures datetime index
    3. Groups data by day starting at configured time
    7. Derive iob, cob, insulin availability and carb availability features

    Returns:
        Dictionary mapping patient IDs to their processed DataFrames
    """
    # TODO: Likely the clean_dataset function is creating bugs that makes the processed dataset be incorrect.
    # TODO: At this point we have multiple patients in the same file, we need to separate them.
    # TODO: Create an option for both serial and parallel processing of the multipatient files.
    logger.info("==============================")
    logger.info(f"Preprocessing patient {p_num}")
    logger.info("==============================\n")

    check_data_format(df)
    patient_df = df.copy(deep=True)
    patient_df = ensure_datetime_index(patient_df)
    print(patient_df.index)
    processed_df = derive_features(patient_df)
    return processed_df


def check_data_format(df: pd.DataFrame) -> bool:
    """
    Checks if the data is in the correct format.
    """
    required_columns_set = set(required_columns)
    df_columns_set = set(df.columns)
    if not required_columns_set.issubset(df_columns_set):
        raise ValueError(
            f"Data is not in the correct format. Please make sure these columns are present: {required_columns_set - df_columns_set}"
        )
    return True
