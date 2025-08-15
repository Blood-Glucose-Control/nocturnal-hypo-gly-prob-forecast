import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from src.data.preprocessing.generic_cleaning import clean_dataset
from src.data.preprocessing.feature_engineering import derive_features

logger = logging.getLogger(__name__)

required_columns = [
    "datetime",  # Datetime of the data (not the index)
    "p_num",  # Patient number (id)
    "bg_mM",  # Blood glucose in mmol/L
    "msg_type",  # Message type: ANNOUNCE_MEAL | ''
    "food_g",  # Carbs in grams
    "steps",  # Steps
    "dose_units",  # Insulin units
]

def process_single_patient(patient_data: pd.DataFrame) -> pd.DataFrame:
    """Process a single patient's data through the pipeline."""
    # Single patient processing
    p_num = patient_data['p_num'].iloc[0]
    logger.info(
        f"=============================="
    )
    logger.info(
        f"Processing single patient {p_num}"
    )
    logger.info(
        f"=============================="
    )    
    patient_df = patient_data.copy(deep=True)
    patient_df = clean_dataset(patient_df)
    return derive_features(patient_df)

def preprocessing_pipeline_parallel(df: pd.DataFrame, max_workers: int = 9) -> dict[str, pd.DataFrame]:
    """
    Parallel processing version of the preprocessing pipeline.
    Splits data by patient and processes each patient in parallel.
    Returns a dictionary mapping patient IDs to their processed DataFrames.
    """
    # Group by patient
    patient_groups = df.groupby('p_num')
    
    # Prepare data for parallel processing
    patient_data_list = [(p_num, group.copy()) for p_num, group in patient_groups]
    logger.info(f"Processing {len(patient_data_list)} patients in parallel:")
    for patient, _ in patient_data_list:
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
                logger.error(f'Patient {p_num} generated an exception: {exc}')
    
    return processed_results

def preprocessing_pipeline(df: pd.DataFrame, parallel: bool = True) -> dict[str, pd.DataFrame]:
    """
    The entry point for the preprocessing pipeline.
    This function does the following:
    1. Ensures datetime index
    2. Coerces timestamps to regular intervals
    3. Groups data by day starting at configured time
    4. Removes consecutive NaN values exceeding threshold
    5. Handles meal overlaps
    6. Keeps only top N carb meals
    7. Derive iob, cob, insulin availability and carb availability features
    
    Returns:
        Dictionary mapping patient IDs to their processed DataFrames
    """
    # TODO: Likely the clean_dataset function is creating bugs that makes the processed dataset be incorrect.
    # TODO: At this point we have multiple patients in the same file, we need to separate them.
    # TODO: Create an option for both serial and parallel processing of the multipatient files.
    check_data_format(df)
    
    # Check if we have multiple patients
    unique_patients = df['p_num'].nunique()
    if unique_patients > 1:
        logger.info(
            f"Processing {unique_patients} unique patients, using {'parallel' if parallel else 'serial'} processing..."
        )
        if parallel:
            return preprocessing_pipeline_parallel(df)
        else:
            # Process each patient separately in serial
            processed_results = {}
            for p_num, patient_data in df.groupby('p_num'):
                processed_patient = process_single_patient(patient_data)
                processed_results[p_num] = processed_patient
            return processed_results
    else:
        # Single patient processing - still return as dict
        p_num = df['p_num'].iloc[0]
        logger.info(
            f"=============================="
        )
        logger.info(
            f"Processing single patient {p_num}"
        )
        logger.info(
            f"=============================="
        )
        
        processed_df = process_single_patient(df)
        return {p_num: processed_df}


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
