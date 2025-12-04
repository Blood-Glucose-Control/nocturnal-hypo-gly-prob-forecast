"""
Data cleaning module for Tamborlane 2008 CGM dataset.

This module provides functions to clean and preprocess CGM data from the Tamborlane 2008 study.
The data format includes columns like RecID, PtID, DeviceDate, DeviceTime, GlucoseValue, etc.
"""

import logging
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def clean_tamborlane_2008_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform data from the Tamborlane 2008 dataset.

    This function processes the dataset by:
    1. Standardizing column names
    2. Converting glucose units if necessary
    3. Handling missing values
    4. Creating datetime columns
    5. Removing outliers and invalid measurements

    Args:
        df: Raw dataframe containing Tamborlane 2008 CGM data

    Returns:
        A cleaned DataFrame with standardized columns and processed measurements.
    """
    logger.info("Starting Tamborlane 2008 data cleaning process...")

    # Create a copy to avoid modifying the original
    data = df.copy()

    # Standardize column names based on the actual CGM format
    column_mapping = {
        # Map from actual column names to standardized names
        "RecID": "record_id",
        "PtID": "p_num",
        "DeviceDate": "device_date",
        "DeviceTime": "device_time",
        "GlucoseValue": "bg_mg_dl",
        "GlucoseDisplayTime": "display_time",
        # Legacy mappings for compatibility
        "Subject ID": "p_num",
        "Subject_ID": "p_num",
        "patient_id": "p_num",
        "Patient_ID": "p_num",
        "Time": "datetime",
        "timestamp": "datetime",
        "Timestamp": "datetime",
        "Date": "date",
        "Glucose": "bg_mg_dl",
        "glucose": "bg_mg_dl",
        "CGM": "bg_mg_dl",
        "cgm": "bg_mg_dl",
        "BG": "bg_mg_dl",
        "bg": "bg_mg_dl",
    }

    # Apply column name mapping
    data.rename(columns=column_mapping, inplace=True)

    # Create datetime column from DeviceDate and DeviceTime if available
    if "device_date" in data.columns and "device_time" in data.columns:
        # Combine date and time - use infer_datetime_format for flexibility
        # errors="coerce" ensures unparseable values become NaT instead of raising
        datetime_str = (
            data["device_date"].astype(str) + " " + data["device_time"].astype(str)
        )
        data["datetime"] = pd.to_datetime(datetime_str, errors="coerce")
        logger.info("Created datetime column from device_date and device_time")
    elif "display_time" in data.columns:
        # Use display time if available
        data["datetime"] = pd.to_datetime(data["display_time"], errors="coerce")
        logger.info("Created datetime column from display_time")
    elif "date" in data.columns and "time" in data.columns:
        # Combine date and time columns (legacy format)
        data["datetime"] = pd.to_datetime(
            data["date"].astype(str) + " " + data["time"].astype(str), errors="coerce"
        )

    # Ensure patient ID column exists and is string type
    if "p_num" in data.columns:
        data["p_num"] = data["p_num"].astype(str)
    else:
        logger.warning("No patient ID column found, will assign generic ID")
        data["p_num"] = "patient_001"

    # Convert glucose from mg/dL to mmol/L for consistency
    if "bg_mg_dl" in data.columns:
        # Handle potential string values and convert to numeric
        data["bg_mg_dl"] = pd.to_numeric(data["bg_mg_dl"], errors="coerce")
        data["bg_mM"] = (
            data["bg_mg_dl"] / 18.0182
        )  # Conversion factor from mg/dL to mmol/L
        logger.info("Converted glucose values from mg/dL to mmol/L")

    # Handle missing values
    # Remove rows where glucose is NaN or 0 (invalid readings)
    initial_rows = len(data)
    if "bg_mM" in data.columns:
        data = data[data["bg_mM"].notna() & (data["bg_mM"] > 0)]
    elif "bg_mg_dl" in data.columns:
        data = data[data["bg_mg_dl"].notna() & (data["bg_mg_dl"] > 0)]
    removed_rows = initial_rows - len(data)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows with invalid glucose readings")

    # Remove outliers (glucose values outside physiological range)
    # Normal range: 1.1 - 33.3 mmol/L (20 - 600 mg/dL)
    outliers_before = len(data)
    if "bg_mM" in data.columns:
        data = data[(data["bg_mM"] >= 1.1) & (data["bg_mM"] <= 33.3)]
    elif "bg_mg_dl" in data.columns:
        data = data[(data["bg_mg_dl"] >= 20) & (data["bg_mg_dl"] <= 600)]
    outliers_removed = outliers_before - len(data)
    if outliers_removed > 0:
        logger.info(f"Removed {outliers_removed} outlier glucose readings")

    # Remove invalid datetime entries
    if "datetime" in data.columns:
        invalid_dt = data["datetime"].isna().sum()
        if invalid_dt > 0:
            logger.warning(f"Found {invalid_dt} rows with invalid datetime values")
            data = data[data["datetime"].notna()]

    # Add additional columns that might be useful
    # Message type for compatibility with other datasets
    data["msg_type"] = "cgm"  # All readings are CGM in this dataset

    # Sort by patient and time
    if "p_num" in data.columns and "datetime" in data.columns:
        data = data.sort_values(["p_num", "datetime"])

    logger.info(f"Cleaning complete. Final dataset has {len(data)} rows")

    return data


def process_single_patient_tamborlane(
    patient_data_tuple: Tuple, store_intermediate_data: bool = False
) -> Tuple[str, pd.DataFrame]:
    """
    Process a single patient's data including datetime creation and preprocessing.

    Args:
        patient_data_tuple: Tuple containing (p_num, data, generic_patient_start_date)
        store_intermediate_data: Whether to save intermediate data to cache

    Returns:
        Tuple containing (p_num, processed_data)
    """
    p_num, data, generic_patient_start_date = patient_data_tuple
    logger.info(f"Processing Tamborlane patient {p_num} data...")

    # Create a copy to avoid modifying the original
    data_copy = data.copy()

    # Handle datetime column
    if "datetime" in data_copy.columns:
        # If datetime is already a proper datetime column
        if not pd.api.types.is_datetime64_any_dtype(data_copy["datetime"]):
            data_copy["datetime"] = pd.to_datetime(
                data_copy["datetime"], errors="coerce"
            )
    else:
        # Create datetime column if it doesn't exist
        logger.warning(
            f"No datetime column found for patient {p_num}, creating from index or row number"
        )
        # Assuming 5-minute intervals for CGM data if no time info available
        data_copy["datetime"] = pd.date_range(
            start=generic_patient_start_date, periods=len(data_copy), freq="5min"
        )

    # Set datetime as index
    if "datetime" in data_copy.columns:
        data_copy = data_copy.set_index("datetime", drop=False)  # Keep datetime column

    # Store intermediate data if requested
    if store_intermediate_data:
        from src.data.cache_manager import get_cache_manager

        cache_manager = get_cache_manager()
        dir_path = (
            cache_manager.get_cleaning_step_data_path("tamborlane_2008")
            / "datetime_index"
        )
        dir_path.mkdir(parents=True, exist_ok=True)
        data_copy.to_csv(dir_path / f"{p_num}.csv", index=True)

    # Run preprocessing pipeline if available
    try:
        from src.data.preprocessing.preprocessing_pipeline import preprocessing_pipeline

        processed_data = preprocessing_pipeline(p_num, data_copy)
    except ImportError:
        logger.warning("Preprocessing pipeline not available, returning data as-is")
        processed_data = data_copy

    return p_num, processed_data


def extract_cgm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract CGM-specific features from the glucose data.

    Args:
        df: DataFrame with glucose measurements

    Returns:
        DataFrame with additional CGM features
    """
    df = df.copy()

    # Calculate rate of change if we have regular time intervals
    if "bg_mM" in df.columns:
        # Calculate glucose rate of change (mmol/L per 5 minutes)
        df["glucose_roc"] = df["bg_mM"].diff()

        # Calculate rolling statistics (window-based features)
        # Using time-based windows for robustness
        if isinstance(df.index, pd.DatetimeIndex):
            df["glucose_1h_mean"] = df["bg_mM"].rolling("1h", center=True).mean()
            df["glucose_1h_std"] = df["bg_mM"].rolling("1h", center=True).std()
            df["glucose_3h_mean"] = df["bg_mM"].rolling("3h", center=True).mean()
            df["glucose_3h_std"] = df["bg_mM"].rolling("3h", center=True).std()
        else:
            # Fall back to fixed window sizes (12 periods = 1 hour at 5-min intervals)
            df["glucose_1h_mean"] = df["bg_mM"].rolling(12, center=True).mean()
            df["glucose_1h_std"] = df["bg_mM"].rolling(12, center=True).std()
            df["glucose_3h_mean"] = df["bg_mM"].rolling(36, center=True).mean()
            df["glucose_3h_std"] = df["bg_mM"].rolling(36, center=True).std()

        # Time in range calculations (3.9 - 10.0 mmol/L is typical target range)
        df["in_range"] = ((df["bg_mM"] >= 3.9) & (df["bg_mM"] <= 10.0)).astype(int)
        df["below_range"] = (df["bg_mM"] < 3.9).astype(int)
        df["above_range"] = (df["bg_mM"] > 10.0).astype(int)

        # Hypoglycemia and hyperglycemia flags
        df["hypo_mild"] = (df["bg_mM"] < 3.9).astype(int)  # < 70 mg/dL
        df["hypo_severe"] = (df["bg_mM"] < 3.0).astype(int)  # < 54 mg/dL
        df["hyper_mild"] = (df["bg_mM"] > 10.0).astype(int)  # > 180 mg/dL
        df["hyper_severe"] = (df["bg_mM"] > 13.9).astype(int)  # > 250 mg/dL

        # Glucose variability metrics
        df["glucose_cv"] = (
            df["glucose_1h_std"] / df["glucose_1h_mean"]
        )  # Coefficient of variation

        # Time-based features if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

            # Time of day categories
            df["time_of_day"] = pd.cut(
                df["hour"],
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                include_lowest=True,
            )

    return df


def validate_tamborlane_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate the cleaned Tamborlane dataset and return quality metrics.

    Args:
        df: Cleaned DataFrame

    Returns:
        Dictionary containing validation metrics
    """
    metrics = {}

    # Basic statistics
    metrics["total_rows"] = len(df)
    metrics["unique_patients"] = df["p_num"].nunique() if "p_num" in df.columns else 0

    # Data completeness
    if "bg_mM" in df.columns:
        metrics["missing_glucose"] = df["bg_mM"].isna().sum()
    elif "bg_mg_dl" in df.columns:
        metrics["missing_glucose"] = df["bg_mg_dl"].isna().sum()
    else:
        metrics["missing_glucose"] = 0

    metrics["missing_datetime"] = (
        df.index.isna().sum() if isinstance(df.index, pd.DatetimeIndex) else 0
    )

    # Glucose statistics
    if "bg_mM" in df.columns:
        metrics["glucose_mean"] = df["bg_mM"].mean()
        metrics["glucose_std"] = df["bg_mM"].std()
        metrics["glucose_min"] = df["bg_mM"].min()
        metrics["glucose_max"] = df["bg_mM"].max()

        # Time in range metrics
        if "in_range" in df.columns:
            metrics["time_in_range"] = df["in_range"].mean() * 100
            metrics["time_below_range"] = df["below_range"].mean() * 100
            metrics["time_above_range"] = df["above_range"].mean() * 100
    elif "bg_mg_dl" in df.columns:
        metrics["glucose_mean_mg_dl"] = df["bg_mg_dl"].mean()
        metrics["glucose_std_mg_dl"] = df["bg_mg_dl"].std()
        metrics["glucose_min_mg_dl"] = df["bg_mg_dl"].min()
        metrics["glucose_max_mg_dl"] = df["bg_mg_dl"].max()

    # Data frequency
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        time_diffs = df.index.to_series().diff()
        metrics["median_interval_minutes"] = time_diffs.median().total_seconds() / 60
        metrics["mean_interval_minutes"] = time_diffs.mean().total_seconds() / 60

    # Per-patient statistics
    if "p_num" in df.columns and ("bg_mM" in df.columns or "bg_mg_dl" in df.columns):
        glucose_col = "bg_mM" if "bg_mM" in df.columns else "bg_mg_dl"
        patient_stats = df.groupby("p_num").agg({glucose_col: ["count", "mean", "std"]})
        metrics["mean_readings_per_patient"] = patient_stats[
            (glucose_col, "count")
        ].mean()
        metrics["std_readings_per_patient"] = patient_stats[
            (glucose_col, "count")
        ].std()
        metrics["min_readings_per_patient"] = patient_stats[
            (glucose_col, "count")
        ].min()
        metrics["max_readings_per_patient"] = patient_stats[
            (glucose_col, "count")
        ].max()

    return metrics


def prepare_for_modeling(
    df: pd.DataFrame, lookback_hours: int = 4, prediction_horizon_hours: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the data for machine learning modeling with lookback features.

    Args:
        df: Cleaned DataFrame with glucose measurements
        lookback_hours: Number of hours to look back for features
        prediction_horizon_hours: Number of hours ahead to predict

    Returns:
        Tuple of (features_df, target_df)
    """
    df = df.copy()

    # Ensure we have datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise ValueError("DataFrame must have DatetimeIndex or datetime column")

    # Determine glucose column
    glucose_col = "bg_mM" if "bg_mM" in df.columns else "bg_mg_dl"

    # Create lagged features
    lookback_periods = int(lookback_hours * 12)  # Assuming 5-minute intervals
    for i in range(1, lookback_periods + 1):
        df[f"bg_lag_{i*5}min"] = df[glucose_col].shift(i)

    # Create target (future glucose value)
    prediction_periods = int(prediction_horizon_hours * 12)
    df["target"] = df[glucose_col].shift(-prediction_periods)

    # Add additional engineered features
    if "glucose_roc" in df.columns:
        # Include rate of change features
        for i in range(1, min(6, lookback_periods)):
            df[f"roc_lag_{i*5}min"] = df["glucose_roc"].shift(i)

    # Remove rows with NaN values from lagging
    df = df.dropna()

    # Separate features and target
    feature_cols = [
        col for col in df.columns if col.startswith(("bg_lag_", "roc_lag_", "glucose_"))
    ]

    # Add time-based features if available
    if "hour" in df.columns:
        feature_cols.extend(["hour", "day_of_week", "is_weekend"])

    features_df = df[feature_cols]
    target_df = df[["target"]]

    return features_df, target_df
