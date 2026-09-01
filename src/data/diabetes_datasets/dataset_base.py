# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Dataset handling base module.

This module provides the foundation for all dataset-related functionality through
the DatasetBase abstract class. It establishes a consistent interface for loading,
processing, and validating different types of datasets used throughout the project.

The module implements a standard two-phase data loading pattern:
1. Load raw data from source
2. Process raw data into a format suitable for analysis and modeling

This design promotes:
- Consistency across different dataset implementations
- Separation of concerns between data acquisition and processing
- Easy extensibility for new dataset types
- Data validation to prevent downstream errors

Example:
    To use this module, create a subclass of DatasetBase for your specific dataset:

    ```python
    from ..dataset_base import DatasetBase

    class MyCustomDataset(DatasetBase):
        @property
        def dataset_name(self):
            return "custom_dataset"

        def load_raw(self):
            # Implementation for loading raw data
            return raw_df

        def load_data(self):
            if self.raw_data is None:
                self.raw_data = self.load_raw()
            return self._process_raw_data()

        def _process_raw_data(self):
            # Implementation for processing the raw data
            return processed_df
    ```
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, TypeGuard, cast

import pandas as pd

ProcessedPatientDataFrames = dict[str, pd.DataFrame]

logger = logging.getLogger(__name__)


class DatasetBase(ABC):
    """Base class for dataset loading and processing.

    This abstract base class defines the interface for dataset handling classes
    in the nocturnal project. It provides a standardized way to load, process,
    and validate datasets. All dataset loaders should inherit from this class
    and implement its abstract methods.

    The class follows a two-phase data loading pattern:
    1. Load raw data using `load_raw()`
    2. Process that data into a usable form using `load_data()`

    Attributes:
        processed_data: Processed per-patient data keyed by patient ID.
        raw_data: The raw dataset without processing.
    """

    def __init__(self):
        self.processed_data: ProcessedPatientDataFrames | None = None
        self.raw_data: Any = None

    # ==================== Properties ====================
    @property
    @abstractmethod
    def dataset_name(self):
        """Get the name of the dataset.

        Returns:
            str: Name of the dataset, use: DatasetSourceType.[data_name].value
        """
        raise NotImplementedError("'dataset_name()' must be implemented by subclass")

    @property
    @abstractmethod
    def description(self):
        """Get the description of the dataset.

        Returns:
            str: Description of the dataset
        """
        raise NotImplementedError("'description()' must be implemented by subclass")

    @property
    def num_patients(self) -> int:
        """Get the number of patients in the dataset.

        Returns:
            int: The count of processed patients, or 0 if no data is loaded.
        """
        return len(self.processed_data) if isinstance(self.processed_data, dict) else 0

    @property
    def patient_ids(self) -> list[str]:
        """Get list of patient IDs in the dataset.

        Returns:
            list[str]: List of patient ID strings, or empty list if no data.
        """
        if not isinstance(self.processed_data, dict):
            return []
        return [str(patient_id) for patient_id in self.processed_data.keys()]

    @property
    def data_shape_summary(self) -> dict[str, tuple[int, int]]:
        """Get a summary of the data shape.

        Returns a dict mapping patient_id to shape tuple (rows, cols).

        Returns:
            dict[str, tuple[int, int]]: Patient-to-shape mapping.
                Empty dict if no processed data is loaded.
        """
        if not isinstance(self.processed_data, dict):
            return {}

        result: dict[str, tuple[int, int]] = {}
        for patient_id, patient_data in self.processed_data.items():
            if isinstance(patient_data, pd.DataFrame):
                result[str(patient_id)] = patient_data.shape
        return result

    @property
    def dataset_info(self) -> dict[str, object]:
        """Get high-level dataset metadata and summary statistics."""
        info: dict[str, object] = {
            "dataset_name": self.dataset_name,
            "num_patients": self.num_patients,
            "patient_ids": self.patient_ids,
            "timesteps_per_patient": {
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "median": 0.0,
                "total": 0,
            },
            "date_span": {"start": None, "end": None, "num_days": 0},
            "glucose_summary_mmol_l": {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0,
            },
        }

        if not self.processed_data:
            return info

        patient_lengths = pd.Series(
            [len(patient_df) for patient_df in self.processed_data.values()],
            dtype="int64",
        )
        info["timesteps_per_patient"] = {
            "min": int(patient_lengths.min()),
            "max": int(patient_lengths.max()),
            "mean": float(patient_lengths.mean()),
            "median": float(patient_lengths.median()),
            "total": int(patient_lengths.sum()),
        }

        all_datetime_indexes: list[pd.DatetimeIndex] = []
        glucose_series_list: list[pd.Series] = []
        for patient_df in self.processed_data.values():
            if isinstance(patient_df.index, pd.DatetimeIndex):
                patient_datetime = patient_df.index
            elif "datetime" in patient_df.columns:
                patient_datetime = pd.DatetimeIndex(
                    pd.to_datetime(patient_df["datetime"], errors="coerce")
                )
            else:
                patient_datetime = pd.DatetimeIndex([])

            patient_datetime = patient_datetime[~pd.isna(patient_datetime)]
            if len(patient_datetime) > 0:
                all_datetime_indexes.append(patient_datetime)

            if "bg_mM" in patient_df.columns:
                patient_bg = pd.to_numeric(
                    patient_df["bg_mM"], errors="coerce"
                ).dropna()
            elif "bg_mg_dl" in patient_df.columns:
                patient_bg = (
                    pd.to_numeric(patient_df["bg_mg_dl"], errors="coerce").dropna()
                    / 18.0
                )
            else:
                patient_bg = pd.Series(dtype="float64")

            if not patient_bg.empty:
                glucose_series_list.append(patient_bg)

        if all_datetime_indexes:
            all_datetimes = all_datetime_indexes[0]
            for patient_datetime in all_datetime_indexes[1:]:
                all_datetimes = all_datetimes.append(patient_datetime)

            span_start = all_datetimes.min()
            span_end = all_datetimes.max()
            span_days = int((span_end.normalize() - span_start.normalize()).days + 1)
            info["date_span"] = {
                "start": span_start,
                "end": span_end,
                "num_days": span_days,
            }

        if glucose_series_list:
            all_glucose = pd.concat(glucose_series_list, ignore_index=True)
            info["glucose_summary_mmol_l"] = {
                "mean": float(all_glucose.mean()),
                "std": float(all_glucose.std()),
                "min": float(all_glucose.min()),
                "max": float(all_glucose.max()),
                "count": int(all_glucose.count()),
            }

        data_metrics = getattr(self, "data_metrics", None)
        if data_metrics:
            info["metrics"] = data_metrics

        return info

    # ==================== Public Abstract Methods ====================
    @abstractmethod
    def load_raw(self):
        """Load the raw dataset without any processing.

        Returns:
            pd.DataFrame or pd.Series: The raw dataset
        """
        raise NotImplementedError("'load_raw()' must be implemented by subclass")

    # ==================== Public Methods ====================
    def load_data(self) -> None:
        """Load processed data, split train/validation data, and populate metadata.

        The default flow is:
        1. Try loading cached processed data when `use_cached=True`.
        2. If no cache is available, call the subclass `_process_and_cache_data()`.
        3. Split `processed_data` into train/validation dictionaries.

        Side Effects:
            Sets self.processed_data
        """
        need_to_process_data = True
        if getattr(self, "use_cached", False):
            cached_data = self._load_cached_processed_data()
            if cached_data is not None:
                filtered_cached_data = self._apply_keep_columns_filter(cached_data)
                self.processed_data = cast(
                    ProcessedPatientDataFrames, filtered_cached_data
                )
                need_to_process_data = False

        if need_to_process_data:
            self._process_and_cache_data()
            if self._is_processed_patient_data(self.processed_data):
                filtered_processed_data = self._apply_keep_columns_filter(
                    self.processed_data
                )
                self.processed_data = cast(
                    ProcessedPatientDataFrames, filtered_processed_data
                )

    def get_patient_dataframe(self, patient_id: str) -> pd.DataFrame | None:
        """Get processed data for a single patient.

        Args:
            patient_id: Patient identifier string.

        Returns:
            DataFrame for the patient, or None if not found.
        """
        if not self.processed_data:
            return None
        return self.processed_data.get(patient_id)

    def get_stacked_patient_dataframe(self) -> pd.DataFrame:
        """Combine all patients' data into a single DataFrame.

        Returns:
            Combined DataFrame with patient data indexed by (patient_id, datetime).

        Raises:
            ValueError: If no data available.
        """
        if not self.processed_data:
            raise ValueError("No processed data available.")

        return pd.concat(
            self.processed_data.values(),
            keys=self.processed_data.keys(),
            names=["patient_id", "datetime"],
        )

    def create_validation_table(self) -> pd.DataFrame:
        """Create a validation table for the dataset.

        This method extracts comprehensive statistics for each patient in the dataset,
        including temporal information, demographics, and physiological measurements.

        Date type detection:
        - Inspects each patient's datetime index to determine if dates are 'artificial' or 'real'
        - If patient's start date matches generic_patient_start_date exactly, marks as 'artificial'
        - If all timestamps fall within the same year as generic_patient_start_date, marks as 'artificial'
        - Otherwise marks as 'real'
        - If datetime cannot be determined, marks as 'unknown'

        Returns:
            pd.DataFrame: A DataFrame containing validation results with columns:
                - patient_id: Patient identifier
                - num_days: Number of unique days in patient data
                - num_data_points: Total number of data points (timestamps) for patient
                - start_date: First timestamp in patient data
                - end_date: Last timestamp in patient data
                - date_type: 'artificial', 'real', or 'unknown' based on datetime inspection
                - age: Patient age (if available)
                - sex: Patient sex (if available)
                - avg_bg_mM: Average blood glucose in mmol/L
                - min_bg_mM: Minimum blood glucose in mmol/L
                - max_bg_mM: Maximum blood glucose in mmol/L
                - avg_carbs_g: Average carbohydrate intake in grams
                - min_carbs_g: Minimum carbohydrate intake in grams
                - max_carbs_g: Maximum carbohydrate intake in grams
                - avg_insulin_units: Average insulin dose in units
                - min_insulin_units: Minimum insulin dose in units
                - max_insulin_units: Maximum insulin dose in units
        """
        if self.processed_data is None:
            raise ValueError("Processed data is not loaded. Call load_data() first.")
        if not isinstance(self.processed_data, dict):
            raise TypeError(
                "Processed data must be a dict-like patient map. Call load_data() first."
            )

        validation_rows = [
            self._extract_patient_stats(patient_id, patient_df)
            for patient_id, patient_df in self.processed_data.items()
            if isinstance(patient_df, pd.DataFrame) and not patient_df.empty
        ]

        return pd.DataFrame(validation_rows)

    # ==================== Protected Abstract Methods ====================
    @abstractmethod
    def _process_and_cache_data(self) -> ProcessedPatientDataFrames:
        """Process raw data and save/load it into self.processed_data."""
        raise NotImplementedError(
            "'_process_and_cache_data()' must be implemented by subclass"
        )

    # ==================== Protected Methods ====================
    def _is_processed_patient_data(
        self, data: ProcessedPatientDataFrames | None
    ) -> TypeGuard[ProcessedPatientDataFrames]:
        """Check whether data is in the standardized patient->DataFrame shape."""
        return isinstance(data, dict) and all(
            isinstance(patient_data, pd.DataFrame) for patient_data in data.values()
        )

    def _load_cached_processed_data(self) -> ProcessedPatientDataFrames | None:
        """Load processed patient data from cache when available."""
        cache_manager = getattr(self, "cache_manager", None)
        if cache_manager is None:
            raise ValueError(
                "Cache manager is not initialized. Child class must set self.cache_manager."
            )

        cached_data = cache_manager.load_processed_data(
            self.dataset_name,
            file_format="csv",
        )
        if cached_data is None:
            return None
        return cached_data

    def _apply_keep_columns_filter(
        self, patient_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Filter processed patient data to keep_columns while preserving datetime index."""
        keep_columns = getattr(self, "keep_columns", None)
        if keep_columns is None:
            return patient_data

        columns_to_keep = [column for column in keep_columns if column != "datetime"]
        if not columns_to_keep:
            return patient_data

        filtered_data: dict[str, pd.DataFrame] = {}
        for patient_id, patient_df in patient_data.items():
            available_cols = [
                col for col in columns_to_keep if col in patient_df.columns
            ]
            missing_cols = [
                col for col in columns_to_keep if col not in patient_df.columns
            ]
            if missing_cols:
                logger.warning(
                    "Patient %s missing requested columns %s; available=%s",
                    patient_id,
                    missing_cols,
                    list(patient_df.columns),
                )

            filtered_df = patient_df[available_cols] if available_cols else patient_df
            if patient_df.index.name != "datetime" and "datetime" in patient_df.columns:
                filtered_df = filtered_df.set_index(patient_df["datetime"])
                filtered_df.index.name = "datetime"
            filtered_data[patient_id] = filtered_df

        if filtered_data:
            first_patient_df = next(iter(filtered_data.values()))
            self.keep_columns = ["datetime", *first_patient_df.columns.tolist()]
        return filtered_data

    def _validate_data(self, data):
        """Validate the loaded data.

        Args:
            data (pd.DataFrame or pd.Series): Data to validate

        Returns:
            bool:True if data is valid, raises exception otherwise
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError("Data must be a pandas DataFrame or Series")
        if data.empty:
            raise ValueError("Dataset is empty")
        return True

    def _get_patient_datetime_index(
        self, patient_df: pd.DataFrame
    ) -> pd.DatetimeIndex | None:
        """Extract a valid datetime index for a patient DataFrame."""
        if isinstance(patient_df.index, pd.DatetimeIndex):
            datetime_index = patient_df.index
        elif "datetime" in patient_df.columns:
            try:
                datetime_index = pd.DatetimeIndex(
                    pd.to_datetime(patient_df["datetime"], errors="coerce")
                )
            except (TypeError, ValueError, AttributeError):
                return None
        else:
            return None

        datetime_index = datetime_index[~pd.isna(datetime_index)]
        if datetime_index.empty:
            return None
        return datetime_index

    def _determine_date_type(
        self, patient_df: pd.DataFrame, datetime_index: pd.DatetimeIndex | None = None
    ) -> str:
        """
        Determine whether patient datetimes are 'artificial' or 'real' through heuristic analysis.

        This method inspects the datetime index to determine authenticity by:
        1. Checking if the loader has a generic_patient_start_date attribute
        2. Comparing patient's earliest date with the generic start date
        3. Analyzing whether all dates fall within a single year matching the generic year

        Args:
            patient_df: Patient's DataFrame (should have datetime index or datetime column)

        Returns:
            str: One of 'artificial', 'real', or 'unknown'
                - 'artificial': Dates were synthetically generated from time-only data
                - 'real': Dates appear to be actual calendar dates from the dataset
                - 'unknown': Cannot determine (missing or invalid datetime information)

        Rules:
            - If no datetime information available -> 'unknown'
            - If loader has no generic_patient_start_date -> 'real'
            - If patient's start date exactly matches generic_patient_start_date -> 'artificial'
            - If all timestamps fall in same year as generic_patient_start_date -> 'artificial'
            - Otherwise -> 'real'
        """
        idx = (
            datetime_index
            if datetime_index is not None
            else self._get_patient_datetime_index(patient_df)
        )
        if idx is None:
            return "unknown"

        generic_date = getattr(self, "generic_patient_start_date", None)
        if generic_date is None:
            return "real"

        # Normalize for comparison
        try:
            idx_min = idx.min().normalize()
            gen_norm = pd.Timestamp(generic_date).normalize()
        except (TypeError, ValueError, AttributeError):
            return "Coould not complete date normalization for comparison."

        # Exact match to generic start date
        if idx_min == gen_norm:
            return "artificial"

        # If all timestamps fall in the same year and match generic year, consider artificial
        years = pd.Index(idx.year).dropna().unique()
        if len(years) == 1 and int(years[0]) == pd.Timestamp(generic_date).year:
            return "artificial"

        return "real"

    def _extract_patient_stats(self, patient_id: str, patient_df: pd.DataFrame) -> dict:
        """Extract statistics for a single patient.

        Args:
            patient_id: Patient identifier
            patient_df: Patient's DataFrame with datetime index

        Returns:
            dict: Dictionary containing patient statistics
        """
        # Ensure datetime index
        idx = self._get_patient_datetime_index(patient_df)
        if idx is None:
            return {"patient_id": patient_id, "error": "No datetime index available"}

        # Calculate temporal statistics
        num_days = idx.normalize().nunique()
        num_data_points = len(patient_df)
        start_date = idx.min()
        end_date = idx.max()

        # Determine date_type per patient using the already prepared datetime index
        date_type = self._determine_date_type(patient_df, datetime_index=idx)

        # Initialize stats dictionary
        stats = {
            "patient_id": patient_id,
            "num_days": num_days,
            "num_data_points": num_data_points,
            "start_date": start_date,
            "end_date": end_date,
            "date_type": date_type,
        }

        # Extract demographics (if available in DataFrame)
        for demo_col in ["age", "sex"]:
            if demo_col in patient_df.columns:
                # Get the most common value (mode) for this patient
                values = patient_df[demo_col].dropna()
                if not values.empty:
                    mode_values = values.mode()
                    stats[demo_col] = (
                        mode_values.iloc[0] if not mode_values.empty else values.iloc[0]
                    )
                else:
                    stats[demo_col] = None
            else:
                stats[demo_col] = None

        # Extract blood glucose statistics (bg_mM column)
        if "bg_mM" in patient_df.columns:
            bg_data = patient_df["bg_mM"].dropna()
            if not bg_data.empty:
                stats["avg_bg_mM"] = bg_data.mean()
                stats["min_bg_mM"] = bg_data.min()
                stats["max_bg_mM"] = bg_data.max()
            else:
                stats["avg_bg_mM"] = None
                stats["min_bg_mM"] = None
                stats["max_bg_mM"] = None
        else:
            stats["avg_bg_mM"] = None
            stats["min_bg_mM"] = None
            stats["max_bg_mM"] = None

        # Extract carbohydrate statistics (food_g column)
        if "food_g" in patient_df.columns:
            carbs_data = patient_df["food_g"].dropna()
            # Filter out zeros for min calculation to get actual carb intake events
            carbs_nonzero = carbs_data[carbs_data > 0]
            if not carbs_data.empty:
                stats["avg_carbs_g"] = carbs_data.mean()
                stats["min_carbs_g"] = (
                    carbs_nonzero.min() if not carbs_nonzero.empty else 0.0
                )
                stats["max_carbs_g"] = carbs_data.max()
            else:
                stats["avg_carbs_g"] = None
                stats["min_carbs_g"] = None
                stats["max_carbs_g"] = None
        else:
            stats["avg_carbs_g"] = None
            stats["min_carbs_g"] = None
            stats["max_carbs_g"] = None

        # Extract insulin statistics (dose_units column)
        if "dose_units" in patient_df.columns:
            insulin_data = patient_df["dose_units"].dropna()
            # Filter out zeros for min calculation to get actual insulin doses
            insulin_nonzero = insulin_data[insulin_data > 0]
            if not insulin_data.empty:
                stats["avg_insulin_units"] = insulin_data.mean()
                stats["min_insulin_units"] = (
                    insulin_nonzero.min() if not insulin_nonzero.empty else 0.0
                )
                stats["max_insulin_units"] = insulin_data.max()
            else:
                stats["avg_insulin_units"] = None
                stats["min_insulin_units"] = None
                stats["max_insulin_units"] = None
        else:
            stats["avg_insulin_units"] = None
            stats["min_insulin_units"] = None
            stats["max_insulin_units"] = None

        return stats

        # def _validate_brown_dataset(self) -> None:
        #     """Compute and store validation metrics for the dataset."""
        #     if not self.processed_data:
        #         self.data_metrics = {}
        #         return

        #     # Combine all data for statistics
        #     all_data = pd.concat(self.processed_data.values())

        #     self.data_metrics = {
        #         "total_rows": len(all_data),
        #         "unique_patients": len(self.processed_data),
        #         "patients_with_insulin": sum(
        #             1
        #             for df in self.processed_data.values()
        #             if ColumnNames.IOB.value in df.columns
        #             and df[ColumnNames.IOB.value].notna().any()
        #         ),
        #         "patients_cgm_only": sum(
        #             1
        #             for df in self.processed_data.values()
        #             if ColumnNames.IOB.value not in df.columns
        #             or df[ColumnNames.IOB.value].isna().all()
        #         ),
        #     }

        #     # Glucose statistics
        #     if ColumnNames.BG.value in all_data.columns:
        #         bg = all_data[ColumnNames.BG.value].dropna()
        #         self.data_metrics.update(
        #             {
        #                 "glucose_mean_mmol": round(bg.mean(), 2),
        #                 "glucose_std_mmol": round(bg.std(), 2),
        #                 "glucose_min_mmol": round(bg.min(), 2),
        #                 "glucose_max_mmol": round(bg.max(), 2),
        #             }
        #         )

        #     logger.info(f"Dataset validation: {self.data_metrics}")

        # def _validate_tam_dataset(self) -> None:
        #     """Validate the loaded dataset and compute quality metrics.

        #     Combines all patient data and computes validation metrics including
        #     total rows, unique patients, glucose statistics, and time-in-range
        #     percentages. Results are stored in self.data_metrics and logged.

        #     Side Effects:
        #         Sets self.data_metrics with computed validation metrics.
        #     """
        #     if not self.processed_data:
        #         logger.warning("No data to validate")
        #         return

        #     # Combine all patient data for overall metrics
        #     all_data = []
        #     for patient_df in self.processed_data.values():
        #         if isinstance(patient_df, pd.DataFrame):
        #             all_data.append(patient_df)

        #     if all_data:
        #         combined_df = pd.concat(all_data, ignore_index=False)
        #         self.data_metrics = validate_tamborlane_data(combined_df)

        #         logger.info("Dataset validation complete:")
        #         logger.info(f"  Total rows: {self.data_metrics.get('total_rows', 0)}")
        #         logger.info(
        #             f"  Unique patients: {self.data_metrics.get('unique_patients', 0)}"
        #         )
        #         if "glucose_mean" in self.data_metrics:
        #             logger.info(
        #                 f"  Mean glucose: {self.data_metrics['glucose_mean']:.2f} mmol/L"
        #             )
        #             logger.info(
        #                 f"  Std glucose: {self.data_metrics['glucose_std']:.2f} mmol/L"
        #             )
        #         elif "glucose_mean_mg_dl" in self.data_metrics:
        #             logger.info(
        #                 f"  Mean glucose: {self.data_metrics['glucose_mean_mg_dl']:.2f} mg/dL"
        #             )
        #             logger.info(
        #                 f"  Std glucose: {self.data_metrics['glucose_std_mg_dl']:.2f} mg/dL"
        #             )
        #         if "time_in_range" in self.data_metrics:
        #             logger.info(
        #                 f"  Time in range: {self.data_metrics['time_in_range']:.1f}%"
        #             )
        #             logger.info(
        #                 f"  Time below range: {self.data_metrics['time_below_range']:.1f}%"
        #             )
        #             logger.info(
        #                 f"  Time above range: {self.data_metrics['time_above_range']:.1f}%"
        #             )

        # def _validate_lynch_dataset(self) -> None:
        """
        Validate that each patient's processed data has required structure.

        Required columns:
            - cgm
            - bolus
            - carbs
            - exercise
            - basal
            - iob
            - cob
        """
        required_columns = {"iob", "cob"}
        if self.processed_data is None:
            raise ValueError("processed_data is not loaded.")

        for patient_id, patient_df in self.processed_data.items():
            if not isinstance(patient_df, pd.DataFrame):
                raise TypeError(
                    f"Patient {patient_id} processed data must be a DataFrame, got {type(patient_df)}"
                )

            missing_columns = required_columns - set(patient_df.columns)
            if missing_columns:
                raise ValueError(
                    f"Patient {patient_id} is missing required columns: {sorted(missing_columns)}"
                )
