# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Holdout manager for applying data splits according to holdout configurations.

This module provides utilities to split data into training and holdout sets
based on configured strategies, ensuring consistency across all experiments.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .holdout_config import (
    HoldoutConfig,
    HoldoutType,
)

logger = logging.getLogger(__name__)


class HoldoutManager:
    """Manages data splits according to holdout configurations."""

    def __init__(self, config: HoldoutConfig):
        """Initialize holdout manager with configuration.

        Args:
            config: Holdout configuration specifying split strategy
        """
        self.config = config
        self._train_patients: Optional[List[str]] = None
        self._holdout_patients: Optional[List[str]] = None
        self._split_metadata: Dict = {
            "skipped_patients": {},   # patient_id -> reason string
            "adjusted_patients": {},  # patient_id -> adjustment details
            "nan_p_num_filled": 0,    # count of NaN p_num values filled
        }

    def get_split_metadata(self) -> Dict:
        """Get metadata about split adjustments and skipped patients.

        Returns:
            Dict with keys:
                - skipped_patients: {patient_id: reason}
                - adjusted_patients: {patient_id: details}
                - nan_p_num_filled: count of NaN p_num values filled
        """
        return self._split_metadata

    def split_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        patient_col: str = "p_num",
        time_col: str = "time",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and holdout sets.

        Args:
            data: Either a multi-patient DataFrame or dict of patient DataFrames
            patient_col: Column name for patient IDs (if DataFrame)
            time_col: Column name for timestamps

        Returns:
            Tuple of (train_data, holdout_data)
        """
        # Convert to patient dictionary format if needed
        if isinstance(data, pd.DataFrame):
            patient_data = self._split_by_patient(data, patient_col)
        else:
            patient_data = data

        # Fill NaN patient IDs using the dict key (patient ID).
        # Raw data from resampling/time-alignment can contain gap-fill rows
        # where p_num and other original columns are NaN. Since each dict entry
        # is keyed by the known patient ID, we can safely fill these.
        if patient_col:
            n_filled_total = 0
            for patient_id, df in patient_data.items():
                if patient_col in df.columns:
                    n_nan = df[patient_col].isna().sum()
                    if n_nan > 0:
                        df[patient_col] = df[patient_col].fillna(patient_id)
                        n_filled_total += n_nan
            if n_filled_total > 0:
                logger.info(
                    f"Filled {n_filled_total:,} NaN {patient_col} values "
                    f"using patient dict keys (gap-fill rows from resampling)"
                )
                self._split_metadata["nan_p_num_filled"] = int(n_filled_total)

        # Apply holdout strategy
        if self.config.holdout_type == HoldoutType.TEMPORAL:
            return self._apply_temporal_split(patient_data, time_col)
        elif self.config.holdout_type == HoldoutType.PATIENT_BASED:
            return self._apply_patient_split(patient_data)
        elif self.config.holdout_type == HoldoutType.HYBRID:
            return self._apply_hybrid_split(patient_data, time_col)
        else:
            raise ValueError(f"Unknown holdout type: {self.config.holdout_type}")

    def _split_by_patient(
        self, data: pd.DataFrame, patient_col: str
    ) -> Dict[str, pd.DataFrame]:
        """Split multi-patient DataFrame into per-patient DataFrames."""
        unique_patients = data[patient_col].unique()
        return {
            str(patient_id): data[data[patient_col] == patient_id].copy()
            for patient_id in unique_patients
        }

    def _apply_temporal_split(
        self,
        patient_data: Dict[str, pd.DataFrame],
        time_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply temporal split to each patient's data.

        Holds out the last X% of each patient's time series.
        """
        config = self.config.temporal_config
        train_dfs = []
        holdout_dfs = []

        for patient_id, df in patient_data.items():
            # Ensure data is sorted by time
            # Handle both cases: time_col as index or as column
            if time_col in df.columns:
                df_sorted = df.sort_values(time_col)
            elif isinstance(df.index, pd.DatetimeIndex) or df.index.name == time_col:
                df_sorted = df.sort_index()
            else:
                # Try to find datetime column or index
                if "datetime" in df.columns:
                    df_sorted = df.sort_values("datetime")
                elif isinstance(df.index, pd.DatetimeIndex):
                    df_sorted = df.sort_index()
                else:
                    logger.warning(
                        f"Patient {patient_id}: Cannot find time column '{time_col}'. Using unsorted data."
                    )
                    df_sorted = df.copy()

            n_samples = len(df_sorted)

            # Calculate split point
            n_holdout = int(n_samples * config.holdout_percentage)
            n_train = n_samples - n_holdout

            # Validate minimum samples
            if n_train < config.min_train_samples:
                logger.warning(
                    f"Patient {patient_id} has only {n_train} training samples "
                    f"(min: {config.min_train_samples}). Skipping this patient."
                )
                self._split_metadata["skipped_patients"][patient_id] = (
                    f"Only {n_train} training samples (min: {config.min_train_samples}), "
                    f"total samples: {n_samples}"
                )
                continue

            if n_holdout < config.min_holdout_samples:
                logger.warning(
                    f"Patient {patient_id} has only {n_holdout} holdout samples "
                    f"(min: {config.min_holdout_samples}). Adjusting split."
                )
                self._split_metadata["adjusted_patients"][patient_id] = (
                    f"Holdout adjusted from {n_holdout} to {config.min_holdout_samples} samples "
                    f"(training reduced from {n_train} to {n_samples - config.min_holdout_samples}), "
                    f"total samples: {n_samples}"
                )
                n_holdout = config.min_holdout_samples
                n_train = n_samples - n_holdout

                # After expanding holdout, check if training still meets minimum
                if n_train < config.min_train_samples:
                    logger.warning(
                        f"Patient {patient_id}: after holdout adjustment, only {n_train} "
                        f"training samples remain (min: {config.min_train_samples}). "
                        f"Skipping this patient."
                    )
                    self._split_metadata["skipped_patients"][patient_id] = (
                        f"After holdout adjustment: only {n_train} training samples "
                        f"(min: {config.min_train_samples}), total samples: {n_samples}"
                    )
                    # Remove from adjusted since we're skipping entirely
                    self._split_metadata["adjusted_patients"].pop(patient_id, None)
                    continue

            # Split data
            train_df = df_sorted.iloc[:n_train].copy()
            holdout_df = df_sorted.iloc[n_train:].copy()

            train_dfs.append(train_df)
            holdout_dfs.append(holdout_df)

            logger.debug(
                f"Patient {patient_id}: {n_train} train samples, {n_holdout} holdout samples"
            )

        # Reset index to convert datetime from index to column before concatenation
        # This preserves datetime as a feature while avoiding index conflicts across patients
        train_dfs_reset = [df.reset_index() for df in train_dfs]
        holdout_dfs_reset = [df.reset_index() for df in holdout_dfs]

        # Combine all patients
        train_data = (
            pd.concat(train_dfs_reset, ignore_index=True)
            if train_dfs_reset
            else pd.DataFrame()
        )
        holdout_data = (
            pd.concat(holdout_dfs_reset, ignore_index=True)
            if holdout_dfs_reset
            else pd.DataFrame()
        )

        logger.info(
            f"Temporal split: {len(train_data):,} train samples, "
            f"{len(holdout_data):,} holdout samples from {len(train_dfs)} patients"
        )

        n_skipped = len(self._split_metadata["skipped_patients"])
        n_adjusted = len(self._split_metadata["adjusted_patients"])
        if n_skipped > 0:
            skipped_ids = list(self._split_metadata["skipped_patients"].keys())
            logger.info(
                f"  Skipped {n_skipped} patients (insufficient samples): {skipped_ids}"
            )
        if n_adjusted > 0:
            adjusted_ids = list(self._split_metadata["adjusted_patients"].keys())
            logger.info(
                f"  Adjusted split for {n_adjusted} patients "
                f"(holdout expanded to min): {adjusted_ids}"
            )

        return train_data, holdout_data

    def _apply_patient_split(
        self,
        patient_data: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply patient-based split.

        Holds out specific patients entirely from training.
        """
        config = self.config.patient_config
        all_patients = list(patient_data.keys())

        # Determine holdout patients
        if config.has_predefined_patients():
            # Use predefined patient list
            holdout_patients = [p for p in config.holdout_patients if p in all_patients]
            if len(holdout_patients) != len(config.holdout_patients):
                missing = set(config.holdout_patients) - set(holdout_patients)
                logger.warning(f"Some predefined holdout patients not found: {missing}")
        else:
            # Select patients randomly based on percentage
            if config.holdout_percentage is None:
                raise ValueError(
                    "Either holdout_patients or holdout_percentage must be specified"
                )

            n_holdout = max(
                config.min_holdout_patients,
                int(len(all_patients) * config.holdout_percentage),
            )

            # Use random seed for reproducibility
            rng = np.random.RandomState(config.random_seed)
            holdout_patients = rng.choice(
                all_patients, size=min(n_holdout, len(all_patients)), replace=False
            ).tolist()

        # Validate minimum patients
        train_patients = [p for p in all_patients if p not in holdout_patients]

        if len(train_patients) < config.min_train_patients:
            raise ValueError(
                f"Not enough training patients: {len(train_patients)} "
                f"(min: {config.min_train_patients})"
            )

        if len(holdout_patients) < config.min_holdout_patients:
            raise ValueError(
                f"Not enough holdout patients: {len(holdout_patients)} "
                f"(min: {config.min_holdout_patients})"
            )

        # Store patient lists for reference
        self._train_patients = train_patients
        self._holdout_patients = holdout_patients

        # Combine data
        # Reset index to convert datetime from index to column before concatenation
        # This preserves datetime as a feature while avoiding index conflicts across patients
        train_dfs = [patient_data[p].reset_index() for p in train_patients]
        holdout_dfs = [patient_data[p].reset_index() for p in holdout_patients]

        train_data = (
            pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        )
        holdout_data = (
            pd.concat(holdout_dfs, ignore_index=True) if holdout_dfs else pd.DataFrame()
        )

        # Format patient lists for display (show first 5)
        train_patients_display = train_patients[:5] + (
            ["..."] if len(train_patients) > 5 else []
        )
        holdout_patients_display = holdout_patients[:5] + (
            ["..."] if len(holdout_patients) > 5 else []
        )

        logger.info(
            f"Patient split: {len(train_patients)} train patients ({len(train_data):,} samples), "
            f"{len(holdout_patients)} holdout patients ({len(holdout_data):,} samples)"
        )
        logger.info(f"Train patients: {train_patients_display}")
        logger.info(f"Holdout patients: {holdout_patients_display}")

        return train_data, holdout_data

    def _apply_hybrid_split(
        self,
        patient_data: Dict[str, pd.DataFrame],
        time_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply hybrid split (both patient-based and temporal).

        First splits patients, then applies temporal split to training patients.
        Holdout patients are kept entirely separate.
        """
        # First apply patient split
        patient_config_backup = self.config.patient_config
        temp_config = HoldoutConfig(
            dataset_name=self.config.dataset_name,
            holdout_type=HoldoutType.PATIENT_BASED,
            patient_config=patient_config_backup,
        )
        temp_manager = HoldoutManager(temp_config)
        train_data_full, holdout_patients_data = temp_manager.split_data(patient_data)

        # Store holdout patients
        self._holdout_patients = temp_manager._holdout_patients
        self._train_patients = temp_manager._train_patients

        # Apply temporal split to training patients only
        train_patient_data = {p: patient_data[p] for p in self._train_patients}
        temporal_config_backup = self.config.temporal_config
        temp_config = HoldoutConfig(
            dataset_name=self.config.dataset_name,
            holdout_type=HoldoutType.TEMPORAL,
            temporal_config=temporal_config_backup,
        )
        temporal_temp_manager = HoldoutManager(temp_config)
        train_data, temporal_holdout_data = temporal_temp_manager.split_data(
            train_patient_data, time_col=time_col
        )

        # Propagate metadata from temporal split's temp manager to the parent
        self._split_metadata["skipped_patients"].update(
            temporal_temp_manager._split_metadata["skipped_patients"]
        )
        self._split_metadata["adjusted_patients"].update(
            temporal_temp_manager._split_metadata["adjusted_patients"]
        )

        # Combine temporal holdout with patient holdout
        # Both DataFrames should already have datetime as a column from their respective split methods
        holdout_data = pd.concat(
            [temporal_holdout_data, holdout_patients_data], ignore_index=True
        )

        logger.info(
            "Hybrid split: \n"
            + " " * 23
            + f"{len(train_data):,} train samples from {len(self._train_patients)} patients, "
            f"\n" + " " * 23 + f"{len(holdout_data):,} holdout timesteps "
            f"({len(temporal_holdout_data):,} temporal + {len(holdout_patients_data):,} patient-based)"
        )

        n_skipped = len(self._split_metadata["skipped_patients"])
        n_adjusted = len(self._split_metadata["adjusted_patients"])
        if n_skipped > 0:
            skipped_ids = list(self._split_metadata["skipped_patients"].keys())
            logger.info(
                f"  Skipped {n_skipped} patients from temporal split "
                f"(insufficient samples): {skipped_ids}"
            )
        if n_adjusted > 0:
            adjusted_ids = list(self._split_metadata["adjusted_patients"].keys())
            logger.info(
                f"  Adjusted split for {n_adjusted} patients "
                f"(holdout expanded to min): {adjusted_ids}"
            )

        return train_data, holdout_data

    def get_train_patients(self) -> Optional[List[str]]:
        """Get list of training patients (for patient-based or hybrid splits)."""
        return self._train_patients

    def get_holdout_patients(self) -> Optional[List[str]]:
        """Get list of holdout patients (for patient-based or hybrid splits)."""
        return self._holdout_patients

    def validate_split(
        self,
        train_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
        patient_col: str = "p_num",
    ) -> Dict[str, bool]:
        """Validate that split was performed correctly.

        Returns:
            Dict with validation results
        """
        results = {
            "no_overlap": True,
            "train_not_empty": len(train_data) > 0,
            "holdout_not_empty": len(holdout_data) > 0,
        }

        # Check for patient overlap based on split type
        if self.config.holdout_type == HoldoutType.TEMPORAL:
            # For temporal splits, patient overlap is EXPECTED (same patients, different time periods)
            # So we skip this check or mark it as passed
            results["no_overlap"] = True
            logger.debug("Temporal split: patient overlap is expected and allowed")

        elif self.config.holdout_type == HoldoutType.PATIENT_BASED:
            # For patient-based splits, NO overlap should exist
            if (
                patient_col in train_data.columns
                and patient_col in holdout_data.columns
            ):
                train_patients = set(train_data[patient_col].unique())
                holdout_patients = set(holdout_data[patient_col].unique())
                overlap = train_patients & holdout_patients

                if overlap:
                    logger.error(
                        f"Patient overlap detected in patient-based split: {overlap}"
                    )
                    results["no_overlap"] = False
            else:
                logger.warning(
                    f"Cannot validate patient overlap: '{patient_col}' column not found"
                )
                results["no_overlap"] = False

        elif self.config.holdout_type == HoldoutType.HYBRID:
            # For hybrid splits, check that HOLDOUT patients don't appear in training
            # (but training patients CAN appear in holdout due to temporal component)
            if self._holdout_patients and patient_col in train_data.columns:
                # Convert patient IDs to strings for comparison
                train_patients_str = set(
                    str(p) for p in train_data[patient_col].unique()
                )
                holdout_patients_str = set(self._holdout_patients)

                # Check if any of the designated holdout patients appear in training
                overlap = holdout_patients_str & train_patients_str

                if overlap:
                    logger.error(
                        f"Holdout patients found in training set (hybrid split): {overlap}"
                    )
                    results["no_overlap"] = False
            else:
                logger.warning(
                    "Cannot validate hybrid split: patient column or list not available"
                )
                results["no_overlap"] = False

        return results


def generate_patient_holdout_list(
    all_patients: List[str],
    holdout_percentage: float = 0.2,
    random_seed: int = 42,
    min_holdout: int = 1,
) -> List[str]:
    """Generate a random list of patients to hold out.

    Use this function once to generate the initial patient list,
    then save it to configuration for reproducibility.

    Args:
        all_patients: List of all patient IDs
        holdout_percentage: Percentage of patients to hold out
        random_seed: Random seed for reproducibility
        min_holdout: Minimum number of patients to hold out

    Returns:
        List of patient IDs to hold out
    """
    n_holdout = max(min_holdout, int(len(all_patients) * holdout_percentage))
    n_holdout = min(n_holdout, len(all_patients))  # Can't hold out more than available

    rng = np.random.RandomState(random_seed)
    holdout_patients = rng.choice(all_patients, size=n_holdout, replace=False).tolist()

    logger.info(
        f"Generated holdout list: {len(holdout_patients)} patients out of "
        f"{len(all_patients)} ({holdout_percentage*100:.1f}%)"
    )
    logger.info(f"Holdout patients: {holdout_patients}")

    return holdout_patients
