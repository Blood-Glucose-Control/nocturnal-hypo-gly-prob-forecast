# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
Core utility functions for holdout configuration generation and validation.

This module provides reusable library functions for:
- Generating holdout configurations
- Validating holdout splits
- Analyzing dataset splits

These functions are used by CLI scripts but can also be imported
and used programmatically in workflows.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tabulate import tabulate

from ..diabetes_datasets.data_loader import get_loader
from .dataset_registry import DatasetRegistry
from .holdout_config import (
    HoldoutConfig,
    HoldoutType,
    PatientHoldoutConfig,
    TemporalHoldoutConfig,
)
from .holdout_manager import generate_patient_holdout_list

logger = logging.getLogger(__name__)


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================


def get_patient_ids_from_dataset(dataset_name: str) -> List[str]:
    """Load dataset and extract unique patient IDs.

    Args:
        dataset_name: Name of dataset to load

    Returns:
        List of unique patient IDs as strings
    """
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        loader = get_loader(dataset_name, use_cached=True)  # type: ignore[arg-type]
        data = loader.processed_data

        # Handle different data formats
        if isinstance(data, dict):
            # Dictionary format: {patient_id: DataFrame, ...}
            patient_ids = list(data.keys())
            logger.info(f"Data is dict format with {len(patient_ids)} patients")

        elif isinstance(data, pd.DataFrame):
            # Single DataFrame with patient column
            if "p_num" in data.columns:
                patient_ids = data["p_num"].unique().tolist()
                logger.info(
                    f"Data is DataFrame with 'p_num' column, {len(patient_ids)} unique patients"
                )
            elif "patient_id" in data.columns:
                patient_ids = data["patient_id"].unique().tolist()
                logger.info(
                    f"Data is DataFrame with 'patient_id' column, {len(patient_ids)} unique patients"
                )
            else:
                logger.warning(
                    f"No patient column found in {dataset_name}, treating as single patient"
                )
                patient_ids = ["patient_1"]
        else:
            logger.warning(
                f"Unknown data format for {dataset_name}: {type(data)}, treating as single patient"
            )
            patient_ids = ["patient_1"]

        # Convert all patient IDs to strings for consistency
        patient_ids = [str(pid) for pid in patient_ids]

        logger.info(f"Found {len(patient_ids)} patients in {dataset_name}")
        return patient_ids

    except Exception as e:
        logger.error(f"Error loading {dataset_name}: {e}")
        return []


def create_hybrid_holdout_config(
    dataset_name: str,
    patient_ids: List[str],
    temporal_pct: float,
    patient_pct: float,
    seed: int,
    min_train_samples: int = 608,
    min_holdout_samples: int = 608,
    min_train_patients: int = 10,
    min_holdout_patients: int = 10,
) -> HoldoutConfig:
    """Create hybrid holdout configuration with both temporal and patient splits.

    Args:
        dataset_name: Name of the dataset
        patient_ids: List of all patient IDs in the dataset
        temporal_pct: Temporal holdout percentage (e.g., 0.05 for 5%)
        patient_pct: Patient holdout percentage (e.g., 0.05 for 5%)
        seed: Random seed for patient selection
        min_train_samples: Minimum samples per patient in training
        min_holdout_samples: Minimum samples per patient in holdout
        min_train_patients: Minimum number of patients in training
        min_holdout_patients: Minimum number of patients in holdout
    Returns:
        HoldoutConfig with hybrid strategy
    """
    # Generate patient holdout list
    holdout_patients = generate_patient_holdout_list(
        all_patients=patient_ids,
        holdout_percentage=patient_pct,
        random_seed=seed,
    )

    # Create configs
    temporal_config = TemporalHoldoutConfig(
        holdout_percentage=temporal_pct,
        min_train_samples=min_train_samples,
        min_holdout_samples=min_holdout_samples,
    )

    patient_config = PatientHoldoutConfig(
        holdout_patients=holdout_patients,
        holdout_percentage=patient_pct,
        min_train_patients=min_train_patients,
        min_holdout_patients=min_holdout_patients,
        random_seed=seed,
    )

    config = HoldoutConfig(
        dataset_name=dataset_name,
        holdout_type=HoldoutType.HYBRID,
        temporal_config=temporal_config,
        patient_config=patient_config,
        description=(
            f"Hybrid holdout strategy: {temporal_pct * 100:.0f}% temporal split "
            f"+ {patient_pct * 100:.0f}% patient holdout. "
            f"Holdout patients: {len(holdout_patients)}. "
            f"Fixed seed={seed} for reproducibility."
        ),
        created_date=datetime.now().isoformat(),
        version="1.0",
    )

    return config


def generate_holdout_configs_for_datasets(
    datasets: List[str],
    output_dir: Path,
    temporal_pct: float = 0.05,
    patient_pct: float = 0.05,
    seed: int = 42,
    min_train_samples: int = 608,
    min_holdout_samples: int = 608,
    min_train_patients: int = 5,
    min_holdout_patients: int = 5,
) -> Dict[str, Any]:
    """Generate holdout configurations for multiple datasets.

    Args:
        datasets: List of dataset names
        output_dir: Directory to save config files
        temporal_pct: Temporal holdout percentage
        patient_pct: Patient holdout percentage
        seed: Random seed for reproducibility
        min_train_samples: Minimum samples per patient in training
        min_holdout_samples: Minimum samples per patient in holdout
        min_train_patients: Minimum number of patients in training
        min_holdout_patients: Minimum number of patients in holdout

    Returns:
        Dictionary with 'success' mapping dataset names to success status and
        'generated_files' list of paths to generated config files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    generated_files = []

    for dataset_name in datasets:
        try:
            logger.info("\n")
            logger.info(f"{'=' * 60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'=' * 60}")

            # Get patient IDs
            patient_ids = get_patient_ids_from_dataset(dataset_name)
            if not patient_ids:
                logger.error(f"No patients found for {dataset_name}")
                results[dataset_name] = False
                continue

            # Create hybrid config
            config = create_hybrid_holdout_config(
                dataset_name=dataset_name,
                patient_ids=patient_ids,
                temporal_pct=temporal_pct,
                patient_pct=patient_pct,
                seed=seed,
                min_train_samples=min_train_samples,
                min_holdout_samples=min_holdout_samples,
                min_train_patients=min_train_patients,
                min_holdout_patients=min_holdout_patients,
            )

            # Save config
            config_path = output_dir / f"{dataset_name}.yaml"
            config.save(str(config_path))
            logger.info(f"✓ Saved configuration to: {config_path}")

            results[dataset_name] = True
            generated_files.append(str(config_path))

        except Exception as e:
            logger.error(f"✗ Failed to process {dataset_name}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            results[dataset_name] = False

    return {"success": results, "generated_files": generated_files}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def _validate_patient_split_checks(
    *,
    config: HoldoutConfig,
    train_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    results: Dict[str, Any],
    verbose: bool,
) -> None:
    """Validate patient/hybrid split constraints and leakage semantics."""
    if verbose:
        logger.info("")
        logger.info(f"Validating patient holdout split for: {config.dataset_name}")
    if config.holdout_type not in (HoldoutType.PATIENT_BASED, HoldoutType.HYBRID):
        raise ValueError(
            "_validate_patient_split_checks called for non-patient holdout type: "
            f"{config.holdout_type.value}"
        )

    has_train_patient_col = "p_num" in train_data.columns
    has_holdout_patient_col = "p_num" in holdout_data.columns

    if has_train_patient_col and has_holdout_patient_col:
        train_patients = sorted(str(p) for p in train_data["p_num"].dropna().unique())
        holdout_patients = sorted(
            str(p) for p in holdout_data["p_num"].dropna().unique()
        )
        results["train_patients"] = train_patients
        results["holdout_patients"] = holdout_patients

        if verbose:
            logger.info(f"  - Training patients: {len(train_patients)}")
            logger.info(f"  - Holdout patients: {len(holdout_patients)}")

        overlap = set(train_patients) & set(holdout_patients)
        if config.holdout_type == HoldoutType.PATIENT_BASED:
            if overlap:
                results["errors"].append(
                    f"Patient overlap detected: {len(overlap)} patients"
                )
                if verbose:
                    logger.error(f"✗ Patient overlap detected: {len(overlap)} patients")
            else:
                results["no_data_leakage"] = True
                if verbose:
                    logger.info("✓ No patient overlap (patient-based split)")
        elif config.holdout_type == HoldoutType.HYBRID:
            designated_holdout = (
                set(str(p) for p in config.patient_config.holdout_patients)
                if config.patient_config is not None
                else set()
            )
            leak = designated_holdout & set(train_patients)
            if leak:
                results["errors"].append(
                    f"Designated holdout patients in training: {leak}"
                )
                if verbose:
                    logger.error(f"✗ Designated holdout patients leaked: {leak}")
            else:
                results["no_data_leakage"] = True
                if verbose:
                    logger.info(
                        f"✓ Hybrid split valid: {len(overlap)} patients overlap (temporal)"
                    )
    elif has_train_patient_col != has_holdout_patient_col:
        missing_side = "holdout" if has_train_patient_col else "train"
        results["errors"].append(
            f"Inconsistent split columns: 'p_num' missing from {missing_side} data"
        )
        if verbose:
            logger.error(
                f"✗ Inconsistent split columns: 'p_num' missing from {missing_side} data"
            )
    else:
        results["errors"].append(
            "Invalid split columns: patient/hybrid holdout requires "
            "'p_num' in both splits"
        )
        if verbose:
            logger.error(
                "✗ Invalid split columns: patient/hybrid holdout requires "
                "'p_num' in both splits"
            )

    if (
        config.patient_config is not None
        and has_train_patient_col
        and has_holdout_patient_col
    ):
        min_train_patients = config.patient_config.min_train_patients
        min_holdout_patients = config.patient_config.min_holdout_patients
        train_patient_count = int(train_data["p_num"].dropna().nunique())
        holdout_patient_count = int(holdout_data["p_num"].dropna().nunique())

        if train_patient_count < min_train_patients:
            results["errors"].append(
                f"Training patient count below configured minimum: "
                f"{train_patient_count} < {min_train_patients}"
            )
            if verbose:
                logger.warning(
                    "⚠ Training patient count below configured minimum: "
                    f"{train_patient_count} < {min_train_patients}"
                )

        if (
            config.holdout_type == HoldoutType.PATIENT_BASED
            and holdout_patient_count < min_holdout_patients
        ):
            results["errors"].append(
                f"Holdout patient count below configured minimum: "
                f"{holdout_patient_count} < {min_holdout_patients}"
            )
            if verbose:
                logger.warning(
                    "⚠ Holdout patient count below configured minimum: "
                    f"{holdout_patient_count} < {min_holdout_patients}"
                )

    return None


def _validate_temporal_checks(
    *,
    config: HoldoutConfig,
    train_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    has_train_patient_col: bool,
    has_holdout_patient_col: bool,
    results: Dict[str, Any],
    verbose: bool,
) -> None:
    """Validate temporal thresholds and ordering constraints."""
    if verbose:
        logger.info("")
        logger.info(f"Validating temporal holdout split for: {config.dataset_name}")
    if config.holdout_type not in (HoldoutType.TEMPORAL, HoldoutType.HYBRID):
        raise ValueError(
            "_validate_temporal_checks called for non-temporal holdout type: "
            f"{config.holdout_type.value}"
        )
    if config.temporal_config is None:
        raise ValueError(
            f"{config.holdout_type.value} holdout missing required temporal_config"
        )

    if has_train_patient_col != has_holdout_patient_col:
        if config.holdout_type == HoldoutType.TEMPORAL:
            missing_side = "holdout" if has_train_patient_col else "train"
            results["errors"].append(
                f"Inconsistent split columns: 'p_num' missing from {missing_side} data"
            )
            if verbose:
                logger.error(
                    "✗ Inconsistent split columns: "
                    f"'p_num' missing from {missing_side} data"
                )
        return

    if config.holdout_type == HoldoutType.TEMPORAL:
        if has_train_patient_col and has_holdout_patient_col:
            train_patients = sorted(
                str(p) for p in train_data["p_num"].dropna().unique()
            )
            holdout_patients = sorted(
                str(p) for p in holdout_data["p_num"].dropna().unique()
            )
            results["train_patients"] = train_patients
            results["holdout_patients"] = holdout_patients
            results["no_data_leakage"] = True
            if verbose:
                overlap = set(train_patients) & set(holdout_patients)
                logger.info(f"  - Training patients: {len(train_patients)}")
                logger.info(f"  - Holdout patients: {len(holdout_patients)}")
                logger.info(
                    f"✓ Temporal split: {len(overlap)} patients overlap (expected)"
                )
        else:
            results["no_data_leakage"] = True
            if verbose:
                logger.info(
                    "✓ No patient identifier column; using temporal-only split validation"
                )

    min_train_samples = config.temporal_config.min_train_samples
    min_holdout_samples = config.temporal_config.min_holdout_samples

    if results["train_sample_count"] < min_train_samples:
        results["errors"].append(
            f"Training set below configured minimum: "
            f"{results['train_sample_count']} < {min_train_samples}"
        )
        if verbose:
            logger.warning(
                "⚠ Training set below configured minimum: "
                f"{results['train_sample_count']} < {min_train_samples}"
            )

    if results["holdout_sample_count"] < min_holdout_samples:
        results["errors"].append(
            f"Holdout set below configured minimum: "
            f"{results['holdout_sample_count']} < {min_holdout_samples}"
        )
        if verbose:
            logger.warning(
                "⚠ Holdout set below configured minimum: "
                f"{results['holdout_sample_count']} < {min_holdout_samples}"
            )

    if "datetime" not in train_data.columns or "datetime" not in holdout_data.columns:
        results["errors"].append(
            "Cannot verify temporal ordering: missing required 'datetime' column "
            "in train and/or holdout split"
        )
        if verbose:
            logger.warning(
                "⚠ Cannot verify temporal ordering: missing required 'datetime' "
                "column in train and/or holdout split"
            )
        return

    try:
        train_time = pd.to_datetime(train_data["datetime"])
        holdout_time = pd.to_datetime(holdout_data["datetime"])
    except Exception as exc:
        results["errors"].append(f"Could not verify temporal ordering: {exc}")
        if verbose:
            logger.warning(f"⚠ Could not verify temporal ordering: {exc}")
        return

    if has_train_patient_col and has_holdout_patient_col:
        train_by_patient = (
            pd.DataFrame({"p_num": train_data["p_num"], "datetime": train_time})
            .dropna(subset=["p_num", "datetime"])
            .groupby("p_num", dropna=True)["datetime"]
            .max()
            .rename("max_train_datetime")
        )
        holdout_by_patient = (
            pd.DataFrame({"p_num": holdout_data["p_num"], "datetime": holdout_time})
            .dropna(subset=["p_num", "datetime"])
            .groupby("p_num", dropna=True)["datetime"]
            .min()
            .rename("min_holdout_datetime")
        )
        temporal_alignment = train_by_patient.to_frame().join(
            holdout_by_patient.to_frame(), how="inner"
        )
        temporal_issues = temporal_alignment.index[
            temporal_alignment["max_train_datetime"]
            >= temporal_alignment["min_holdout_datetime"]
        ].tolist()

        if temporal_issues:
            results["errors"].append(f"Temporal ordering issue: {temporal_issues}")
            if verbose:
                logger.error(
                    f"✗ Temporal ordering issue for patients: {temporal_issues}"
                )
        elif verbose:
            logger.info("✓ Temporal ordering correct for all patients")
    else:
        if train_time.max() >= holdout_time.min():
            results["errors"].append(
                "Temporal ordering issue: train extends into holdout period"
            )
            if verbose:
                logger.error("✗ Temporal ordering issue")
        elif verbose:
            logger.info("✓ Temporal ordering correct")


def _log_dataset_runtime_info(dataset_info: Dict[str, object]) -> None:
    """Log loader-provided runtime dataset summary."""

    def as_dict(value: object) -> Dict[str, object]:
        return value if isinstance(value, dict) else {}

    def fmt_int(value: object) -> str:
        return f"{int(value):,}" if isinstance(value, (int, float)) else "N/A"

    def fmt_float(value: object, decimals: int = 2) -> str:
        return (
            f"{float(value):,.{decimals}f}"
            if isinstance(value, (int, float))
            else "N/A"
        )

    patient_ids_raw = dataset_info.get("patient_ids", [])
    patient_ids: List[str] = (
        [str(patient_id) for patient_id in patient_ids_raw]
        if isinstance(patient_ids_raw, list)
        else []
    )
    timesteps = as_dict(dataset_info.get("timesteps_per_patient", {}))
    date_span = as_dict(dataset_info.get("date_span", {}))
    glucose = as_dict(dataset_info.get("glucose_summary_mmol_l", {}))

    patient_preview = patient_ids[:5]
    hidden_count = len(patient_ids) - len(patient_preview)
    patient_ids_display = (
        f"{patient_preview} (+{hidden_count} more)"
        if hidden_count > 0
        else str(patient_preview)
    )

    logger.info("")
    logger.info("Dataset Runtime Summary:")
    logger.info(f"  Dataset: {dataset_info.get('dataset_name')}")
    logger.info(f"  Number of patients: {dataset_info.get('num_patients')}")
    logger.info(f"  Patient IDs (first 5): {patient_ids_display}")
    logger.info(
        "  Timesteps per patient: min=%s, max=%s, mean=%s, median=%s, total=%s",
        fmt_int(timesteps.get("min")),
        fmt_int(timesteps.get("max")),
        fmt_float(timesteps.get("mean")),
        fmt_float(timesteps.get("median")),
        fmt_int(timesteps.get("total")),
    )
    logger.info(
        "  Date span: start=%s, end=%s, num_days=%s",
        date_span.get("start"),
        date_span.get("end"),
        date_span.get("num_days"),
    )
    logger.info(
        "  Glucose (mmol/L): mean=%s, std=%s, min=%s, max=%s, count=%s",
        fmt_float(glucose.get("mean")),
        fmt_float(glucose.get("std")),
        fmt_float(glucose.get("min")),
        fmt_float(glucose.get("max")),
        fmt_int(glucose.get("count")),
    )


def validate_holdout_config(
    dataset_name: str, registry: DatasetRegistry, verbose: bool = True
) -> Dict:
    """Validate holdout configuration for a dataset.

    Performs comprehensive validation including:
    - Config existence check
    - Data loading and split verification
    - Data leakage detection
    - Temporal ordering validation
    - Minimum sample requirements

    Args:
        dataset_name: Name of dataset to validate
        registry: DatasetRegistry instance
        verbose: If True, log detailed information

    Returns:
        Dictionary with validation results containing:
        - dataset_name: str
        - config_exists: bool
        - load_successful: bool
        - no_data_leakage: bool
        - train_sample_count: int
        - holdout_sample_count: int
        - train_patients: List[str]
        - holdout_patients: List[str]
        - errors: List[str]
    """
    if verbose:
        logger.info(f"{'=' * 60}")
        logger.info(f"Validating holdout config for: {dataset_name}")
        logger.info(f"{'=' * 60}")

    results = {
        "dataset_name": dataset_name,
        "config_exists": False,
        "load_successful": False,
        "no_data_leakage": False,
        "train_sample_count": 0,
        "holdout_sample_count": 0,
        "train_patients": [],
        "holdout_patients": [],
        "errors": [],
    }

    try:
        # Check if config exists
        config = registry.get_holdout_config(dataset_name)
        if config is None:
            results["errors"].append("No holdout configuration found")
            return results
        results["config_exists"] = True
        if verbose:
            logger.info(f"✓ Configuration found: {config.holdout_type.value}")

        # Try to load and split data
        train_data, holdout_data = registry.load_dataset_with_split(dataset_name)
        results["load_successful"] = True
        results["train_sample_count"] = len(train_data)
        results["holdout_sample_count"] = len(holdout_data)

        if verbose:
            logger.info("✓ Data loaded successfully")
            logger.info(f"  - Training samples: {len(train_data):,}")
            logger.info(f"  - Holdout samples: {len(holdout_data):,}")
            _log_dataset_runtime_info(registry.get_dataset_runtime_info(dataset_name))

        has_train_patient_col = "p_num" in train_data.columns
        has_holdout_patient_col = "p_num" in holdout_data.columns

        if config.holdout_type == HoldoutType.PATIENT_BASED:
            _validate_patient_split_checks(
                config=config,
                train_data=train_data,
                holdout_data=holdout_data,
                results=results,
                verbose=verbose,
            )
        elif config.holdout_type == HoldoutType.TEMPORAL:
            _validate_temporal_checks(
                config=config,
                train_data=train_data,
                holdout_data=holdout_data,
                has_train_patient_col=has_train_patient_col,
                has_holdout_patient_col=has_holdout_patient_col,
                results=results,
                verbose=verbose,
            )
        elif config.holdout_type == HoldoutType.HYBRID:
            _validate_patient_split_checks(
                config=config,
                train_data=train_data,
                holdout_data=holdout_data,
                results=results,
                verbose=verbose,
            )
            _validate_temporal_checks(
                config=config,
                train_data=train_data,
                holdout_data=holdout_data,
                has_train_patient_col=has_train_patient_col,
                has_holdout_patient_col=has_holdout_patient_col,
                results=results,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported holdout type: {config.holdout_type.value}")

    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            logger.error(f"✗ Error during validation: {e}")
        return results

    # Final status
    if verbose:
        if not results["errors"]:
            logger.info("")
            logger.info(f"✓ All validations passed for {dataset_name}")
        else:
            logger.warning(
                f"⚠ Validation completed with {len(results['errors'])} issue(s)"
            )

    return results


def validate_all_datasets(
    config_dir: str = "configs/data/holdout", verbose: bool = True
) -> List[Dict]:
    """Validate all datasets with holdout configurations.

    Args:
        config_dir: Directory containing holdout config files
        verbose: If True, log detailed information

    Returns:
        List of validation result dictionaries, one per dataset
    """
    registry = DatasetRegistry(holdout_config_dir=config_dir)
    available_datasets = registry.list_available_datasets()
    logger.info(f"{'=' * 80}")
    logger.info(f"{'=' * 80}")
    logger.info(f"{' ' * 27} DATASET HOLDOUT VALIDATOR ")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'=' * 80}")
    if not available_datasets:
        logger.error("No datasets with holdout configurations found!")
        logger.info("Run: python scripts/data_processing/generate_holdout_configs.py")
        return []

    if verbose:
        logger.info(f"Found {len(available_datasets)} datasets with holdout configs.")
        logger.info(f"\tDatasets: {', '.join(available_datasets)}\n")

    all_results = []

    for dataset_name in available_datasets:
        try:
            results = validate_holdout_config(dataset_name, registry, verbose=verbose)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to validate {dataset_name}: {e}")
            all_results.append(
                {
                    "dataset_name": dataset_name,
                    "config_exists": False,
                    "load_successful": False,
                    "no_data_leakage": False,
                    "train_sample_count": 0,
                    "holdout_sample_count": 0,
                    "errors": [str(e)],
                }
            )

    return all_results


def print_validation_summary(results: List[Dict], verbose: bool = True):
    """Print summary table of validation results.

    Args:
        results: List of validation result dictionaries
        verbose: If True, print detailed error information
    """
    if verbose:
        logger.info(f"{'=' * 80}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'=' * 80}")

    table_data = []
    for r in results:
        status = "✓ PASS" if not r["errors"] else f"✗ FAIL ({len(r['errors'])})"
        table_data.append(
            [
                r["dataset_name"],
                "✓" if r["config_exists"] else "✗",
                "✓" if r["load_successful"] else "✗",
                "✓" if r["no_data_leakage"] else "✗",
                f"{r['train_sample_count']:,}",
                f"{r['holdout_sample_count']:,}",
                len(r.get("train_patients", [])),
                len(r.get("holdout_patients", [])),
                status,
            ]
        )

    headers = [
        "Dataset",
        "Config",
        "Load",
        "No Leak",
        "Train Samples",
        "Holdout Samples",
        "Train Pat's",
        "Hold Pat's",
        "Status",
    ]

    logger.info(
        "Data Validation Summary Table\n"
        + tabulate(table_data, headers=headers, tablefmt="grid")
    )

    # Print detailed errors if verbose
    if verbose:
        errors_found = False
        for r in results:
            if r["errors"]:
                errors_found = True
                logger.error(f"\nErrors for {r['dataset_name']}:")
                for error in r["errors"]:
                    logger.error(f"  - {error}")

        if not errors_found:
            logger.info("")
            logger.info("✓ All datasets validated successfully!")
            logger.info("")
        else:
            logger.warning("")
            logger.warning(
                "⚠ Some datasets have validation issues. Review errors above."
            )
