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
from typing import Dict, List

import pandas as pd
from tabulate import tabulate

from src.data.diabetes_datasets.data_loader import get_loader
from src.data.versioning.holdout_config import (
    HoldoutConfig,
    HoldoutType,
    PatientHoldoutConfig,
    TemporalHoldoutConfig,
)
from src.data.versioning.holdout_manager import generate_patient_holdout_list
from src.data.versioning.dataset_registry import DatasetRegistry

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
    min_train_samples: int = 50,
    min_holdout_samples: int = 96,
    min_train_patients: int = 2,
    min_holdout_patients: int = 1,
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
            f"Hybrid holdout strategy: {temporal_pct*100:.0f}% temporal split "
            f"+ {patient_pct*100:.0f}% patient holdout. "
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
) -> Dict[str, any]:
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
            logger.info(f"{'='*60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*60}")

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
        - train_size: int
        - holdout_size: int
        - train_patients: List[str]
        - holdout_patients: List[str]
        - errors: List[str]
    """
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating holdout config for: {dataset_name}")
        logger.info(f"{'='*60}")

    results = {
        "dataset_name": dataset_name,
        "config_exists": False,
        "load_successful": False,
        "no_data_leakage": False,
        "train_size": 0,
        "holdout_size": 0,
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
        results["train_size"] = len(train_data)
        results["holdout_size"] = len(holdout_data)

        if verbose:
            logger.info("✓ Data loaded successfully")
            logger.info(f"  - Training samples: {len(train_data):,}")
            logger.info(f"  - Holdout samples: {len(holdout_data):,}")

        # Check for patient-based split
        if "p_num" in train_data.columns and "p_num" in holdout_data.columns:
            train_patients = sorted([str(p) for p in train_data["p_num"].unique()])
            holdout_patients = sorted([str(p) for p in holdout_data["p_num"].unique()])

            results["train_patients"] = train_patients
            results["holdout_patients"] = holdout_patients

            if verbose:
                logger.info(f"  - Training patients: {len(train_patients)}")
                logger.info(f"  - Holdout patients: {len(holdout_patients)}")

            # Check for overlap based on split type
            overlap = set(train_patients) & set(holdout_patients)

            # Validate based on split type
            if config.holdout_type.value == "patient":
                # Patient-based splits should have NO overlap
                if overlap:
                    results["errors"].append(
                        f"Patient overlap detected: {len(overlap)} patients"
                    )
                    if verbose:
                        logger.error(
                            f"✗ Patient overlap detected: {len(overlap)} patients"
                        )
                else:
                    results["no_data_leakage"] = True
                    if verbose:
                        logger.info("✓ No patient overlap (patient-based split)")

            elif config.holdout_type.value == "hybrid":
                # Hybrid splits: designated holdout patients should NOT appear in training
                if config.patient_config and hasattr(
                    config.patient_config, "holdout_patients"
                ):
                    designated_holdout = set(
                        str(p) for p in config.patient_config.holdout_patients
                    )

                    # Check if any designated holdout patients appear in training set
                    leak = designated_holdout & set(train_patients)
                    if leak:
                        results["errors"].append(
                            f"Designated holdout patients in training: {leak}"
                        )
                        if verbose:
                            logger.error(
                                f"✗ Designated holdout patients leaked: {leak}"
                            )
                    else:
                        results["no_data_leakage"] = True
                        if verbose:
                            logger.info(
                                f"✓ Hybrid split valid: {len(overlap)} patients overlap (temporal)"
                            )
                else:
                    results["no_data_leakage"] = True
                    if verbose:
                        logger.info(
                            f"✓ Hybrid split: {len(overlap)} patients overlap (expected)"
                        )
            else:
                # Temporal splits: overlap is EXPECTED
                results["no_data_leakage"] = True
                if verbose:
                    logger.info(
                        f"✓ Temporal split: {len(overlap)} patients overlap (expected)"
                    )
        else:
            # Temporal-only splits
            results["no_data_leakage"] = True
            if verbose:
                logger.info("✓ Temporal-only split (no patient column)")

        # Validate minimum samples
        if results["train_size"] < 100:
            results["errors"].append(
                f"Training set very small: {results['train_size']} samples"
            )
            if verbose:
                logger.warning("⚠ Training set very small")

        if results["holdout_size"] < 20:
            results["errors"].append(
                f"Holdout set very small: {results['holdout_size']} samples"
            )
            if verbose:
                logger.warning("⚠ Holdout set very small")

        # Check temporal ordering for temporal splits
        if (
            config.temporal_config
            and "time" in train_data.columns
            and "time" in holdout_data.columns
        ):
            try:
                train_data["time"] = pd.to_datetime(train_data["time"])
                holdout_data["time"] = pd.to_datetime(holdout_data["time"])

                # Check per patient if patient column exists
                if "p_num" in train_data.columns:
                    temporal_issues = []
                    for patient in train_patients:
                        patient_train = train_data[train_data["p_num"] == patient]
                        if len(patient_train) > 0:
                            max_pt = patient_train["time"].max()
                            patient_holdout = holdout_data[
                                holdout_data["p_num"] == patient
                            ]
                            if len(patient_holdout) > 0:
                                min_ph = patient_holdout["time"].min()
                                if max_pt >= min_ph:
                                    temporal_issues.append(patient)

                    if temporal_issues:
                        results["errors"].append(
                            f"Temporal ordering issue: {temporal_issues}"
                        )
                        if verbose:
                            logger.error(
                                f"✗ Temporal ordering issue for patients: {temporal_issues}"
                            )
                    else:
                        if verbose:
                            logger.info("✓ Temporal ordering correct for all patients")
                else:
                    max_train_time = train_data["time"].max()
                    min_holdout_time = holdout_data["time"].min()
                    if max_train_time >= min_holdout_time:
                        results["errors"].append(
                            "Temporal ordering issue: train extends into holdout period"
                        )
                        if verbose:
                            logger.error("✗ Temporal ordering issue")
                    else:
                        if verbose:
                            logger.info("✓ Temporal ordering correct")
            except Exception as e:
                if verbose:
                    logger.warning(f"⚠ Could not verify temporal ordering: {e}")

    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            logger.error(f"✗ Error during validation: {e}")
        return results

    # Final status
    if verbose:
        if not results["errors"]:
            logger.info(f"\n✓ All validations passed for {dataset_name}")
        else:
            logger.warning(
                f"\n⚠ Validation completed with {len(results['errors'])} issue(s)"
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

    if not available_datasets:
        logger.error("No datasets with holdout configurations found!")
        logger.info(
            "Run: python scripts/data_processing_scripts/generate_holdout_configs.py"
        )
        return []

    if verbose:
        logger.info(f"\nFound {len(available_datasets)} datasets with holdout configs")
        logger.info(f"Datasets: {', '.join(available_datasets)}\n")

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
                    "train_size": 0,
                    "holdout_size": 0,
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
        logger.info(f"{'='*80}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'='*80}")

    table_data = []
    for r in results:
        status = "✓ PASS" if not r["errors"] else f"✗ FAIL ({len(r['errors'])})"
        table_data.append(
            [
                r["dataset_name"],
                "✓" if r["config_exists"] else "✗",
                "✓" if r["load_successful"] else "✗",
                "✓" if r["no_data_leakage"] else "✗",
                f"{r['train_size']:,}",
                f"{r['holdout_size']:,}",
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
        "Train Size",
        "Holdout Size",
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
            logger.info("\n✓ All datasets validated successfully!")
        else:
            logger.warning(
                "\n⚠ Some datasets have validation issues. Review errors above."
            )
