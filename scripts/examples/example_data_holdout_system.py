#!/usr/bin/env python3
# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
Example script demonstrating the data holdout system.

This script shows how to:
1. Load training data for model training
2. Load holdout data for evaluation
3. Validate data splits
4. Get information about holdout configurations
"""

import logging

import pandas as pd

from src.data.versioning.dataset_registry import (
    get_dataset_registry,
    load_holdout_data,
    load_split_data,
    load_training_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_load_training_data():
    """Example 1: Load training data only (recommended for model training)."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Loading Training Data Only")
    logger.info("=" * 60)

    dataset_name = "kaggle_brisT1D"

    # This is the recommended way to load data for training
    # It ensures holdout data is never accidentally used
    train_data = load_training_data(dataset_name)

    logger.info(f"Loaded training data for {dataset_name}")
    logger.info(f"  Shape: {train_data.shape}")
    logger.info(f"  Columns: {list(train_data.columns)}")

    if "p_num" in train_data.columns:
        unique_patients = train_data["p_num"].unique()
        logger.info(f"  Patients: {len(unique_patients)} ({list(unique_patients)})")

    # Now you can train your model on train_data
    # model.fit(train_data)

    return train_data


def example_2_load_holdout_data():
    """Example 2: Load holdout data only (for final evaluation)."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Loading Holdout Data Only")
    logger.info("=" * 60)

    dataset_name = "kaggle_brisT1D"

    # This should ONLY be used for final model evaluation
    # Never use this data for training or hyperparameter tuning
    holdout_data = load_holdout_data(dataset_name)

    logger.info(f"Loaded holdout data for {dataset_name}")
    logger.info(f"  Shape: {holdout_data.shape}")

    if "p_num" in holdout_data.columns:
        unique_patients = holdout_data["p_num"].unique()
        logger.info(f"  Patients: {len(unique_patients)} ({list(unique_patients)})")

    # Now you can evaluate your trained model on holdout_data
    # results = model.evaluate(holdout_data)

    return holdout_data


def example_3_load_both_splits():
    """Example 3: Load both training and holdout data."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Loading Both Splits")
    logger.info("=" * 60)

    dataset_name = "kaggle_brisT1D"

    # Load both splits at once
    train_data, holdout_data = load_split_data(dataset_name)

    logger.info(f"Loaded split data for {dataset_name}")
    logger.info(f"  Training: {train_data.shape}")
    logger.info(f"  Holdout: {holdout_data.shape}")
    logger.info(
        f"  Split ratio: {len(train_data)/(len(train_data)+len(holdout_data)):.1%} train"
    )

    # Verify no overlap
    if "p_num" in train_data.columns and "p_num" in holdout_data.columns:
        train_patients = set(train_data["p_num"].unique())
        holdout_patients = set(holdout_data["p_num"].unique())
        overlap = train_patients & holdout_patients

        if overlap:
            logger.error(f"❌ Patient overlap detected: {overlap}")
        else:
            logger.info("✓ No patient overlap - split is valid")

    return train_data, holdout_data


def example_4_get_split_info():
    """Example 4: Get information about holdout configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Getting Split Information")
    logger.info("=" * 60)

    registry = get_dataset_registry()

    # Get all available datasets
    datasets = registry.list_available_datasets()
    logger.info(f"Available datasets: {datasets}")

    # Get detailed info for a specific dataset
    dataset_name = "kaggle_brisT1D"
    info = registry.get_split_info(dataset_name)

    logger.info(f"\nConfiguration for {dataset_name}:")
    logger.info(f"  Holdout Type: {info['holdout_type']}")
    logger.info(f"  Description: {info['description']}")
    logger.info(f"  Version: {info['version']}")

    if "temporal_split" in info:
        ts = info["temporal_split"]
        logger.info("\n  Temporal Split:")
        logger.info(f"    Holdout %: {ts['holdout_percentage']*100:.1f}%")
        logger.info(f"    Min train samples: {ts['min_train_samples']}")
        logger.info(f"    Min holdout samples: {ts['min_holdout_samples']}")

    if "patient_split" in info:
        ps = info["patient_split"]
        logger.info("\n  Patient Split:")
        logger.info(f"    Holdout patients: {ps['holdout_patients']}")
        logger.info(f"    Count: {ps['num_holdout_patients']}")
        logger.info(f"    Random seed: {ps['random_seed']}")


def example_5_validate_split():
    """Example 5: Validate a data split."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Validating Data Split")
    logger.info("=" * 60)

    from src.data.versioning.holdout_config import HoldoutConfig
    from src.data.versioning.holdout_manager import HoldoutManager

    dataset_name = "kaggle_brisT1D"

    # Load configuration
    config = HoldoutConfig.load(f"configs/data/holdout/{dataset_name}.yaml")
    logger.info(f"Loaded config: {config.holdout_type.value} split")

    # Load data and create manager
    from src.data.diabetes_datasets.data_loader import get_loader

    loader = get_loader(dataset_name, use_cached=True)
    # Data is already loaded in __init__, just access it
    data = loader.processed_data

    manager = HoldoutManager(config)
    train_data, holdout_data = manager.split_data(data)

    # Validate split
    validation = manager.validate_split(train_data, holdout_data)

    logger.info("\nValidation Results:")
    for check, passed in validation.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        logger.info(f"  {check}: {status}")

    # Get patient lists (for patient-based or hybrid splits)
    train_patients = manager.get_train_patients()
    holdout_patients = manager.get_holdout_patients()

    if train_patients and holdout_patients:
        logger.info("\nPatient Distribution:")
        logger.info(f"  Training: {len(train_patients)} patients")
        logger.info(f"  Holdout: {len(holdout_patients)} patients")


def example_6_typical_training_workflow():
    """Example 6: Typical workflow for training a model."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Typical Training Workflow")
    logger.info("=" * 60)

    dataset_name = "kaggle_brisT1D"

    # Step 1: Load training data only
    logger.info("\n1. Loading training data...")
    train_data = load_training_data(dataset_name)
    logger.info(f"   Loaded {len(train_data)} training samples")

    # Step 2: Train your model (placeholder)
    logger.info("\n2. Training model...")
    logger.info("   model = MyModel()")
    logger.info("   model.fit(train_data)")
    logger.info("   # ... training logic ...")

    # Step 3: Save the trained model
    logger.info("\n3. Saving trained model...")
    logger.info("   model.save('trained_models/artifacts/my_model/')")

    # Step 4: Final evaluation on holdout set
    logger.info("\n4. Evaluating on holdout set (FINAL EVALUATION ONLY)...")
    holdout_data = load_holdout_data(dataset_name)
    logger.info(f"   Loaded {len(holdout_data)} holdout samples")
    logger.info("   results = model.evaluate(holdout_data)")
    logger.info("   # ... evaluation logic ...")

    logger.info("\n✓ Training workflow complete!")
    logger.info("\n⚠️  IMPORTANT:")
    logger.info("   - Always use load_training_data() for training")
    logger.info("   - Only use load_holdout_data() for final evaluation")
    logger.info("   - Never train on holdout data")


def example_7_multi_dataset_training():
    """Example 7: Training on multiple datasets."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 7: Multi-Dataset Training")
    logger.info("=" * 60)

    datasets = ["kaggle_brisT1D", "gluroo", "aleppo_2017"]

    all_train_data = []
    all_holdout_data = []

    for dataset_name in datasets:
        try:
            logger.info(f"\nLoading {dataset_name}...")
            train_data = load_training_data(dataset_name)
            holdout_data = load_holdout_data(dataset_name)

            all_train_data.append(train_data)
            all_holdout_data.append(holdout_data)

            logger.info(f"  Train: {len(train_data)} samples")
            logger.info(f"  Holdout: {len(holdout_data)} samples")
        except Exception as e:
            logger.warning(f"  Failed to load {dataset_name}: {e}")

    if all_train_data:
        # Combine all training data
        combined_train = pd.concat(all_train_data, ignore_index=True)
        logger.info(
            f"\nCombined training data: {len(combined_train)} samples from {len(all_train_data)} datasets"
        )

        # Train model on combined data
        logger.info("Training on combined dataset...")
        # model.fit(combined_train)

    if all_holdout_data:
        # Evaluate on each dataset's holdout set separately
        logger.info("\nEvaluating on each dataset's holdout set...")
        for i, (dataset_name, holdout_data) in enumerate(
            zip(datasets, all_holdout_data)
        ):
            logger.info(f"  {dataset_name}: {len(holdout_data)} samples")
            # results = model.evaluate(holdout_data)


def main():
    """Run all examples."""
    logger.info("=" * 60)
    logger.info("Data Holdout System - Examples")
    logger.info("=" * 60)

    try:
        # Run examples
        example_1_load_training_data()
        example_2_load_holdout_data()
        example_3_load_both_splits()
        example_4_get_split_info()
        example_5_validate_split()
        example_6_typical_training_workflow()
        example_7_multi_dataset_training()

        logger.info("\n" + "=" * 60)
        logger.info("✓ All examples completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ Error running examples: {e}")
        logger.info("\nMake sure you have:")
        logger.info(
            "  1. Generated holdout configs: python scripts/data_processing_scripts/generate_holdout_configs.py"
        )
        logger.info("  2. Datasets are available in cache/data/")
        raise


if __name__ == "__main__":
    main()
