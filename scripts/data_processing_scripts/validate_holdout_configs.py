# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
CLI wrapper for validating holdout configurations.

This script is a thin wrapper around holdout_utils validation functions.
Core logic is in src/data/versioning/holdout_utils.py for reusability.

Usage Examples:
    # Validate all datasets in default config directory
    python scripts/data_processing_scripts/validate_holdout_configs.py

    # Validate all datasets in 5% holdout configs
    python scripts/data_processing_scripts/validate_holdout_configs.py --config-dir configs/data/holdout_5pct

    # Validate specific dataset (shows detailed info + validation)
    python scripts/data_processing_scripts/validate_holdout_configs.py lynch_2022

    # Validate specific dataset with custom config directory
    python scripts/data_processing_scripts/validate_holdout_configs.py lynch_2022 --config-dir configs/data/holdout_5pct

Output:
    - Prints validation results for each dataset
    - Shows summary table with train/holdout sizes and patient counts
    - Reports any data leakage or configuration issues
    - Verifies temporal ordering for temporal splits
"""

import argparse
import logging

from src.data.versioning import holdout_utils
from src.data.versioning.dataset_registry import DatasetRegistry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate holdout configurations for datasets"
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        help="Specific dataset to validate (if omitted, validates all datasets)",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/data/holdout",
        help="Directory containing holdout config files (default: configs/data/holdout)",
    )

    args = parser.parse_args()

    if args.dataset_name:
        # Validate specific dataset - show detailed info first
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset Information: {args.dataset_name}")
        logger.info(f"{'='*60}")

        registry = DatasetRegistry(holdout_config_dir=args.config_dir)

        try:
            info = registry.get_split_info(args.dataset_name)

            if "error" in info:
                logger.error(f"Error: {info['error']}")
            else:
                logger.info("\nGeneral Information:")
                logger.info(f"  Dataset: {info['dataset_name']}")
                logger.info(f"  Holdout Type: {info['holdout_type']}")
                logger.info(f"  Description: {info['description']}")
                logger.info(f"  Version: {info['version']}")
                logger.info(f"  Created: {info['created_date']}")

                if "temporal_split" in info:
                    ts = info["temporal_split"]
                    logger.info("\nTemporal Split Configuration:")
                    logger.info(
                        f"  Holdout Percentage: {ts['holdout_percentage']*100:.1f}%"
                    )
                    logger.info(f"  Min Train Samples: {ts['min_train_samples']}")
                    logger.info(f"  Min Holdout Samples: {ts['min_holdout_samples']}")

                if "patient_split" in info:
                    ps = info["patient_split"]
                    logger.info("\nPatient Split Configuration:")
                    logger.info(
                        f"  Number of Holdout Patients: {ps['num_holdout_patients']}"
                    )
                    logger.info(
                        f"  Holdout Patients: {', '.join(ps['holdout_patients'])}"
                    )
                    logger.info(f"  Min Train Patients: {ps['min_train_patients']}")
                    logger.info(f"  Random Seed: {ps['random_seed']}")

        except Exception as e:
            logger.error(f"Error retrieving dataset info: {e}")

        # Then validate
        holdout_utils.validate_holdout_config(args.dataset_name, registry)
    else:
        # Validate all datasets
        holdout_utils.validate_all_datasets(config_dir=args.config_dir)
