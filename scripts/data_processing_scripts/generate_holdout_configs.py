# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""
CLI wrapper for generating holdout configurations.

This script is a thin wrapper around holdout_utils.generate_holdout_configs_for_datasets().
Core logic is in src/data/versioning/holdout_utils.py for reusability.

Usage Examples:
    # Generate hybrid (temporal + patient) holdout configs for default datasets
    python scripts/data_processing_scripts/generate_holdout_configs.py
    
    # Generate for specific datasets
    python scripts/data_processing_scripts/generate_holdout_configs.py --datasets lynch_2022 brown_2019
    
    # Generate temporal-only configs
    python scripts/data_processing_scripts/generate_holdout_configs.py --split-type temporal
    
    # Generate patient-only configs
    python scripts/data_processing_scripts/generate_holdout_configs.py --split-type patient
    
    # Custom holdout percentages
    python scripts/data_processing_scripts/generate_holdout_configs.py --temporal-pct 0.15 --patient-pct 0.15
"""

import argparse
import logging
from pathlib import Path

from src.data.versioning import holdout_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configuration parameters
DEFAULT_DATASETS = ["lynch_2022", "tamborlane_2008", "brown_2019", "colas_2019"]
DEFAULT_SPLIT_TYPE = "hybrid"
DEFAULT_TEMPORAL_HOLDOUT_PERCENTAGE = 0.2
DEFAULT_PATIENT_HOLDOUT_PERCENTAGE = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUTPUT_DIR = Path("configs/data/holdout")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate holdout configurations for datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate hybrid (temporal + patient) configs for default datasets
  python scripts/data_processing_scripts/generate_holdout_configs.py
  
  # Generate for specific datasets
  python scripts/data_processing_scripts/generate_holdout_configs.py --datasets lynch_2022 brown_2019
  
  # Generate temporal-only configs
  python scripts/data_processing_scripts/generate_holdout_configs.py --split-type temporal
  
  # Custom holdout percentages
  python scripts/data_processing_scripts/generate_holdout_configs.py --temporal-pct 0.15 --patient-pct 0.15
        """
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        help=f"Dataset names to process (default: {DEFAULT_DATASETS})"
    )
    
    parser.add_argument(
        "--split-type",
        choices=["hybrid", "temporal", "patient"],
        default=DEFAULT_SPLIT_TYPE,
        help=f"Type of holdout split (default: {DEFAULT_SPLIT_TYPE})"
    )
    
    parser.add_argument(
        "--temporal-pct",
        type=float,
        default=DEFAULT_TEMPORAL_HOLDOUT_PERCENTAGE,
        help=f"Temporal holdout percentage, e.g., 0.2 for 20%% (default: {DEFAULT_TEMPORAL_HOLDOUT_PERCENTAGE})"
    )
    
    parser.add_argument(
        "--patient-pct",
        type=float,
        default=DEFAULT_PATIENT_HOLDOUT_PERCENTAGE,
        help=f"Patient holdout percentage, e.g., 0.2 for 20%% (default: {DEFAULT_PATIENT_HOLDOUT_PERCENTAGE})"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducible patient selection (default: {DEFAULT_RANDOM_SEED})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for configuration files (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Use defaults if not specified
    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    output_dir = Path(args.output_dir)
    
    # Log configuration
    logger.info("="*60)
    logger.info("CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info(f"Split Type: {args.split_type}")
    logger.info(f"Temporal Holdout: {args.temporal_pct*100:.1f}%")
    logger.info(f"Patient Holdout: {args.patient_pct*100:.1f}%")
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*60)
    
    # Call library function to generate configs
    holdout_utils.generate_holdout_configs_for_datasets(
        datasets=datasets,
        split_type=args.split_type,
        temporal_pct=args.temporal_pct,
        patient_pct=args.patient_pct,
        seed=args.seed,
        output_dir=output_dir,
    )

