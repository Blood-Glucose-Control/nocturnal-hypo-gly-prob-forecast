#!/usr/bin/env python3
"""
Example script demonstrating how to load training and holdout data.

This script shows the recommended way to load datasets with holdout splits
using the DatasetRegistry API.

Usage:
    # Load training data only (for model training)
    python scripts/examples/load_holdout_data_example.py --dataset lynch_2022 --mode train
    
    # Load holdout data only (for final evaluation)
    python scripts/examples/load_holdout_data_example.py --dataset lynch_2022 --mode holdout
    
    # Load both (for inspection)
    python scripts/examples/load_holdout_data_example.py --dataset lynch_2022 --mode both
"""

import argparse
import logging
from pathlib import Path

from src.data.versioning.dataset_registry import DatasetRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_example(dataset_name: str, config_dir: str):
    """Example: Load training data only.
    
    This is the primary method to use when training models to ensure
    holdout data is never accidentally used.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Loading training data for: {dataset_name}")
    logger.info(f"{'='*70}\n")
    
    # Create registry
    registry = DatasetRegistry(holdout_config_dir=config_dir)
    
    # Load training data only
    train_data = registry.load_training_data_only(dataset_name)
    
    # Print summary
    logger.info(f"✓ Training data loaded: {len(train_data):,} samples")
    logger.info(f"  Columns: {list(train_data.columns)}")
    
    if 'p_num' in train_data.columns:
        n_patients = len(train_data['p_num'].unique())
        logger.info(f"  Patients: {n_patients}")
    
    return train_data


def load_holdout_example(dataset_name: str, config_dir: str):
    """Example: Load holdout data only.
    
    Use this for final model evaluation only.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Loading holdout data for: {dataset_name}")
    logger.info(f"{'='*70}\n")
    
    # Create registry
    registry = DatasetRegistry(holdout_config_dir=config_dir)
    
    # Load holdout data only
    holdout_data = registry.load_holdout_data_only(dataset_name)
    
    # Print summary
    logger.info(f"✓ Holdout data loaded: {len(holdout_data):,} samples")
    logger.info(f"  Columns: {list(holdout_data.columns)}")
    
    if 'p_num' in holdout_data.columns:
        n_patients = len(holdout_data['p_num'].unique())
        logger.info(f"  Patients: {n_patients}")
    
    return holdout_data


def load_both_example(dataset_name: str, config_dir: str):
    """Example: Load both training and holdout data.
    
    Use this for inspection or when you need both splits.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Loading train/holdout split for: {dataset_name}")
    logger.info(f"{'='*70}\n")
    
    # Create registry
    registry = DatasetRegistry(holdout_config_dir=config_dir)
    
    # Get split info first (doesn't load data)
    split_info = registry.get_split_info(dataset_name)
    logger.info(f"Split configuration:")
    logger.info(f"  Type: {split_info['holdout_type']}")
    if 'temporal_split' in split_info:
        logger.info(f"  Temporal holdout: {split_info['temporal_split']['holdout_percentage']*100}%")
    if 'patient_split' in split_info:
        logger.info(f"  Holdout patients: {split_info['patient_split']['num_holdout_patients']}")
    
    # Load both splits
    train_data, holdout_data = registry.load_dataset_with_split(dataset_name)
    
    # Print summary
    logger.info(f"\n✓ Data loaded successfully:")
    logger.info(f"  Training: {len(train_data):,} samples")
    logger.info(f"  Holdout: {len(holdout_data):,} samples")
    
    if 'p_num' in train_data.columns:
        train_patients = set(str(p) for p in train_data['p_num'].unique())
        holdout_patients = set(str(p) for p in holdout_data['p_num'].unique())
        overlap = train_patients & holdout_patients
        
        logger.info(f"\n  Patient statistics:")
        logger.info(f"    Training patients: {len(train_patients)}")
        logger.info(f"    Holdout patients: {len(holdout_patients)}")
        logger.info(f"    Overlap: {len(overlap)} patients")
        
        if overlap:
            logger.info(f"    ℹ Patient overlap expected for temporal/hybrid splits")
    
    return train_data, holdout_data


def main():
    parser = argparse.ArgumentParser(
        description="Example script for loading holdout data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., lynch_2022, aleppo, brown_2019)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "holdout", "both"],
        default="train",
        help="What to load: train only, holdout only, or both"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout",
        help="Directory containing holdout config files"
    )
    
    args = parser.parse_args()
    
    # Check if config directory exists
    config_path = Path(args.config_dir)
    if not config_path.exists():
        logger.error(f"Config directory not found: {args.config_dir}")
        logger.info("Available config directories:")
        for d in Path("configs/data").glob("holdout*"):
            logger.info(f"  - {d}")
        return
    
    # Load data based on mode
    try:
        if args.mode == "train":
            load_training_example(args.dataset, args.config_dir)
        elif args.mode == "holdout":
            load_holdout_example(args.dataset, args.config_dir)
        elif args.mode == "both":
            load_both_example(args.dataset, args.config_dir)
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ Example completed successfully!")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
