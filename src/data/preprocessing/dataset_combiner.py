"""
Dataset combination utilities for multi-dataset training.

This module provides functions to combine multiple datasets into a single
training set, handling column alignment and validation.
"""

import pandas as pd
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def combine_datasets_for_training(
    dataset_names: List[str],
    registry,
    config_dir: str = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Combine multiple datasets into a single training DataFrame.
    
    This function loads training data from multiple datasets and combines them
    into a single DataFrame for multi-dataset training. It handles column
    alignment by finding the intersection of columns across all datasets.
    
    Args:
        dataset_names: List of dataset names to combine (e.g., ['lynch_2022', 'brown_2019'])
        registry: DatasetRegistry instance for loading data
        config_dir: Optional holdout config directory (if using holdout configs)
    
    Returns:
        Tuple of:
            - Combined DataFrame with all training data
            - Dictionary mapping dataset names to their original column lists
    
    Example:
        >>> from src.data.versioning.dataset_registry import DatasetRegistry
        >>> registry = DatasetRegistry(holdout_config_dir="configs/data/holdout_5pct")
        >>> combined_data, column_info = combine_datasets_for_training(
        ...     ['lynch_2022', 'brown_2019'],
        ...     registry
        ... )
    """
    logger.info(f"Combining {len(dataset_names)} datasets for training...")
    
    # Load all datasets
    datasets = {}
    column_info = {}
    
    for dataset_name in dataset_names:
        logger.info(f"  Loading {dataset_name}...")
        try:
            if config_dir:
                # Load with holdout split
                data = registry.load_training_data_only(dataset_name)
            else:
                # Load full dataset
                data = registry.load_dataset(dataset_name)
            
            datasets[dataset_name] = data
            column_info[dataset_name] = list(data.columns)
            logger.info(f"    ✓ Loaded: {len(data):,} samples, {len(data.columns)} columns")
            
        except Exception as e:
            logger.error(f"    ✗ Failed to load {dataset_name}: {e}")
            raise
    
    # Find common columns across all datasets
    all_columns = [set(cols) for cols in column_info.values()]
    common_columns = set.intersection(*all_columns)
    
    logger.info(f"\nColumn analysis:")
    logger.info(f"  Common columns across all datasets: {len(common_columns)}")
    logger.info(f"  Columns: {sorted(common_columns)}")
    
    # Check for missing columns in each dataset
    for dataset_name, cols in column_info.items():
        missing = common_columns - set(cols)
        extra = set(cols) - common_columns
        if missing:
            logger.warning(f"  {dataset_name} missing columns: {sorted(missing)}")
        if extra:
            logger.info(f"  {dataset_name} extra columns: {sorted(extra)}")
    
    # Combine datasets using only common columns
    combined_dfs = []
    for dataset_name, data in datasets.items():
        # Select only common columns
        data_subset = data[sorted(common_columns)].copy()
        
        # Add dataset identifier column
        data_subset['source_dataset'] = dataset_name
        
        combined_dfs.append(data_subset)
        logger.info(f"  Adding {dataset_name}: {len(data_subset):,} samples")
    
    # Concatenate all datasets
    combined_data = pd.concat(combined_dfs, ignore_index=True)
    
    logger.info(f"\n✓ Combined dataset created:")
    logger.info(f"  Total samples: {len(combined_data):,}")
    logger.info(f"  Total columns: {len(combined_data.columns)}")
    logger.info(f"  Final columns: {list(combined_data.columns)}")
    
    # Print dataset distribution
    if 'source_dataset' in combined_data.columns:
        logger.info(f"\nDataset distribution:")
        for dataset_name in dataset_names:
            count = len(combined_data[combined_data['source_dataset'] == dataset_name])
            pct = 100 * count / len(combined_data)
            logger.info(f"  {dataset_name}: {count:,} samples ({pct:.1f}%)")
    
    # Print sample counts per patient if available
    if 'p_num' in combined_data.columns or 'id' in combined_data.columns:
        patient_col = 'p_num' if 'p_num' in combined_data.columns else 'id'
        n_patients = len(combined_data[patient_col].unique())
        logger.info(f"\nTotal unique patients: {n_patients}")
    
    return combined_data, column_info


def print_dataset_column_table(column_info: Dict[str, List[str]], final_columns: List[str]):
    """
    Print a formatted table showing columns in each dataset and the final combined columns.
    
    Args:
        column_info: Dictionary mapping dataset names to their column lists
        final_columns: List of columns in the final combined dataset
    """
    logger.info("\n" + "="*80)
    logger.info("DATASET COLUMN COMPARISON TABLE")
    logger.info("="*80)
    
    # Get all unique columns across all datasets
    all_cols = set()
    for cols in column_info.values():
        all_cols.update(cols)
    all_cols = sorted(all_cols)
    
    # Print header
    header = f"{'Column':<30}"
    for dataset_name in column_info.keys():
        header += f" | {dataset_name:<15}"
    header += f" | {'Final':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Print each column
    for col in all_cols:
        row = f"{col:<30}"
        for dataset_name, cols in column_info.items():
            marker = "✓" if col in cols else "✗"
            row += f" | {marker:<15}"
        
        final_marker = "✓" if col in final_columns else "✗"
        row += f" | {final_marker:<10}"
        logger.info(row)
    
    logger.info("="*80)
    
    # Print summary statistics
    logger.info("\nSummary:")
    for dataset_name, cols in column_info.items():
        logger.info(f"  {dataset_name}: {len(cols)} columns")
    logger.info(f"  Final combined dataset: {len(final_columns)} columns")
    logger.info(f"  Common across all: {len(set(final_columns) - {'source_dataset'})} columns")
    logger.info("="*80 + "\n")
