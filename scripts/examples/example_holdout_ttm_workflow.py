#!/usr/bin/env python3
"""
End-to-end example: Holdout system with TTM training and evaluation.

This script demonstrates the complete workflow:
1. Generate holdout configurations
2. Validate configurations
3. Load and combine training data from multiple datasets
4. Train TTM model on combined data
5. Save model
6. Load model
7. Evaluate on holdout sets per dataset

Usage:
    # Run with default settings (combine all 4 datasets, 5% holdout, 1 epoch)
    sbatch scripts/examples/run_holdout_ttm_workflow.sh
    
    # Run with specific datasets combined
    sbatch --export=DATASETS="lynch_2022 aleppo" scripts/examples/run_holdout_ttm_workflow.sh
    
    # Use different config directory
    sbatch --export=DATASETS="lynch_2022 brown_2019",CONFIG_DIR="configs/data/holdout" scripts/examples/run_holdout_ttm_workflow.sh
    
    # Customize number of epochs
    sbatch --export=DATASETS="aleppo brown_2019",EPOCHS=2 scripts/examples/run_holdout_ttm_workflow.sh
    
    # Direct python call (for testing)
    python scripts/examples/example_holdout_ttm_workflow.py --datasets lynch_2022 brown_2019 --config-dir configs/data/holdout_5pct --epochs 1
"""

import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime

from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.preprocessing.dataset_combiner import combine_datasets_for_training, print_dataset_column_table
from src.models.base import DistributedConfig, GPUManager
from src.models.ttm import TTMForecaster, TTMConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from data processing modules
logging.getLogger('src.data').setLevel(logging.WARNING)
logging.getLogger('src.models').setLevel(logging.WARNING)
logging.getLogger('src.utils').setLevel(logging.WARNING)


def step1_generate_holdout_configs(config_dir: str = "configs/data/holdout", output_dir: str = None, datasets: list = None):
    """Step 1: Generate holdout configurations and copy to artifacts directory."""
    logger.info("\n")
    logger.info("#"*80)
    logger.info("### STEP 1: Generate Holdout Configurations")
    logger.info("#"*80 + "\n")

    config_path = Path(config_dir)
    
    if config_path.exists():
        configs = list(config_path.glob("*.yaml"))
        logger.info(f"✓ Holdout configs already exist: {len(configs)} datasets")
        for cfg in configs:
            logger.info(f"  - {cfg.stem}")
        
        # Copy only configs for datasets being used in this run
        if output_dir and datasets:
            artifacts_config_dir = Path(output_dir) / "configs"
            artifacts_config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying configs to artifacts directory: {artifacts_config_dir}")
            logger.info(f"Datasets in this run: {', '.join(datasets)}")
            
            copied_count = 0
            for cfg in configs:
                # Only copy if this config matches one of the datasets being used
                if cfg.stem in datasets:
                    dest = artifacts_config_dir / cfg.name
                    shutil.copy2(cfg, dest)
                    logger.info(f"  ✓ Copied: {cfg.name}")
                    copied_count += 1
            
            logger.info(f"✓ Copied {copied_count}/{len(datasets)} configs to: {artifacts_config_dir}")
        
        return True
    else:
        logger.warning(f"⚠ Config directory does not exist: {config_dir}")
        logger.info("  Run: python scripts/data_processing_scripts/generate_holdout_configs.py")
        return False


def step2_validate_holdout_configs(dataset_name: str, config_dir: str, index: int):
    """Step 2: Validate holdout configuration with comprehensive checks."""
    logger.info("="*80)
    logger.info(f"STEP 2.{index + 1}: Validate Holdout Configuration")
    logger.info(f"Validating dataset {index + 1}: {dataset_name}")
    logger.info("="*80)
    
    registry = DatasetRegistry(holdout_config_dir=config_dir)
    
    # Get config
    config = registry.get_holdout_config(dataset_name)
    if config is None:
        logger.error(f"✗ No config found for {dataset_name}")
        return False
    
    logger.info(f"✓ Config loaded for {dataset_name}")
    logger.info(f"  Type: {config.holdout_type.value}")
    
    if config.temporal_config:
        logger.info(f"  Temporal holdout: {config.temporal_config.holdout_percentage*100}%")
    
    if config.patient_config:
        logger.info(f"  Holdout patients: {len(config.patient_config.holdout_patients)}")
    
    # Run comprehensive validation from holdout_utils
    from src.data.versioning import holdout_utils
    try:
        results = holdout_utils.validate_holdout_config(dataset_name, registry)
        
        # Check results
        if results["errors"]:
            logger.error(f"✗ Validation failed with {len(results['errors'])} error(s):")
            for error in results["errors"]:
                logger.error(f"  - {error}")
            return False
        else:
            logger.info(f"✓ All comprehensive validations passed")
            return True
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return False



def step3_load_training_data(dataset_names: list, config_dir: str):
    """Step 3: Load and combine training data from multiple datasets."""
    logger.info("="*80)
    logger.info("STEP 3: Load Training Data")
    logger.info("="*80)
    
    registry = DatasetRegistry(holdout_config_dir=config_dir)
    
    # Combine multiple datasets
    combined_data, column_info = combine_datasets_for_training(
        dataset_names=dataset_names,
        registry=registry,
        config_dir=config_dir
    )
    # Print detailed column comparison table
    print_dataset_column_table(column_info, list(combined_data.columns))
    
    logger.info(f"✓ Combined training data ready")
    logger.info(f"  Total samples: {len(combined_data):,}")
    logger.info(f"  Total columns: {len(combined_data.columns)}")
    logger.info(f"  First 5 columns: {combined_data.columns[:5].tolist()}")
    logger.info(f"  Datasets: {', '.join(dataset_names)}")
    
    if 'p_num' in combined_data.columns or 'id' in combined_data.columns:
        patient_col = 'p_num' if 'p_num' in combined_data.columns else 'id'
        n_patients = len(combined_data[patient_col].unique())
        logger.info(f"  Total patients: {n_patients}")
    
    return combined_data


def step4_train_ttm_model(combined_data, dataset_names: list, output_dir: str, num_epochs: int = 1):
    """Step 4: Train TTM model on combined dataset."""    
    logger.info("="*80)
    logger.info("STEP 4: Train TTM Model")
    logger.info(f"Datasets: {', '.join(dataset_names)}")
    logger.info("="*80)
    
    # GPU setup
    gpu_info = GPUManager.get_gpu_info()
    logger.info(f"GPU available: {gpu_info['gpu_available']}")
    logger.info(f"GPU count: {gpu_info['gpu_count']}")
    
    # Single GPU training (no distributed)
    distributed_config = DistributedConfig(enabled=False)
    use_cpu = not gpu_info["gpu_available"]
    
    # TTM configuration
    config = TTMConfig(
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        forecast_length=96,
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=num_epochs,
        use_cpu=use_cpu,
        fp16=gpu_info["gpu_available"] and not use_cpu,
    )
    
    logger.info(f"Model config:")
    logger.info(f"  Context length: {config.context_length}")
    logger.info(f"  Forecast length: {config.forecast_length}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.num_epochs}")
    
    # Create model
    model = TTMForecaster(config, distributed_config=distributed_config)
    logger.info("✓ TTM model created")
    
    # Train
    print(f"\n>>> Starting training on combined datasets: {', '.join(dataset_names)}")
    print(f">>> Output directory: {output_dir}")
    print(f">>> Training with {num_epochs} epoch(s)...\n")
    logger.info(f"Training on combined datasets: {', '.join(dataset_names)}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Pass the combined DataFrame directly instead of dataset name
        results = model.fit(train_data=combined_data, output_dir=output_dir)
        print(f"\n>>> Training completed successfully on combined datasets\n")
        logger.info("✓ Training completed")
        logger.info(f"  Results: {list(results.keys())}")
        return model, results
    except Exception as e:
        print(f"\n>>> ERROR: Training failed: {e}\n")
        logger.error(f"✗ Training failed: {e}")
        raise


def step5_save_model(model: TTMForecaster, save_path: str):
    """Step 5: Save trained model."""
    logger.info("="*80)
    logger.info("STEP 5: Save Trained Model")
    logger.info("="*80)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        model.save(str(save_path))
        logger.info(f"✓ Model saved to: {save_path}")
        logger.info(f"  Size: {save_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save model: {e}")
        return False


def step6_load_model(load_path: str, config: TTMConfig):
    """Step 6: Load trained model."""
    print("\n" + "#"*80)
    print("### STEP 6: Load Trained Model")
    print("#"*80 + "\n")
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Load Trained Model")
    logger.info("="*80)
    
    load_path = Path(load_path)
    
    if not load_path.exists():
        logger.error(f"✗ Model file not found: {load_path}")
        return None
    
    try:
        # Create new model instance with same config
        distributed_config = DistributedConfig(enabled=False)
        model = TTMForecaster(config, distributed_config=distributed_config)
        
        # Load weights
        model.load(str(load_path))
        logger.info(f"✓ Model loaded from: {load_path}")
        return model
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return None


def step7_evaluate_on_holdout(model: TTMForecaster, dataset_name: str, config_dir: str):
    """Step 7: Evaluate model on holdout set."""
    print("\n" + "#"*80)
    print(f"### STEP 7: Evaluate on Holdout Set for {dataset_name}")
    print("#"*80 + "\n")
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Evaluate on Holdout Set")
    logger.info("="*80)
    
    registry = DatasetRegistry(holdout_config_dir=config_dir)
    
    # Load holdout data
    holdout_data = registry.load_holdout_data_only(dataset_name)
    logger.info(f"✓ Holdout data loaded: {len(holdout_data):,} samples")
    
    # Evaluate
    try:
        # Note: This is a placeholder - actual evaluation depends on your TTM model's predict method
        logger.info("Running evaluation on holdout set...")
        
        # Example evaluation (you'll need to implement based on your model's API)
        # predictions = model.predict(holdout_data)
        # metrics = calculate_metrics(holdout_data, predictions)
        
        logger.info("✓ Evaluation completed")
        logger.info("  Metrics (placeholder):")
        logger.info("    - MSE: TBD")
        logger.info("    - MAE: TBD")
        logger.info("    - RMSE: TBD")
        
        # Note: Split evaluation by patient if needed
        if 'p_num' in holdout_data.columns:
            holdout_patients = holdout_data['p_num'].unique()
            logger.info(f"\n  Per-patient evaluation available for {len(holdout_patients)} patients")
        
        return True
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end holdout workflow with TTM training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        required=True,
        help="Dataset names to combine (e.g., lynch_2022 aleppo brown_2019)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/data/holdout_5pct",
        help="Holdout config directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Training output directory"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        args.output_dir = f"./trained_models/artifacts/_tsfm_testing/{timestamp}_default_dir_holdout_workflow"
    
    model_path = Path(args.output_dir) / "model.pt"
        
    logger.info("="*80)
    logger.info("🚀 HOLDOUT SYSTEM WORKFLOW WITH TTM")
    logger.info("Start of: example_holdout_ttm_workflow.py")
    logger.info("="*80)
    logger.info(f"### Datasets: {', '.join(args.datasets)}")
    logger.info(f"### Config dir: {args.config_dir}")
    logger.info(f"### Output dir: {args.output_dir}")
    logger.info(f"### Epochs: {args.epochs}")
    logger.info("="*80)
    
    try:
        # Step 1: Check/generate holdout configs
        if not step1_generate_holdout_configs(args.config_dir, args.output_dir, args.datasets):
            logger.error("Please generate holdout configs first")
            return
        
        # Step 2: Validate configuration for all datasets
        logger.info("="*80)
        logger.info("STEP 2: Validate Holdout Configurations")
        logger.info(f"Validating {len(args.datasets)} dataset(s)")
        logger.info("="*80)
        
        from src.data.versioning import holdout_utils
        registry = DatasetRegistry(holdout_config_dir=args.config_dir)
        
        # Validate all datasets and collect results (suppress verbose per-dataset logging)
        validation_results = []
        for dataset_name in args.datasets:
            results = holdout_utils.validate_holdout_config(dataset_name, registry, verbose=False)
            validation_results.append(results)
            
            # Log brief status
            if results["errors"]:
                logger.error(f"✗ {dataset_name}: {len(results['errors'])} error(s)")
            else:
                logger.info(f"✓ {dataset_name}: Validation passed")
        
        # Print summary table
        holdout_utils.print_validation_summary(validation_results, verbose=False)
        
        # Check if any failed
        failed_datasets = [r["dataset_name"] for r in validation_results if r["errors"]]
        if failed_datasets:
            logger.error(f"Configuration validation failed for: {', '.join(failed_datasets)}")
            return
        
        # Step 3: Load and combine training data
        combined_train_data = step3_load_training_data(args.datasets, args.config_dir)
        
        if not args.skip_training:
            # Step 4: Train model on combined data
            model, results = step4_train_ttm_model(combined_train_data, args.datasets, args.output_dir, num_epochs=args.epochs)
            
            # Step 5: Save model
            if not step5_save_model(model, str(model_path)):
                logger.error("Failed to save model")
                return
        else:
            # Step 4: Skip training (use existing model)
            logger.info("="*80)
            logger.info("STEP 4: Train TTM Model")
            logger.info("⏭️  Skipping training (using existing model)")
            logger.info("="*80)
            # Step 5: Skip saving model
            logger.info("="*80)
            logger.info("STEP 5: Save Trained Model")
            logger.info("⏭️  Skipping training (No trained model to save)")
            logger.info("="*80)

        # Step 6: Load model
        # Recreate config for loading
        gpu_info = GPUManager.get_gpu_info()
        config = TTMConfig(
            model_path="ibm-granite/granite-timeseries-ttm-r2",
            context_length=512,
            forecast_length=96,
            batch_size=16,
            use_cpu=not gpu_info["gpu_available"],
        )
        
        model = step6_load_model(str(model_path), config)
        if model is None:
            logger.error("Failed to load model")
            return
        
        # Step 7: Evaluate on holdout for each dataset
        for dataset_name in args.datasets:
            logger.info(f"\nEvaluating on holdout set for: {dataset_name}")
            step7_evaluate_on_holdout(model, dataset_name, args.config_dir)
        
        print("\n" + "#"*80)
        print("### ✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("#"*80)
        print(f"### Model saved at: {model_path}")
        print(f"### Training outputs in: {args.output_dir}")
        print("#"*80 + "\n")
        
        logger.info("\n" + "="*80)
        logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Training outputs in: {args.output_dir}")
        logger.info("End of : example_holdout_ttm_workflow.py")
        
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Workflow interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
