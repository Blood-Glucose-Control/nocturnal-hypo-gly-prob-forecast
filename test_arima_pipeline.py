#!/usr/bin/env python3
"""
Example script to test the ARIMA training pipeline.
This demonstrates how to use the statistical trainer with the Kaggle BRIS T1D dataset.
"""

import sys
from pathlib import Path
from src.training.pipeline import TrainingPipeline

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Test the ARIMA training pipeline."""

    # Test with ARIMA config
    config_path = "scripts/training/configs/arima_config.yaml"

    print("Testing ARIMA training pipeline...")
    print(f"Using config: {config_path}")

    try:
        # Create pipeline from config
        pipeline = TrainingPipeline.from_config_file(config_path)

        print("Pipeline created successfully!")
        print(f"Model type: {pipeline.config.model_type}")
        print(f"Model name: {pipeline.config.model_name}")
        print(f"Dataset: {pipeline.config.dataset_name}")

        # Train on a subset of patients for testing (optional)
        # You can specify patient IDs or leave as None to use all
        test_patients = None  # or ["patient_1", "patient_2"] for specific patients

        print("Starting training...")
        results = pipeline.train(patient_ids=test_patients)

        print("Training completed successfully!")
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        # Save the trained model
        if pipeline.trainer.model is not None:
            model_path = pipeline.save_model(pipeline.trainer.model, results)
            print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
