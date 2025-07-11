#!/usr/bin/env python3
"""
Simple test script for the ARIMA training pipeline using mock data.
This avoids loading the full dataset and focuses on testing the training logic.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.trainers.statistical_trainer import StatisticalTrainer


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self):
        # Create mock glucose time series data
        dates = pd.date_range("2023-01-01", periods=1000, freq="5min")
        # Generate realistic glucose values with trend and noise
        base_glucose = 120 + 10 * np.sin(
            np.arange(1000) * 2 * np.pi / 288
        )  # Daily pattern
        noise = np.random.normal(0, 10, 1000)
        glucose_values = base_glucose + noise
        glucose_values = np.clip(glucose_values, 70, 300)  # Realistic glucose range

        self.processed_data = {
            "patient_001": pd.DataFrame(
                {"timestamp": dates, "glucose": glucose_values}
            ).set_index("timestamp")
        }

    def get_data(self, patient_id=None):
        if patient_id and patient_id in self.processed_data:
            return self.processed_data[patient_id]
        elif patient_id is None:
            # Return all data combined
            return pd.concat(list(self.processed_data.values()))
        else:
            return pd.DataFrame()


def test_statistical_trainer():
    """Test the statistical trainer with mock data."""

    print("Testing StatisticalTrainer with mock data...")

    # Create mock data loader
    mock_loader = MockDataLoader()

    # Test ARIMA trainer
    trainer = StatisticalTrainer(model_name="ARIMA")

    # Define simple ARIMA parameters
    model_params = {"order": (1, 1, 1), "trend": "n"}

    print("Training ARIMA model...")
    try:
        results = trainer.train(
            data=mock_loader, patient_ids=["patient_001"], model_params=model_params
        )

        print("Training completed successfully!")
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        # Test prediction
        if trainer.model is not None:
            print("\nTesting prediction...")
            predictions = trainer.predict(trainer.model, None, forecast_horizon=12)
            print(f"Generated {len(predictions)} predictions")
            print(f"First 5 predictions: {predictions[:5].values}")

        return True

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_autoarima_trainer():
    """Test the AutoARIMA trainer with mock data."""

    print("\n" + "=" * 50)
    print("Testing AutoARIMA trainer...")

    # Create mock data loader
    mock_loader = MockDataLoader()

    # Test AutoARIMA trainer
    trainer = StatisticalTrainer(model_name="AutoARIMA")

    # AutoARIMA parameters
    model_params = {
        "start_p": 0,
        "start_q": 0,
        "max_p": 2,
        "max_q": 2,
        "seasonal": False,
        "stepwise": True,
        "suppress_warnings": True,
    }

    print("Training AutoARIMA model...")
    try:
        results = trainer.train(
            data=mock_loader, patient_ids=["patient_001"], model_params=model_params
        )

        print("Training completed successfully!")
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"Error during AutoARIMA training: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""

    print("Testing Statistical Training Pipeline Components")
    print("=" * 60)

    # Test ARIMA
    arima_success = test_statistical_trainer()

    # Test AutoARIMA
    autoarima_success = test_autoarima_trainer()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"ARIMA Test: {'PASSED' if arima_success else 'FAILED'}")
    print(f"AutoARIMA Test: {'PASSED' if autoarima_success else 'FAILED'}")

    if arima_success and autoarima_success:
        print(
            "\nAll tests passed! Your statistical training pipeline is working correctly."
        )
        return 0
    else:
        print("\nSome tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
