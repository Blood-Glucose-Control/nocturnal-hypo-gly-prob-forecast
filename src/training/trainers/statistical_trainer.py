"""Trainer for statistical models like ARIMA, AutoARIMA, etc."""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

from .base_trainer import BaseTrainer
from ..checkpointing.checkpoint_manager import CheckpointManager


class StatisticalTrainer(BaseTrainer):
    """Trainer for statistical forecasting models."""

    def __init__(self, model_name: str, use_gpu: bool = True):
        super().__init__(model_name, use_gpu)
        self.model_classes = {
            "ARIMA": ARIMA,
            "AutoARIMA": AutoARIMA,
            "StatsForecastAutoARIMA": StatsForecastAutoARIMA,
            "STLForecaster": STLForecaster,
            "NaiveForecaster": NaiveForecaster,
        }

    def _create_model(self, **kwargs) -> Any:
        """Create the statistical model with given parameters."""
        if self.model_name not in self.model_classes:
            raise ValueError(
                f"Unknown statistical model: {self.model_name}. "
                f"Available models: {list(self.model_classes.keys())}"
            )

        model_class = self.model_classes[self.model_name]

        # Set default parameters for ARIMA if none provided
        if self.model_name == "ARIMA" and not kwargs:
            kwargs = {"order": (1, 1, 1)}
        elif self.model_name == "AutoARIMA" and not kwargs:
            kwargs = {
                "start_p": 0,
                "start_q": 0,
                "max_p": 3,
                "max_q": 3,
                "seasonal": False,
            }

        return model_class(**kwargs)

    def train(
        self,
        data: Any,
        patient_ids: Optional[List[str]] = None,
        epochs: int = 100,  # Not used for statistical models, kept for API consistency
        checkpoint_manager: Optional[CheckpointManager] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train the statistical model on the provided data.

        Args:
            data: Data loader containing the training data
            patient_ids: List of patient IDs to train on
            epochs: Not used for statistical models (kept for API consistency)
            checkpoint_manager: Optional checkpoint manager
            model_params: Parameters for the statistical model

        Returns:
            Dictionary containing training results and metrics
        """
        if model_params is None:
            model_params = {}

        print(f"Training {self.model_name} model...")

        # Get the training data
        training_data = self._prepare_training_data(data, patient_ids)

        if training_data.empty:
            raise ValueError("No training data available")

        # Create and train the model
        self.model = self._create_model(**model_params)

        try:
            # For statistical models, we typically train on the entire time series
            print(f"Training data type: {type(training_data)}")
            print(f"Training data shape: {training_data.shape}")

            if isinstance(training_data, pd.Series):
                # Data is already a series, use it directly
                y_train = training_data
                print(f"Using Series data directly: {len(y_train)} data points")
            elif isinstance(training_data, pd.DataFrame):
                # Data is a DataFrame, need to select target column
                print(f"Available columns: {list(training_data.columns)}")
                print(f"Data head:\n{training_data.head()}")

                # Try different possible column names for glucose
                glucose_columns = [
                    "glucose",
                    "bg",
                    "blood_glucose",
                    "cgm",
                    "sensor_glucose",
                ]
                target_column = None

                for col in glucose_columns:
                    if col in training_data.columns:
                        target_column = col
                        break

                if target_column is None:
                    # If no glucose column found, use the first numeric column
                    numeric_cols = training_data.select_dtypes(
                        include=["float64", "int64"]
                    ).columns
                    if len(numeric_cols) > 0:
                        target_column = numeric_cols[0]
                        print(
                            f"No glucose column found, using '{target_column}' as target"
                        )
                    else:
                        raise ValueError(
                            f"No suitable target column found. Available columns: {list(training_data.columns)}"
                        )

                y_train = training_data[target_column]
            else:
                raise ValueError(f"Unexpected data type: {type(training_data)}")

            print(f"Training on {len(y_train)} data points...")
            self.model.fit(y_train)

            # Calculate some basic training metrics
            in_sample_predictions = self.model.predict(fh=range(1, len(y_train) + 1))
            mse = ((y_train - in_sample_predictions) ** 2).mean()
            mae = (abs(y_train - in_sample_predictions)).mean()

            results = {
                "model_name": self.model_name,
                "training_samples": len(y_train),
                "patient_ids": patient_ids if patient_ids else "all",
                "in_sample_mse": float(mse),
                "in_sample_mae": float(mae),
                "model_params": model_params,
                "training_complete": True,
            }

            print(f"Training complete. MSE: {mse:.4f}, MAE: {mae:.4f}")
            return results

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def _prepare_training_data(
        self, data_loader: Any, patient_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Prepare training data from the data loader."""
        # This will depend on your data loader structure
        # Assuming the data loader has a method to get processed data
        if hasattr(data_loader, "get_data"):
            if patient_ids:
                # Get data for specific patients
                all_data = []
                for patient_id in patient_ids:
                    patient_data = data_loader.get_data(patient_id)
                    if patient_data is not None and not patient_data.empty:
                        all_data.append(patient_data)
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                # Get all available data
                return data_loader.get_data()
        elif hasattr(data_loader, "processed_data"):
            # Handle case where data loader has processed_data attribute
            if patient_ids:
                all_data = []
                for patient_id in patient_ids:
                    if patient_id in data_loader.processed_data:
                        all_data.append(data_loader.processed_data[patient_id])
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
                else:
                    return pd.DataFrame()
            else:
                # Combine all patient data
                all_data = list(data_loader.processed_data.values())
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
                else:
                    return pd.DataFrame()
        else:
            raise ValueError(
                "Data loader doesn't have expected methods (get_data or processed_data)"
            )

    def save_model(self, model: Any, path: Path) -> None:
        """Save the statistical model to disk."""
        model_file = path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_file}")

    def load_model(self, path: Path) -> Any:
        """Load the statistical model from disk."""
        model_file = path / "model.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return model

    def predict(self, model: Any, data: Any, forecast_horizon: int = 24) -> Any:
        """Make predictions with the statistical model.

        Args:
            model: Trained statistical model
            data: Input data (not used for pure time series forecasting)
            forecast_horizon: Number of steps to forecast ahead

        Returns:
            Model predictions
        """
        return model.predict(fh=range(1, forecast_horizon + 1))
