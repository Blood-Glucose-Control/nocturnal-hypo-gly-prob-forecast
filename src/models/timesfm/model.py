"""
TimesFM model implementation using the base TSFM framework.

This module provides zero-shot forecasting and finetuning using Google's TimesFM
foundation model. TimesFM is a pretrained decoder-only transformer for time series
forecasting.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Local imports
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.models.timesfm.config import TimesFMConfig
from src.utils.logging_helper import info_print, error_print

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TimesFMDataset(Dataset):
    """PyTorch Dataset for TimesFM finetuning with sliding windows.

    Creates samples of (x_context, input_padding, freq, x_future) from time series data.
    This format matches what TimesFM's internal model expects for training.

    Args:
        series: 1D numpy array of time series values
        context_length: Number of past timesteps for context
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0=high/5-min, 1=medium/hourly, 2=low/weekly+)
    """

    def __init__(
        self,
        series: np.ndarray,
        context_length: int,
        horizon_length: int,
        freq_type: int = 0,
    ):
        if freq_type not in [0, 1, 2]:
            raise ValueError("freq_type must be 0, 1, or 2")

        self.series = series.astype(np.float32)
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        """Create sliding window samples from the time series."""
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        total_length = self.context_length + self.horizon_length

        for start_idx in range(0, len(self.series) - total_length + 1):
            end_idx = start_idx + self.context_length
            x_context = self.series[start_idx:end_idx]
            x_future = self.series[end_idx : end_idx + self.horizon_length]
            self.samples.append((x_context, x_future))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x_context, input_padding, freq, x_future) for training."""
        x_context, x_future = self.samples[index]

        x_context_t = torch.tensor(x_context, dtype=torch.float32)
        x_future_t = torch.tensor(x_future, dtype=torch.float32)

        # Padding mask: zeros indicate valid data (no padding)
        input_padding = torch.zeros_like(x_context_t)
        # Frequency tensor
        freq = torch.tensor([self.freq_type], dtype=torch.long)

        return x_context_t, input_padding, freq, x_future_t


class TimesFMForecaster(BaseTimeSeriesFoundationModel):
    """TimesFM forecaster implementation for zero-shot inference and finetuning.

    TimesFM is a pretrained time series foundation model from Google Research.
    This implementation provides both zero-shot forecasting and finetuning capability.

    Example:
        >>> config = TimesFMConfig(context_length=512, horizon_length=72)
        >>> model = TimesFMForecaster(config)
        >>> predictions = model.predict(data_df, prediction_length=72)
    """

    def __init__(
        self, config: TimesFMConfig, lora_config=None, distributed_config=None
    ):
        """Initialize the TimesFM forecaster.

        Args:
            config: TimesFM configuration object
            lora_config: LoRA configuration (ignored for TimesFM)
            distributed_config: Configuration for distributed training
        """
        # Call parent (_initialize_model)
        super().__init__(config, lora_config, distributed_config)

        self.config: TimesFMConfig = self.config

    # Abstract property implementations
    @property
    def training_backend(self) -> TrainingBackend:
        """Return the training backend for TimesFM.

        Returns:
            TrainingBackend.CUSTOM since TimesFM uses its own inference logic.
        """
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        """Check if TimesFM supports LoRA fine-tuning.

        Returns:
            False by default, but could be True with adapter frameworks like AdaPTS.
        """
        return False

    def predict(
        self, data: pd.DataFrame, prediction_length: Optional[int] = None
    ) -> np.ndarray:
        """Make predictions given context data.

        Args:
            data: DataFrame with 'bg_mM' column containing context window
            prediction_length: Number of steps to forecast (defaults to config.horizon_length)

        Returns:
            Forecast as 1D numpy array
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        prediction_length = prediction_length or self.config.horizon_length

        # Extract BG values
        bg_col = "bg_mM"
        if bg_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{bg_col}' column")

        context = data[bg_col].dropna().values.astype(np.float32)

        # Take last context_length values
        context = context[-self.config.context_length :]

        # TimesFM v2.5 API: forecast(horizon, inputs)
        # Returns tuple of (point_forecast, quantile_forecasts)
        point_forecast, _ = self.model.forecast(
            horizon=prediction_length,
            inputs=[context],  # List of 1D numpy arrays
        )

        # Extract forecast from result (first element since we passed single input)
        forecast = point_forecast[0]

        # Ensure it's a 1D numpy array
        if forecast.ndim > 1:
            forecast = forecast.flatten()

        # Truncate to requested prediction_length if needed
        if len(forecast) > prediction_length:
            forecast = forecast[:prediction_length]

        return forecast

    def _initialize_model(self) -> None:
        """Load the TimesFM model from HuggingFace."""
        info_print("Initializing TimesFM model...")

        try:
            import timesfm

            # Determine device
            self.device = (
                "cuda"
                if torch.cuda.is_available() and not self.config.use_cpu
                else "cpu"
            )

            # Initialize TimesFM using the correct v2.5 API
            # TimesFM_2p5_200M_torch is the main model class
            checkpoint = self.config.checkpoint_path or "google/timesfm-2.5-200m-pytorch"
            info_print(f"Loading TimesFM from: {checkpoint}")

            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint)

            # Compile the model with forecast config (required for v2.5)
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=self.config.context_length,
                    max_horizon=self.config.horizon_length,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )

            info_print(f"TimesFM model initialized and compiled on {self.device}")

        except ImportError:
            error_print(
                "timesfm package not installed. Install with: pip install timesfm"
            )
            raise
        except Exception as e:
            error_print(f"Failed to initialize TimesFM: {e}")
            raise

    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare DataLoaders for finetuning.

        Extracts bg_mM values from DataFrame, splits into train/val,
        and creates TimesFMDataset instances.

        Args:
            train_data: DataFrame with 'bg_mM' column, or dict of patient DataFrames

        Returns:
            Tuple of (train_loader, val_loader, None) - test_loader is None
        """
        info_print("Preparing data for TimesFM finetuning...")

        # Handle dict of patient DataFrames
        if isinstance(train_data, dict):
            # Concatenate all patient data
            all_values = []
            for patient_id, df in train_data.items():
                if "bg_mM" in df.columns:
                    values = df["bg_mM"].dropna().values
                    all_values.append(values)
            series = np.concatenate(all_values)
        elif isinstance(train_data, pd.DataFrame):
            if "bg_mM" not in train_data.columns:
                raise ValueError("DataFrame must contain 'bg_mM' column")
            series = train_data["bg_mM"].dropna().values
        else:
            raise ValueError(f"train_data must be DataFrame or dict, got {type(train_data)}")

        series = series.astype(np.float32)
        info_print(f"Total samples: {len(series):,}")

        # Split into train/val based on config.train_split
        train_size = int(len(series) * self.config.train_split)
        train_series = series[:train_size]
        val_series = series[train_size:]

        info_print(f"Train samples: {len(train_series):,}, Val samples: {len(val_series):,}")

        # Create datasets
        train_dataset = TimesFMDataset(
            series=train_series,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
        )
        val_dataset = TimesFMDataset(
            series=val_series,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
        )

        info_print(f"Train windows: {len(train_dataset):,}, Val windows: {len(val_dataset):,}")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, None

    def _save_checkpoint(self, output_dir: str) -> None:
        """Save model checkpoint.

        For zero-shot mode, saves the config so the model can be reloaded
        with the same settings. For finetuned models, also saves the model weights.

        Args:
            output_dir: Directory to save checkpoint
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save TimesFM-specific config
        timesfm_config_path = os.path.join(output_dir, "timesfm_config.json")
        config_dict = {
            "checkpoint_path": self.config.checkpoint_path,
            "context_length": self.config.context_length,
            "horizon_length": self.config.horizon_length,
            "input_patch_len": self.config.input_patch_len,
            "output_patch_len": self.config.output_patch_len,
            "num_layers": self.config.num_layers,
            "model_dims": self.config.model_dims,
            "backend": self.config.backend,
            "use_cpu": self.config.use_cpu,
            "is_finetuned": self.is_fitted,
        }
        with open(timesfm_config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        info_print(f"TimesFM config saved to {timesfm_config_path}")

        # Save finetuned model weights if fitted
        if self.is_fitted and self.model is not None:
            try:
                from safetensors.torch import save_file

                # Get the internal PyTorch module's state dict
                internal_model = self.model.model
                state_dict = internal_model.state_dict()

                # Save using safetensors format
                weights_path = os.path.join(output_dir, "timesfm_finetuned.safetensors")
                save_file(state_dict, weights_path)
                info_print(f"Finetuned weights saved to {weights_path}")

            except ImportError:
                # Fallback to PyTorch native format
                internal_model = self.model.model
                weights_path = os.path.join(output_dir, "timesfm_finetuned.pt")
                torch.save(internal_model.state_dict(), weights_path)
                info_print(f"Finetuned weights saved to {weights_path} (PyTorch format)")

    def _load_checkpoint(self, model_dir: str) -> None:
        """Load model checkpoint.

        Loads the saved config and finetuned weights if available.
        For zero-shot checkpoints, only config is loaded and weights come from HuggingFace.

        Args:
            model_dir: Directory containing saved checkpoint
        """
        timesfm_config_path = os.path.join(model_dir, "timesfm_config.json")

        if os.path.exists(timesfm_config_path):
            with open(timesfm_config_path, "r") as f:
                saved_config = json.load(f)

            # Update config with saved values
            if saved_config.get("checkpoint_path"):
                self.config.checkpoint_path = saved_config["checkpoint_path"]

            info_print(f"TimesFM config loaded from {timesfm_config_path}")

            # Check if this is a finetuned checkpoint
            is_finetuned = saved_config.get("is_finetuned", False)

            if is_finetuned:
                # Try to load finetuned weights
                safetensors_path = os.path.join(model_dir, "timesfm_finetuned.safetensors")
                pytorch_path = os.path.join(model_dir, "timesfm_finetuned.pt")

                if os.path.exists(safetensors_path):
                    try:
                        from safetensors.torch import load_file

                        state_dict = load_file(safetensors_path)
                        self.model.model.load_state_dict(state_dict)
                        self.is_fitted = True
                        info_print(f"Finetuned weights loaded from {safetensors_path}")
                    except Exception as e:
                        error_print(f"Failed to load safetensors weights: {e}")
                        raise

                elif os.path.exists(pytorch_path):
                    state_dict = torch.load(pytorch_path, map_location=self.device)
                    self.model.model.load_state_dict(state_dict)
                    self.is_fitted = True
                    info_print(f"Finetuned weights loaded from {pytorch_path}")

                else:
                    error_print(
                        f"Config indicates finetuned model but no weights found in {model_dir}"
                    )
                    raise FileNotFoundError(f"No finetuned weights found in {model_dir}")
        else:
            info_print(f"No TimesFM config found at {model_dir}, using default config")

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        """Execute TimesFM finetuning.

        Finetunes the internal PyTorch model using the training data.

        Args:
            train_data: Training data (DataFrame or dict of DataFrames)
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments

        Returns:
            Dictionary with training history (train_loss, val_loss per epoch)
        """
        info_print("Starting TimesFM finetuning...")

        # Prepare data loaders
        train_loader, val_loader, _ = self._prepare_training_data(train_data)

        # Access the internal PyTorch module
        internal_model = self.model.model  # TimesFM_2p5_200M_torch_module (nn.Module)
        internal_model.train()

        # Get device from internal model
        device = next(internal_model.parameters()).device
        info_print(f"Training on device: {device}")

        # Setup optimizer
        optimizer = torch.optim.Adam(
            internal_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Loss function (MSE)
        def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Compute MSE loss between predictions and targets."""
            return torch.mean((predictions - targets) ** 2)

        # Training history
        history = {"train_loss": [], "val_loss": []}

        info_print(f"Training for {self.config.num_epochs} epochs...")
        info_print(f"Batch size: {self.config.batch_size}, LR: {self.config.learning_rate}")

        for epoch in range(self.config.num_epochs):
            # Training phase
            internal_model.train()
            train_losses = []

            for batch in train_loader:
                x_context, x_padding, freq, x_future = [t.to(device) for t in batch]

                # Forward pass through internal model
                # Returns: (input_emb, output_emb, output_ts, output_quantile), caches
                outputs, _ = internal_model(x_context, x_padding)
                output_ts = outputs[2]  # Point forecast: shape (B, num_patches, output_patch_len, num_quantiles)

                # Get the last patch prediction (mean/point estimate is index 0)
                # Shape: (B, output_patch_len)
                predictions = output_ts[:, -1, :, 0]

                # Truncate predictions to match target length if needed
                pred_len = min(predictions.shape[-1], x_future.shape[-1])
                predictions = predictions[:, :pred_len]
                targets = x_future[:, :pred_len]

                # Compute loss
                loss = compute_loss(predictions, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            internal_model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    x_context, x_padding, freq, x_future = [t.to(device) for t in batch]

                    outputs, _ = internal_model(x_context, x_padding)
                    output_ts = outputs[2]
                    predictions = output_ts[:, -1, :, 0]

                    pred_len = min(predictions.shape[-1], x_future.shape[-1])
                    predictions = predictions[:, :pred_len]
                    targets = x_future[:, :pred_len]

                    loss = compute_loss(predictions, targets)
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            info_print(
                f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                f"Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
            )

        # Mark as fitted before saving
        self.is_fitted = True

        # Save the finetuned model
        os.makedirs(output_dir, exist_ok=True)
        self._save_checkpoint(output_dir)

        info_print(f"Finetuning complete. Model saved to {output_dir}")

        return {"history": history, "train_loss": history["train_loss"], "val_loss": history["val_loss"]}


def create_timesfm_model(
    checkpoint_path: Optional[str] = None,
    context_length: int = 512,
    horizon_length: int = 128,
    backend: str = "cpu",
    **kwargs,
) -> TimesFMForecaster:
    """Factory function to create a TimesFM model with sensible defaults.

    Args:
        checkpoint_path: Path to TimesFM checkpoint (default: google/timesfm-2.5-200m-pytorch).
        context_length: Input sequence length.
        horizon_length: Output prediction horizon.
        backend: Backend to use ('cpu', 'gpu', 'tpu').
        **kwargs: Additional configuration parameters.

    Returns:
        Initialized TimesFMForecaster instance.

    Example:
        >>> model = create_timesfm_model(
        ...     checkpoint_path="google/timesfm-2.5-200m-pytorch",
        ...     context_length=512,
        ...     horizon_length=128,
        ...     backend="gpu"
        ... )
    """
    config = TimesFMConfig(
        checkpoint_path=checkpoint_path,
        context_length=context_length,
        horizon_length=horizon_length,
        backend=backend,
        **kwargs,
    )

    return TimesFMForecaster(config)
