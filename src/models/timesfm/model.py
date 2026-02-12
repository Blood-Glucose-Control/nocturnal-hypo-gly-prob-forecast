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
        stride: Step size between windows. Default=horizon_length for non-overlapping.
                Use stride=1 for maximum overlap (slow), stride=horizon_length for fast training.
        max_samples: Maximum number of samples to create. None=unlimited.
                     Samples are uniformly distributed across the series if capped.
    """

    def __init__(
        self,
        series: np.ndarray,
        context_length: int,
        horizon_length: int,
        freq_type: int = 0,
        stride: Optional[int] = None,
        max_samples: Optional[int] = None,
    ):
        if freq_type not in [0, 1, 2]:
            raise ValueError("freq_type must be 0, 1, or 2")

        self.series = series.astype(np.float32)
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type
        # Default stride to horizon_length for non-overlapping windows
        self.stride = stride if stride is not None else horizon_length
        self.max_samples = max_samples
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        """Create sliding window samples from the time series with stride."""
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        total_length = self.context_length + self.horizon_length

        # Collect all possible window start indices
        all_indices = list(range(0, len(self.series) - total_length + 1, self.stride))

        # If max_samples is set and we have more windows, subsample uniformly
        if self.max_samples is not None and len(all_indices) > self.max_samples:
            # Uniformly sample indices to maintain temporal distribution
            step = len(all_indices) / self.max_samples
            selected_indices = [all_indices[int(i * step)] for i in range(self.max_samples)]
            all_indices = selected_indices

        for start_idx in all_indices:
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

        # Convert to list of lists format expected by v1.x
        context_list = [context.tolist()]
        point_forecast, _ = self.model.forecast(
            inputs=context_list,
            freq=[0],  # 0 = high frequency (5-min data)
        )
        forecast = point_forecast[0]

        # Ensure it's a 1D numpy array
        if isinstance(forecast, np.ndarray):
            if forecast.ndim > 1:
                forecast = forecast.flatten()
        else:
            forecast = np.array(forecast).flatten()

        # Truncate to requested prediction_length if needed
        if len(forecast) > prediction_length:
            forecast = forecast[:prediction_length]

        return forecast

    def _initialize_model(self) -> None:
        """Load the TimesFM model.

        Uses the v1.x API (TimesFmHparams/TimesFmCheckpoint) which is required
        for finetuning and provides access to the internal model via _model.
        """
        info_print("Initializing TimesFM model...")

        try:
            import timesfm

            cuda_available = torch.cuda.is_available()
            self.device = (
                "cuda"
                if cuda_available and not self.config.use_cpu
                else "cpu"
            )
            info_print(f"Selected device: {self.device}")

            if not (hasattr(timesfm, "TimesFm") and hasattr(timesfm, "TimesFmHparams")):
                raise AttributeError(
                    "Could not find TimesFM v1.x API classes (TimesFm, TimesFmHparams). "
                    "Install with: pip install timesfm==1.3.0"
                )

            checkpoint = self.config.checkpoint_path or "google/timesfm-2.0-500m-pytorch"
            info_print(f"Loading TimesFM from: {checkpoint}")

            # Determine backend
            backend = "gpu" if self.device == "cuda" else "cpu"

            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=self.config.per_core_batch_size,
                    horizon_len=self.config.horizon_length,
                    num_layers=self.config.num_layers,
                    use_positional_embedding=False,
                    context_len=self.config.context_length,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=checkpoint,
                ),
            )
            info_print(f"TimesFM initialized on {self.device}")

        except ImportError:
            error_print(
                "timesfm package not installed. Install with: pip install timesfm==1.3.0"
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

        stride = self.config.window_stride or self.config.horizon_length

        # Create datasets with stride and max_samples for efficient training
        train_dataset = TimesFMDataset(
            series=train_series,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
            stride=stride,
            max_samples=self.config.max_train_windows,
        )
        val_dataset = TimesFMDataset(
            series=val_series,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
            stride=stride,
            max_samples=self.config.max_val_windows,
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
                internal_model = self.model._model
                state_dict = internal_model.state_dict()

                # Save using safetensors format
                weights_path = os.path.join(output_dir, "timesfm_finetuned.safetensors")
                save_file(state_dict, weights_path)
                info_print(f"Finetuned weights saved to {weights_path}")

            except ImportError:
                # Fallback to PyTorch native format
                internal_model = self.model._model
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
                        self.model._model.load_state_dict(state_dict)
                        self.is_fitted = True
                        info_print(f"Finetuned weights loaded from {safetensors_path}")
                    except Exception as e:
                        error_print(f"Failed to load safetensors weights: {e}")
                        raise

                elif os.path.exists(pytorch_path):
                    state_dict = torch.load(pytorch_path, map_location=self.device)
                    self.model._model.load_state_dict(state_dict)
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
        """Execute TimesFM finetuning using official TimesFMFinetuner.

        Args:
            train_data: Training data (DataFrame or dict of DataFrames)
            output_dir: Directory for saving checkpoints
            **kwargs: Additional arguments

        Returns:
            Dictionary with training history
        """
        info_print("Starting TimesFM finetuning with official finetuner...")

        # Import the official finetuning code
        try:
            from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
        except ImportError:
            error_print(
                "Official finetuning code not found. "
                "Please copy finetuning/ folder from timesfm repo."
            )
            raise

        # Prepare time series data
        info_print("Preparing training data...")
        if isinstance(train_data, dict):
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

        # Split into train/val
        train_size = int(len(series) * self.config.train_split)
        train_series = series[:train_size]
        val_series = series[train_size:]
        info_print(f"Train: {len(train_series):,}, Val: {len(val_series):,}")

        # Determine stride (default to horizon_length for non-overlapping windows)
        stride = self.config.window_stride or self.config.horizon_length
        info_print(f"Window stride: {stride} (horizon_length={self.config.horizon_length})")
        info_print(f"Max train windows: {self.config.max_train_windows:,}")
        info_print(f"Max val windows: {self.config.max_val_windows:,}")

        # Create PyTorch Dataset objects for finetuning
        # Uses stride and max_samples to control dataset size
        train_dataset = TimesFMDataset(
            series=train_series,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
            stride=stride,
            max_samples=self.config.max_train_windows,
        )
        val_dataset = TimesFMDataset(
            series=val_series,
            context_length=self.config.context_length,
            horizon_length=self.config.horizon_length,
            freq_type=self.config.freq_type,
            stride=stride,
            max_samples=self.config.max_val_windows,
        )

        info_print(f"Train dataset: {len(train_dataset):,} windows (capped from stride)")
        info_print(f"Val dataset: {len(val_dataset):,} windows (capped from stride)")

        # Configure finetuning
        ft_config = FinetuningConfig(
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            freq_type=self.config.freq_type,
            use_wandb=False,
            device="cuda" if self.device == "cuda" else "cpu",
        )

        # Get the internal PyTorch model for finetuning
        internal_model = self.model._model
        info_print(f"Internal model type: {type(internal_model).__name__}")

        finetuner = TimesFMFinetuner(
            model=internal_model,
            config=ft_config,
        )

        # Run finetuning
        info_print(f"Training for {self.config.num_epochs} epochs...")
        info_print(f"Batch size: {self.config.batch_size}, LR: {self.config.learning_rate}")

        results = finetuner.finetune(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        self.is_fitted = True

        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        self._save_checkpoint(output_dir)

        info_print(f"Finetuning complete. Model saved to {output_dir}")

        return results


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
