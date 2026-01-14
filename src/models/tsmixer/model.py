#!/usr/bin/env python3
"""
TSMixer model implementation using the base TSFM framework.

TSMixer is an MLP-based architecture that does NOT support LoRA
since it doesn't have transformer attention mechanisms.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from src.models.base import BaseTimeSeriesFoundationModel, ModelConfig, TrainingStrategy
from src.utils.logging_helper import info_print, error_print


class TSMixerConfig(ModelConfig):
    """Configuration for TSMixer model."""

    def __init__(self, **kwargs):
        # Extract TSMixer-specific parameters before calling parent
        tsmixer_specific_params = {
            "d_model",
            "n_blocks",
            "mixing_hidden_dim",
        }

        # Filter out TSMixer-specific params from kwargs for parent class
        base_kwargs = {
            k: v for k, v in kwargs.items() if k not in tsmixer_specific_params
        }

        # Call parent with filtered kwargs
        super().__init__(**base_kwargs)

        # Set TSMixer-specific parameters
        self.model_type = "tsmixer"

        # TSMixer architecture specifics
        self.d_model = kwargs.get("d_model", 128)
        self.n_blocks = kwargs.get("n_blocks", 4)
        self.mixing_hidden_dim = kwargs.get("mixing_hidden_dim", 256)


class TSMixerForecaster(BaseTimeSeriesFoundationModel):
    """
    TSMixer forecaster implementation.

    TSMixer is an MLP-based model that does NOT support LoRA fine-tuning
    since it lacks transformer attention mechanisms.
    """

    def __init__(
        self, config: TSMixerConfig, lora_config=None, distributed_config=None
    ):
        """Initialize TSMixer forecaster."""
        if not isinstance(config, TSMixerConfig):
            config = TSMixerConfig(**config.to_dict())

        # Warn about LoRA if it's enabled
        if lora_config and lora_config.enabled:
            info_print("WARNING: LoRA is not supported for TSMixer architecture")
            info_print("LoRA requires transformer models with attention mechanisms")
            lora_config.enabled = False

        super().__init__(config, lora_config, distributed_config)
        self.config: TSMixerConfig = self.config

    def supports_lora(self) -> bool:
        """TSMixer is MLP-based and does NOT support LoRA fine-tuning."""
        return False

    def get_training_strategy(self) -> TrainingStrategy:
        """TSMixer uses custom PyTorch training loops."""
        return TrainingStrategy.PYTORCH

    def _initialize_model(self) -> None:
        """Initialize the TSMixer model architecture."""
        try:
            info_print(f"Initializing TSMixer model from {self.config.model_path}")

            # TODO: Replace with actual TSMixer library import when available
            # Example: from some_tsmixer_library import get_tsmixer_model
            # self.model = get_tsmixer_model(
            #     model_path=self.config.model_path,
            #     context_length=self.config.context_length,
            #     forecast_length=self.config.forecast_length,
            # )

            # Placeholder until TSMixer library is integrated
            info_print("TSMixer integration not yet implemented")
            info_print("This is a placeholder following the same pattern as TTM")
            self.model = None  # Will be replaced with actual library import

        except Exception as e:
            error_print(f"Failed to initialize TSMixer model: {str(e)}")
            raise

    def _prepare_data(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Prepare data loaders for TSMixer training."""
        # TODO: Implement data preparation similar to TTM pattern
        # This would integrate with your existing data loading pipeline
        info_print("TSMixer data preparation not yet implemented")
        raise NotImplementedError(
            "TSMixer integration is a placeholder - not yet implemented"
        )

    def _create_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create TSMixer-specific training arguments."""
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            use_cpu=self.config.use_cpu,
            report_to="none",
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics for TSMixer."""
        try:
            predictions, labels = eval_pred

            # Convert to numpy arrays
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)

            # Compute basic metrics
            mse = np.mean((predictions - labels) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - labels))

            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
            }

        except Exception as e:
            error_print(f"Error computing metrics: {str(e)}")
            return {"eval_loss": float("inf")}

    def _load_model_weights(self, model_dir: str) -> None:
        """Load TSMixer model weights from directory."""
        # TODO: Implement model loading similar to TTM pattern
        # This would use the appropriate TSMixer library's loading mechanism
        info_print(f"TSMixer model loading not yet implemented for {model_dir}")
        raise NotImplementedError(
            "TSMixer integration is a placeholder - not yet implemented"
        )

    def get_tsmixer_specific_info(self) -> Dict[str, Any]:
        """Get TSMixer-specific model information."""
        info = self._get_model_info()
        info.update(
            {
                "tsmixer_specific": {
                    "d_model": self.config.d_model,
                    "n_blocks": self.config.n_blocks,
                    "mixing_hidden_dim": self.config.mixing_hidden_dim,
                    "supports_lora": False,  # TSMixer does not support LoRA
                    "architecture_type": "MLP-based",
                }
            }
        )
        return info
