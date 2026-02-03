"""
Sundial model implementation using the base TSFM framework.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
)

# Local imports
from src.models.sundial.config import SundialConfig
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.utils.logging_helper import info_print, error_print

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SundialForecaster(BaseTimeSeriesFoundationModel):
    """Sundial forecaster implementation."""

    def __init__(self, config: SundialConfig, lora_config=None, distributed_config=None):
        """Initialize the Sundial forecaster.
        Args:
            config: Sundial configuration object
            lora_config: LoRA configuration (ignored for Sundial)
            distributed_config: Configuration for distributed training
        """
        # Call parent (this will call _initialize_model)
        super().__init__(config, lora_config, distributed_config)

        # Type annotation to help linter understand config type
        self.config: SundialConfig = self.config

    def _initialize_model(self) -> None:
        """Load the Sundial model from HuggingFace."""
        info_print("Initializing Sundial model from thuml/sundial-base-128m...")

        self.model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True
        )
        if self.model is None:
            error_print("Failed to load Sundial model.")
            raise ValueError("Model loading failed.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        info_print(f"Sundial model initialized on {self.device}")

    # Properties
    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.TRANSFORMERS

    @property
    def supports_lora(self) -> bool:
        return False
    
    def predict(self, data: Any, batch_size: Optional[int] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")

        seqs = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            samples = self.model.generate(
                seqs,
                max_new_tokens=batch_size,
                num_samples=self.config.num_samples
            )  # shape: (1, num_samples, prediction_length)

        samples = samples.cpu().numpy()[0]  # (num_samples, prediction_length)

        median = np.median(samples, axis=0)

        return median

    def predict_zero_shot(
        self,
        data: Any,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Make zero-shot predictions on DataFrame input.

        Provides same interface as TTM's predict_zero_shot for workflow compatibility.
        Extracts BG column from DataFrame and runs Sundial inference.

        Args:
            data: Input data for prediction (DataFrame with bg_mM column)
            batch_size: Batch size for prediction (unused, kept for API compatibility)

        Returns:
            Model predictions as numpy array with shape (samples, forecast_length, 1)
        """
        info_print("Making zero-shot predictions with Sundial")

        # Convert dict format to DataFrame if needed
        if isinstance(data, dict):
            data = pd.concat(data.values(), ignore_index=True)

        # Extract BG column (Sundial is univariate)
        bg_col = "bg_mM"
        if bg_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{bg_col}' column for Sundial")

        # Get context (all data up to forecast point)
        context = data[bg_col].values[:self.config.context_length]

        info_print(f"Context length: {len(context)}")
        info_print(f"Forecast length: {self.config.forecast_length}")

        # Run prediction using low-level predict()
        predictions = self.predict(context, batch_size=self.config.forecast_length)

        # Reshape to match TTM output format: (samples, forecast_length, channels)
        # For single sample, single channel: (1, forecast_length, 1)
        predictions = predictions.reshape(1, -1, 1)

        info_print(f"Zero-shot predictions shape: {predictions.shape}")

        return predictions

    # Stub implementations for abstract methods (zero-shot only)
    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        raise NotImplementedError("Sundial zero-shot mode does not support training")

    def _save_checkpoint(self, output_dir: str) -> None:
        pass  # No-op for zero-shot

    def _load_checkpoint(self, model_dir: str) -> None:
        pass  # No-op for zero-shot

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError("Sundial zero-shot mode does not support training")
