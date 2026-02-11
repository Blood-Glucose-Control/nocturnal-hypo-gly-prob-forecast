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

    config: SundialConfig  # Override base class typing

    def __init__(
        self, config: SundialConfig, lora_config=None, distributed_config=None
    ):
        """Initialize the Sundial forecaster.
        Args:
            config: Sundial configuration object
            lora_config: LoRA configuration (ignored for Sundial)
            distributed_config: Configuration for distributed training
        """
        # Call parent (this will call _initialize_model)
        super().__init__(config, lora_config, distributed_config)

    def _initialize_model(self) -> None:
        """Load the Sundial model from HuggingFace."""
        info_print("Initializing Sundial model from thuml/sundial-base-128m...")

        self.model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m", trust_remote_code=True
        )
        if self.model is None:
            error_print("Failed to load Sundial model.")
            raise ValueError("Model loading failed.")

        use_cuda = torch.cuda.is_available() and not getattr(
            self.config, "use_cpu", False
        )
        self.device = "cuda" if use_cuda else "cpu"
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

    def predict(self, data: pd.DataFrame, prediction_length: int) -> np.ndarray:
        """Make predictions given context data.

        Args:
            data: DataFrame with 'bg_mM' column containing context window
            prediction_length: Number of steps to forecast

        Returns:
            Forecast as 1D numpy array of length prediction_length
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")

        # Extract BG values from DataFrame
        bg_col = "bg_mM"
        if bg_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{bg_col}' column")

        context = data[bg_col].values

        # Run inference
        seqs = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            samples = self.model.generate(
                seqs,
                max_new_tokens=prediction_length,
                num_samples=self.config.num_samples,
            )  # shape: (1, num_samples, prediction_length)

        samples = samples.cpu().numpy()[0]  # (num_samples, prediction_length)

        # Return median of samples
        return np.median(samples, axis=0)

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
