"""
Toto model implementation using the base TSFM framework.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from toto.model.toto import Toto
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster as _TotoForecaster

from src.models.toto.config import TotoConfig
from src.models.base import BaseTimeSeriesFoundationModel, TrainingBackend
from src.utils.logging_helper import info_print, error_print

logger = logging.getLogger(__name__)

INTERVAL_MINS = 5


class TotoForecaster(BaseTimeSeriesFoundationModel):
    """Toto forecaster implementation."""

    config: TotoConfig

    def __init__(
        self, config: TotoConfig, lora_config=None, distributed_config=None
    ):
        super().__init__(config, lora_config, distributed_config)

    def _initialize_model(self) -> None:
        """Load the Toto model from HuggingFace."""
        info_print("Initializing Toto model from Datadog/Toto-Open-Base-1.0...")

        toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")

        use_cuda = torch.cuda.is_available() and not getattr(
            self.config, "use_cpu", False
        )
        self.device = "cuda" if use_cuda else "cpu"
        toto.to(self.device)

        self.model = toto.model
        self.forecaster = _TotoForecaster(self.model)

        info_print(f"Toto model initialized on {self.device}")

    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.CUSTOM

    @property
    def supports_lora(self) -> bool:
        return False

    @property
    def supports_zero_shot(self) -> bool:
        return True

    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions given context data.

        Args:
            data: DataFrame with 'bg_mM' column and a DatetimeIndex
            **kwargs: Additional options (unused for zero-shot)

        Returns:
            Forecast as 1D numpy array of shape (forecast_length,)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call _initialize_model first.")

        forecast_length = self.config.forecast_length

        bg_col = "bg_mM"
        if bg_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{bg_col}' column")

        context = data[bg_col].values
        timestamps = data.index

        # Build MaskedTimeseries input
        series = (
            torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        ts_seconds = (
            torch.tensor(
                [ts.timestamp() for ts in timestamps], dtype=torch.float32
            )
            .unsqueeze(0)
            .to(self.device)
        )

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.ones_like(series, dtype=torch.bool),
            id_mask=torch.zeros_like(series),
            timestamp_seconds=ts_seconds,
            time_interval_seconds=torch.tensor(
                [INTERVAL_MINS * 60], dtype=torch.float32
            ).to(self.device),
        )

        with torch.no_grad():
            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=forecast_length,
                num_samples=self.config.num_samples,
                samples_per_batch=self.config.samples_per_batch,
            )

        return forecast.median.cpu().numpy()[0]

    # Stub implementations for abstract methods (zero-shot only for now)
    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        raise NotImplementedError("Toto training not yet implemented")

    def _save_checkpoint(self, output_dir: str) -> None:
        pass

    def _load_checkpoint(self, model_dir: str) -> None:
        pass

    def _train_model(
        self, train_data: Any, output_dir: str, **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError("Toto training not yet implemented")
