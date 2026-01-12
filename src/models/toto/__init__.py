"""
Toto model package for time series forecasting.

Provides TotoForecaster and TotoConfig for fine-tuning the Toto
time-series foundation model on blood glucose forecasting tasks.
"""

from src.models.toto.config import (
    TotoConfig,
    TotoDataConfig,
    TotoTrainingConfig,
    create_default_toto_config,
    create_toto_fine_tuning_config,
    create_toto_zero_shot_config,
)
from src.models.toto.model import TotoForecaster, TotoDataset

__all__ = [
    # Config
    "TotoConfig",
    "TotoDataConfig",
    "TotoTrainingConfig",
    "create_default_toto_config",
    "create_toto_fine_tuning_config",
    "create_toto_zero_shot_config",
    # Model
    "TotoForecaster",
    "TotoDataset",
]
