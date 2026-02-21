"""Model factory for creating forecasting models.

This module provides a unified factory interface for creating different
time series forecasting models with their configurations.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from src.models.base import BaseTimeSeriesFoundationModel, ModelConfig

logger = logging.getLogger(__name__)


def create_model_and_config(
    model_type: str, checkpoint: Optional[str] = None, **kwargs
) -> Tuple[BaseTimeSeriesFoundationModel, ModelConfig]:
    """Factory function to create model and config based on type.

    Args:
        model_type: One of 'sundial', 'ttm', 'chronos', 'moirai'
        checkpoint: Optional path to fine-tuned checkpoint
        **kwargs: Additional config parameters (e.g., num_samples, forecast_length)

    Returns:
        Tuple of (model, config)

    Note:
        When checkpoint is provided, the saved config is used as base.
        CLI overrides are validated:
        - batch_size: always allowed (inference-only setting)
        - forecast_length: allowed if <= saved value (truncate predictions)
        - context_length: must match saved value (affects model architecture)
    """
    if model_type == "sundial":
        from src.models.sundial import SundialForecaster, SundialConfig

        if checkpoint:
            # Load model with saved config
            model = SundialForecaster.load(checkpoint)
            config = model.config

            # Apply valid overrides
            if "batch_size" in kwargs:
                config.batch_size = kwargs["batch_size"]
            if "forecast_length" in kwargs:
                requested = kwargs["forecast_length"]
                if requested <= config.forecast_length:
                    logger.info(
                        f"Overriding forecast_length: {config.forecast_length} -> {requested}"
                    )
                    config.forecast_length = requested
                else:
                    logger.warning(
                        f"Cannot increase forecast_length beyond trained value "
                        f"({config.forecast_length}). Using saved value."
                    )
        else:
            config = SundialConfig(
                forecast_length=kwargs.get("forecast_length", 96),
                num_samples=kwargs.get("num_samples", 100),
            )
            model = SundialForecaster(config)
        return model, config

    elif model_type == "ttm":
        from src.models.ttm import TTMForecaster, TTMConfig
        from src.models.base.base_model import ModelConfig as BaseModelConfig
        import dataclasses

        if checkpoint:
            # Load config from training_metadata.json (config.json is overwritten by TSFM)
            metadata_path = os.path.join(checkpoint, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                saved_config = metadata.get("config", {})
            else:
                logger.warning(
                    f"No training_metadata.json found in {checkpoint}, using defaults"
                )
                saved_config = {}

            # Get valid field names from ModelConfig (TTMConfig uses **kwargs)
            base_fields = {f.name for f in dataclasses.fields(BaseModelConfig)}
            # TTM-specific fields handled by TTMConfig.__init__
            ttm_fields = {
                "scaler_type",
                "input_features",
                "target_features",
                "split_config",
                "fewshot_percent",
                "num_input_channels",
                "num_output_channels",
                "prediction_filter_length",
                "resolution_min",
                "use_tracking_callback",
                "find_optimal_lr",
                "logging_dir",
            }
            valid_fields = base_fields | ttm_fields | {"model_path", "training_mode"}

            # Filter to only known fields
            filtered_config = {
                k: v for k, v in saved_config.items() if k in valid_fields
            }
            filtered_config["model_path"] = (
                checkpoint  # Override to load from checkpoint
            )
            filtered_config["training_mode"] = "fine_tune"

            # Log any ignored fields
            ignored = set(saved_config.keys()) - valid_fields
            if ignored:
                logger.debug(
                    f"Ignoring unknown config fields from checkpoint: {ignored}"
                )

            config = TTMConfig(**filtered_config)

            # Apply valid overrides
            if "batch_size" in kwargs:
                config.batch_size = kwargs["batch_size"]

            if "forecast_length" in kwargs:
                requested = kwargs["forecast_length"]
                if requested <= config.forecast_length:
                    logger.info(
                        f"Overriding forecast_length: {config.forecast_length} -> {requested}"
                    )
                    config.forecast_length = requested
                else:
                    logger.warning(
                        f"Cannot increase forecast_length beyond trained value "
                        f"({config.forecast_length}). Using saved value."
                    )

            if "context_length" in kwargs:
                requested = kwargs["context_length"]
                if requested != config.context_length:
                    logger.warning(
                        f"context_length mismatch: requested {requested}, "
                        f"model trained with {config.context_length}. "
                        f"Using saved value."
                    )

            # Create model with config and load checkpoint
            model = TTMForecaster(config)
            model._load_checkpoint(checkpoint)
            model.is_fitted = True
        else:
            config = TTMConfig(
                model_path=kwargs.get(
                    "model_path", "ibm-granite/granite-timeseries-ttm-r2"
                ),
                context_length=kwargs.get("context_length", 512),
                forecast_length=kwargs.get("forecast_length", 96),
                batch_size=kwargs.get("batch_size", 256),
                training_mode="zero_shot",
                freeze_backbone=True,
            )
            model = TTMForecaster(config)
        return model, config

    elif model_type == "chronos":
        raise NotImplementedError("Chronos model not yet implemented")

    elif model_type == "moirai":
        raise NotImplementedError("Moirai model not yet implemented")

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: sundial, ttm, chronos, moirai"
        )
