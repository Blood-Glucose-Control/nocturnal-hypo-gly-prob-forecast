"""Model factory for creating forecasting models.

This module provides a unified factory interface for creating different
time series forecasting models with their configurations.
"""

import json
import logging
import os
from typing import Optional, Tuple

from src.models.base import BaseTimeSeriesFoundationModel, ModelConfig

logger = logging.getLogger(__name__)


def create_model_and_config(
    model_type: str, checkpoint: Optional[str] = None, **kwargs
) -> Tuple[BaseTimeSeriesFoundationModel, ModelConfig]:
    """Factory function to create model and config based on type.

    Args:
        model_type: One of 'sundial', 'ttm', 'chronos', 'tide', 'moirai'
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

    elif model_type == "tide":
        from src.models.tide import TiDEForecaster, TiDEConfig

        if checkpoint:
            model = TiDEForecaster.load(checkpoint)
            config = model.config

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
            config = TiDEConfig(
                context_length=kwargs.get("context_length", 512),
                forecast_length=kwargs.get("forecast_length", 72),
            )
            model = TiDEForecaster(config)
        return model, config

    elif model_type == "moirai":
        raise NotImplementedError("Moirai model not yet implemented")

    elif model_type == "timegrad":
        from src.models.timegrad import TimeGradForecaster, TimeGradConfig

        if checkpoint:
            config_path = os.path.join(checkpoint, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    saved_config = json.load(f)
                # These are set explicitly by TimeGradConfig.__init__; drop them
                # so they don't conflict with the dataclass parent's __init__.
                saved_config.pop("model_type", None)
                saved_config.pop("training_backend", None)
            else:
                logger.warning(f"No config.json found in {checkpoint}, using defaults")
                saved_config = {}

            config = TimeGradConfig(**saved_config)

            # Apply valid overrides
            if "batch_size" in kwargs:
                config.batch_size = kwargs["batch_size"]
            if "num_samples" in kwargs:
                config.num_samples = kwargs["num_samples"]
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

            model = TimeGradForecaster(config)
            model._load_checkpoint(checkpoint)
            model.is_fitted = True
        else:
            config = TimeGradConfig(
                context_length=kwargs.get("context_length", 512),
                forecast_length=kwargs.get("forecast_length", 96),
                batch_size=kwargs.get("batch_size", 64),
            )
            model = TimeGradForecaster(config)
        return model, config

    elif model_type == "timesfm":
        from src.models.timesfm import TimesFMForecaster, TimesFMConfig

        if checkpoint:
            # Handle TimesFM checkpoint structure: if path ends with /model.pt,
            # strip it to get the parent directory (HF model files are in hf_model/ subdirectory)
            checkpoint_dir = checkpoint
            if checkpoint_dir.endswith("/model.pt") or checkpoint_dir.endswith(
                "\\model.pt"
            ):
                checkpoint_dir = os.path.dirname(checkpoint_dir)

            # Try to load config from training_metadata.json first (similar to TTM)
            metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                saved_config = metadata.get("config", {})
            else:
                # Fall back to config.json in hf_model directory
                hf_model_config = os.path.join(
                    checkpoint_dir, "hf_model", "config.json"
                )
                if os.path.exists(hf_model_config):
                    with open(hf_model_config, "r") as f:
                        saved_config = json.load(f)
                else:
                    logger.warning(
                        f"No training_metadata.json or hf_model/config.json found in {checkpoint_dir}, using defaults"
                    )
                    saved_config = {}

            # Remove model_type and training_backend to avoid conflicts
            saved_config.pop("model_type", None)
            saved_config.pop("training_backend", None)

            # Determine which directory to use for loading the model
            # Prefer hf_model subdirectory if it exists (contains model.safetensors)
            hf_model_dir = os.path.join(checkpoint_dir, "hf_model")
            model_load_path = (
                hf_model_dir if os.path.exists(hf_model_dir) else checkpoint_dir
            )

            saved_config["checkpoint_path"] = model_load_path

            config = TimesFMConfig(**saved_config)

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

            model = TimesFMForecaster(config)
            model._load_checkpoint(checkpoint_dir)
            model.is_fitted = True
        else:
            config = TimesFMConfig(
                context_length=kwargs.get("context_length", 512),
                forecast_length=kwargs.get("forecast_length", 96),
                batch_size=kwargs.get("batch_size", 32),
            )
            model = TimesFMForecaster(config)
        return model, config

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: sundial, ttm, chronos, moirai, timegrad, timesfm, tide"
        )
