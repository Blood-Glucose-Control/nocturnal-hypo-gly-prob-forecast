"""Modeling helpers for forecasting workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.schemas import (
    build_model_runtime_config,
    get_model_config_schema,
    load_yaml_as_schema,
)
from src.utils.config_loader import load_yaml_config

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: dict[str, Any] = {}


def register_model(model_type: str):
    """Decorator to register a model type in the factory registry."""

    def decorator(cls):
        SUPPORTED_MODELS[model_type] = cls
        return cls

    return decorator


@dataclass
class GenericModelConfig:
    """Model-agnostic workflow config wrapper."""

    model_type: str
    model_path: str
    context_length: int = 512
    forecast_length: int = 96
    batch_size: int = 2048
    num_epochs: int = 1
    training_mode: str = "fine_tune"
    freeze_backbone: bool = False
    use_cpu: bool = False
    fp16: bool = True
    learning_rate: float = 1e-4
    extra_config: Dict[str, Any] = field(default_factory=dict)


def load_model_config_from_yaml(
    config_path: str, model_type: Optional[str] = None
) -> Dict[str, Any]:
    """Load a model config override dictionary from YAML."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    schema_type = get_model_config_schema(model_type) if model_type else None
    if schema_type is not None:
        validated = load_yaml_as_schema(config_file, schema_type)
        config = validated.model_dump(exclude_none=True)
        logger.info(
            "Validated model config with schema %s for model_type=%s",
            schema_type.__name__,
            model_type,
        )
    else:
        config = load_yaml_config(config_path)
        if config is None:
            logger.warning(f"Model config file is empty: {config_path}")
            return {}
        if not isinstance(config, dict):
            raise ValueError(
                f"Model config must be a YAML mapping/object: {config_path}"
            )

    logger.info(f"Loaded model config from: {config_path}")
    logger.info(f"  Parameters specified: {len(config)}")
    for key, value in config.items():
        logger.info(f"    {key}: {value}")
    return config


class ModelFactory:
    """Factory for creating workflow model instances by model type."""

    @staticmethod
    def get_default_model_path(model_type: str) -> str:
        defaults = {
            "ttm": "ibm-granite/granite-timeseries-ttm-r2",
            "chronos": "amazon/chronos-t5-small",
            "chronos2": "autogluon/chronos-2",
            "moment": "AutonLab/MOMENT-1-small",
            "timesfm": "google/timesfm-2.0-500m-pytorch",
            "timegrad": "",
            "tide": "",
            "toto": "Datadog/Toto-Open-Base-1.0",
            "moirai": "Salesforce/moirai-1.0-R-small",
            "naive_baseline": "",
            "statistical": "",
            "deepar": "",
            "patchtst": "",
            "tft": "",
            "tsmixer": "",
        }
        return defaults.get(model_type, "")

    @staticmethod
    def create_model(config: GenericModelConfig):
        model_type = config.model_type.lower()
        if model_type == "ttm":
            from src.models.ttm import TTMConfig, TTMForecaster

            return TTMForecaster(
                TTMConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    freeze_backbone=config.freeze_backbone,
                    use_cpu=config.use_cpu,
                    fp16=config.fp16,
                    learning_rate=config.learning_rate,
                    **config.extra_config,
                )
            )
        if model_type == "chronos":
            try:
                from src.models.chronos import ChronosConfig, ChronosForecaster
            except ImportError as e:
                raise ImportError(
                    f"Chronos model not available. Install chronos dependencies: {e}"
                ) from e
            return ChronosForecaster(
                ChronosConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    use_cpu=config.use_cpu,
                    fp16=config.fp16,
                    **config.extra_config,
                )
            )
        if model_type == "chronos2":
            try:
                from src.models.chronos2 import Chronos2Config, Chronos2Forecaster
            except ImportError as e:
                raise ImportError(
                    "Chronos-2 model not available. Install with: "
                    f"pip install 'nocturnal-hypo-gly-prob-forecast[chronos2]': {e}"
                ) from e
            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "chronos2",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return Chronos2Forecaster(Chronos2Config(**runtime_config))
        if model_type == "moment":
            try:
                from src.models.moment import MomentConfig, MomentForecaster
            except ImportError as e:
                raise ImportError(
                    f"MOMENT model not available. Install moment dependencies: {e}"
                ) from e
            return MomentForecaster(
                MomentConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    freeze_backbone=config.freeze_backbone,
                    use_cpu=config.use_cpu,
                    fp16=config.fp16,
                    learning_rate=config.learning_rate,
                    **config.extra_config,
                )
            )
        if model_type == "timesfm":
            try:
                from src.models.timesfm import TimesFMConfig, TimesFMForecaster
            except ImportError as e:
                raise ImportError(
                    f"TimesFM model not available. Install with: pip install transformers>=5.2.0: {e}"
                ) from e
            extra = dict(config.extra_config)
            checkpoint_path = extra.pop("checkpoint_path", None) or config.model_path
            return TimesFMForecaster(
                TimesFMConfig(
                    checkpoint_path=checkpoint_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    horizon_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    use_cpu=config.use_cpu,
                    learning_rate=config.learning_rate,
                    **extra,
                )
            )
        if model_type == "timegrad":
            try:
                from src.models.timegrad import TimeGradConfig, TimeGradForecaster
            except ImportError as e:
                raise ImportError(
                    "TimeGrad model not available. Install with: "
                    f"source scripts/setup_model_env.sh timegrad\n{e}"
                ) from e
            return TimeGradForecaster(
                TimeGradConfig(
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    use_cpu=config.use_cpu,
                    learning_rate=config.learning_rate,
                    **config.extra_config,
                )
            )
        if model_type == "tide":
            try:
                from src.models.tide import TiDEConfig, TiDEForecaster
            except ImportError as e:
                raise ImportError(
                    "TiDE model not available. Install with: "
                    f"pip install 'nocturnal-hypo-gly-prob-forecast[tide]': {e}"
                ) from e
            extra = dict(config.extra_config) if config.extra_config else {}
            has_training_mode_override = "training_mode" in extra
            resolved_training_mode = extra.pop("training_mode", config.training_mode)
            if not has_training_mode_override and resolved_training_mode == "fine_tune":
                # GenericModelConfig defaults to fine_tune, but TiDE only supports
                # from-scratch training.
                resolved_training_mode = "from_scratch"
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "tide",
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "training_mode": resolved_training_mode,
                    **extra,
                },
            )
            return TiDEForecaster(TiDEConfig(**runtime_config))
        if model_type == "toto":
            try:
                from src.models.toto import TotoConfig, TotoForecaster
            except ImportError as e:
                raise ImportError(
                    f"Toto model not available. Install with: source scripts/setup_model_env.sh toto\n{e}"
                ) from e
            toto_kwargs = dict(config.extra_config) if config.extra_config else {}
            toto_kwargs.setdefault("context_length", config.context_length)
            toto_kwargs.setdefault("forecast_length", config.forecast_length)
            toto_kwargs.setdefault("batch_size", config.batch_size)
            if config.num_epochs is not None:
                toto_kwargs.setdefault("num_epochs", config.num_epochs)
            if config.learning_rate is not None and "lr" not in toto_kwargs:
                toto_kwargs["lr"] = config.learning_rate
            toto_kwargs.setdefault("use_cpu", config.use_cpu)
            return TotoForecaster(TotoConfig(**toto_kwargs))
        if model_type == "moirai":
            try:
                from src.models.moirai import MoiraiConfig, MoiraiForecaster
            except ImportError as e:
                raise ImportError(
                    "Moirai model not available. Install with: "
                    f"pip install 'nocturnal-hypo-gly-prob-forecast[moirai]'\n{e}"
                ) from e
            moirai_kwargs = dict(config.extra_config) if config.extra_config else {}
            return MoiraiForecaster(
                MoiraiConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    learning_rate=config.learning_rate,
                    **moirai_kwargs,
                )
            )
        if model_type == "naive_baseline":
            from src.models.naive_baseline import (
                NaiveBaselineConfig,
                NaiveBaselineForecaster,
            )

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "naive_baseline",
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "model_name": extra.pop("model_name", "Naive"),
                    "covariate_cols": extra.pop("covariate_cols", []),
                    **extra,
                },
            )
            return NaiveBaselineForecaster(NaiveBaselineConfig(**runtime_config))
        if model_type == "statistical":
            from src.models.statistical import StatisticalConfig, StatisticalForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "statistical",
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "model_name": extra.pop("model_name", "AutoARIMA"),
                    "covariate_cols": extra.pop("covariate_cols", []),
                    **extra,
                },
            )
            return StatisticalForecaster(StatisticalConfig(**runtime_config))
        if model_type == "deepar":
            from src.models.deepar import DeepARConfig, DeepARForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "deepar",
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "covariate_cols": extra.pop("covariate_cols", []),
                    **extra,
                },
            )
            return DeepARForecaster(DeepARConfig(**runtime_config))
        if model_type == "patchtst":
            from src.models.patchtst import PatchTSTConfig, PatchTSTForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "patchtst",
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "covariate_cols": extra.pop("covariate_cols", []),
                    **extra,
                },
            )
            return PatchTSTForecaster(PatchTSTConfig(**runtime_config))
        if model_type == "tft":
            from src.models.tft import TFTConfig, TFTForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "tft",
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "covariate_cols": extra.pop("covariate_cols", []),
                    **extra,
                },
            )
            return TFTForecaster(TFTConfig(**runtime_config))
        if model_type == "tsmixer":
            try:
                from src.models.tsmixer import TSMixerConfig, TSMixerForecaster
            except ImportError as e:
                raise ImportError(
                    "TSMixer model not available. Install with: "
                    "source scripts/setup_model_env.sh tsmixer\n"
                    f"{e}"
                ) from e

            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "tsmixer",
                    "model_path": config.model_path,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TSMixerForecaster(TSMixerConfig(**runtime_config))
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Supported types: ttm, chronos, chronos2, moment, timesfm, timegrad, tide, toto, moirai, "
            "naive_baseline, statistical, deepar, patchtst, tft, tsmixer"
        )

    @staticmethod
    def create_zero_shot_config(
        model_type: str,
        model_path: Optional[str] = None,
        context_length: int = 512,
        forecast_length: int = 96,
        batch_size: int = 2048,
        use_cpu: bool = False,
        fp16: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> GenericModelConfig:
        yaml_config = dict(extra_config) if extra_config else {}
        resolved_model_path = model_path or yaml_config.pop(
            "model_path", ModelFactory.get_default_model_path(model_type)
        )
        resolved_context = yaml_config.pop("context_length", context_length)
        resolved_forecast = yaml_config.pop("forecast_length", forecast_length)
        resolved_batch = yaml_config.pop("batch_size", batch_size)
        for key in [
            "num_epochs",
            "training_mode",
            "freeze_backbone",
            "use_cpu",
            "fp16",
            "learning_rate",
        ]:
            yaml_config.pop(key, None)

        return GenericModelConfig(
            model_type=model_type,
            model_path=resolved_model_path,
            context_length=resolved_context,
            forecast_length=resolved_forecast,
            batch_size=resolved_batch,
            num_epochs=0,
            training_mode="zero_shot",
            freeze_backbone=True,
            use_cpu=use_cpu,
            fp16=fp16,
            extra_config=yaml_config,
        )

    @staticmethod
    def create_finetune_config(
        model_type: str,
        model_path: Optional[str] = None,
        context_length: Optional[int] = None,
        forecast_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        use_cpu: bool = False,
        fp16: bool = True,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> GenericModelConfig:
        yaml_config = dict(extra_config) if extra_config else {}
        yaml_model_path = yaml_config.pop("model_path", None)
        yaml_context = yaml_config.pop("context_length", None)
        yaml_forecast = yaml_config.pop("forecast_length", None)
        yaml_batch = yaml_config.pop("batch_size", None)
        yaml_epochs = yaml_config.pop("num_epochs", None)
        yaml_lr = yaml_config.pop("learning_rate", None)
        yaml_mode = yaml_config.pop("training_mode", None)
        yaml_freeze = yaml_config.pop("freeze_backbone", False)

        resolved_model_path = (
            model_path
            or yaml_model_path
            or ModelFactory.get_default_model_path(model_type)
        )
        resolved_context = (
            context_length
            if context_length is not None
            else (yaml_context if yaml_context is not None else 512)
        )
        resolved_forecast = (
            forecast_length
            if forecast_length is not None
            else (yaml_forecast if yaml_forecast is not None else 96)
        )
        resolved_batch = (
            batch_size
            if batch_size is not None
            else (yaml_batch if yaml_batch is not None else 2048)
        )
        resolved_epochs = (
            num_epochs
            if num_epochs is not None
            else (yaml_epochs if yaml_epochs is not None else 1)
        )
        resolved_lr = (
            learning_rate
            if learning_rate is not None
            else (yaml_lr if yaml_lr is not None else 1e-4)
        )

        yaml_config.pop("use_cpu", None)
        yaml_config.pop("fp16", None)

        default_training_mode = (
            "from_scratch" if model_type.lower() == "tide" else "fine_tune"
        )
        resolved_training_mode = (
            yaml_mode if yaml_mode is not None else default_training_mode
        )

        return GenericModelConfig(
            model_type=model_type,
            model_path=resolved_model_path,
            context_length=resolved_context,
            forecast_length=resolved_forecast,
            batch_size=resolved_batch,
            num_epochs=resolved_epochs,
            training_mode=resolved_training_mode,
            freeze_backbone=yaml_freeze,
            use_cpu=use_cpu,
            fp16=fp16,
            learning_rate=resolved_lr,
            extra_config=yaml_config,
        )

    @staticmethod
    def load_model(
        model_type: str,
        model_path: str,
        config: GenericModelConfig,
    ):
        model_type_lower = model_type.lower()
        if model_type_lower == "ttm":
            from src.models.ttm import TTMConfig, TTMForecaster

            return TTMForecaster.load(
                model_path,
                TTMConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    freeze_backbone=config.freeze_backbone,
                    use_cpu=config.use_cpu,
                    fp16=config.fp16,
                    learning_rate=config.learning_rate,
                    **config.extra_config,
                ),
            )
        if model_type_lower == "chronos":
            from src.models.chronos import ChronosConfig, ChronosForecaster

            return ChronosForecaster.load(
                model_path,
                ChronosConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    use_cpu=config.use_cpu,
                    fp16=config.fp16,
                    **config.extra_config,
                ),
            )
        if model_type_lower == "chronos2":
            from src.models.chronos2 import Chronos2Forecaster

            return Chronos2Forecaster.load(model_path)
        if model_type_lower == "moment":
            from src.models.moment import MomentConfig, MomentForecaster

            return MomentForecaster.load(
                model_path,
                MomentConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    use_cpu=config.use_cpu,
                    fp16=config.fp16,
                    **config.extra_config,
                ),
            )
        if model_type_lower == "timesfm":
            from src.models.timesfm import TimesFMConfig, TimesFMForecaster

            extra = dict(config.extra_config)
            checkpoint_path = extra.pop("checkpoint_path", None) or config.model_path
            return TimesFMForecaster.load(
                model_path,
                TimesFMConfig(
                    checkpoint_path=checkpoint_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    horizon_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    use_cpu=config.use_cpu,
                    learning_rate=config.learning_rate,
                    **extra,
                ),
            )
        if model_type_lower == "timegrad":
            from src.models.timegrad import TimeGradConfig, TimeGradForecaster

            return TimeGradForecaster.load(
                model_path,
                TimeGradConfig(
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    training_mode=config.training_mode,
                    use_cpu=config.use_cpu,
                    learning_rate=config.learning_rate,
                    **config.extra_config,
                ),
            )
        if model_type_lower == "tide":
            from src.models.tide import TiDEForecaster

            return TiDEForecaster.load(model_path)
        if model_type_lower == "toto":
            from src.models.toto import TotoForecaster

            return TotoForecaster.load(model_path)
        if model_type_lower == "moirai":
            from src.models.moirai import MoiraiConfig, MoiraiForecaster

            moirai_kwargs = dict(config.extra_config) if config.extra_config else {}
            return MoiraiForecaster.load(
                model_path,
                MoiraiConfig(
                    model_path=config.model_path,
                    context_length=config.context_length,
                    forecast_length=config.forecast_length,
                    batch_size=config.batch_size,
                    num_epochs=config.num_epochs,
                    learning_rate=config.learning_rate,
                    **moirai_kwargs,
                ),
            )
        if model_type_lower == "naive_baseline":
            from src.models.naive_baseline import NaiveBaselineForecaster

            return NaiveBaselineForecaster.load(model_path)
        if model_type_lower == "statistical":
            from src.models.statistical import StatisticalForecaster

            return StatisticalForecaster.load(model_path)
        if model_type_lower == "deepar":
            from src.models.deepar import DeepARForecaster

            return DeepARForecaster.load(model_path)
        if model_type_lower == "patchtst":
            from src.models.patchtst import PatchTSTForecaster

            return PatchTSTForecaster.load(model_path)
        if model_type_lower == "tft":
            from src.models.tft import TFTForecaster

            return TFTForecaster.load(model_path)
        if model_type_lower == "tsmixer":
            from src.models.tsmixer import TSMixerForecaster

            return TSMixerForecaster.load(model_path)
        raise ValueError(
            f"Unsupported model type for loading: {model_type}. "
            "Supported types: ttm, chronos, chronos2, moment, timesfm, timegrad, tide, toto, moirai, "
            "naive_baseline, statistical, deepar, patchtst, tft, tsmixer"
        )
