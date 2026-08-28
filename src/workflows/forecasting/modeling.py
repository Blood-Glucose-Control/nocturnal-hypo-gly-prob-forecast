"""Modeling helpers for forecasting workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from ...config.schemas import (
    build_model_runtime_config,
    get_model_config_schema,
    get_registered_model_config_types,
    load_yaml_as_schema,
)

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: dict[str, Any] = {}
ZERO_SHOT_INFERENCE_MODEL_TYPES = {
    "sundial",
    "ttm",
    "chronos2",
    "moment",
    "timesfm",
    "toto",
    "moirai",
}


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


def load_model_config_from_yaml(config_path: str, model_type: str) -> Dict[str, Any]:
    """Load a model config override dictionary from YAML."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    if not model_type or not model_type.strip():
        raise ValueError(
            "model_type is required for schema-validated model config loading"
        )
    schema_type = get_model_config_schema(model_type)
    if schema_type is None:
        registered_types = ", ".join(get_registered_model_config_types()) or "(none)"
        raise ValueError(
            f"No model config schema registered for model_type={model_type}. "
            f"Registered schema types: {registered_types}"
        )

    validated = load_yaml_as_schema(config_file, schema_type)
    model_dump = getattr(validated, "model_dump", None)
    if callable(model_dump):
        raw_config = model_dump(exclude_none=True)
    else:
        legacy_dump = getattr(validated, "dict", None)
        if not callable(legacy_dump):
            raise ValueError(
                f"Validated model config does not support dump API: {config_path}"
            )
        raw_config = legacy_dump(exclude_none=True)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Model config must be a YAML mapping/object: {config_path}")
    config = dict(raw_config)
    logger.info(
        "Validated model config with schema %s for model_type=%s",
        schema_type.__name__,
        model_type,
    )

    logger.info(f"Loaded model config from: {config_path}")
    logger.info(f"  Parameters specified: {len(config)}")
    for key, value in config.items():
        logger.info(f"    {key}: {value}")
    return config


def _set_config_attr_if_present(config: Any, field: str, value: Any) -> None:
    if hasattr(config, field):
        setattr(config, field, value)


def _apply_checkpoint_overrides(config: Any, overrides: Dict[str, Any]) -> None:
    """Apply inference-safe overrides to a loaded checkpoint config."""
    if "batch_size" in overrides:
        _set_config_attr_if_present(config, "batch_size", overrides["batch_size"])

    if "forecast_length" in overrides and hasattr(config, "forecast_length"):
        requested = int(overrides["forecast_length"])
        current = int(getattr(config, "forecast_length"))
        if requested <= current:
            logger.info("Overriding forecast_length: %s -> %s", current, requested)
            setattr(config, "forecast_length", requested)
        else:
            logger.warning(
                "Cannot increase forecast_length beyond trained value (%s). "
                "Using saved value.",
                current,
            )

    if "context_length" in overrides and hasattr(config, "context_length"):
        requested = int(overrides["context_length"])
        current = int(getattr(config, "context_length"))
        if requested != current:
            logger.warning(
                "context_length mismatch: requested %s, model trained with %s. "
                "Using saved value.",
                requested,
                current,
            )

    for attr_name, value in overrides.items():
        if attr_name in {
            "model_type",
            "batch_size",
            "forecast_length",
            "context_length",
        }:
            continue
        _set_config_attr_if_present(config, attr_name, value)


def _pop_typed_config_value(
    config_data: Dict[str, Any],
    keys: Sequence[str],
    default: Any,
) -> Any:
    for key in keys:
        if key in config_data:
            return config_data.pop(key)
    return default


def create_model_and_config(
    model_type: str,
    checkpoint: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Schema-routed model/config constructor for workflow entrypoints."""
    model_type_lower = model_type.lower()
    config_data = dict(kwargs)
    config_data.pop("model_type", None)

    context_length = int(_pop_typed_config_value(config_data, ["context_length"], 512))
    forecast_length = int(_pop_typed_config_value(config_data, ["forecast_length"], 96))
    batch_size = int(_pop_typed_config_value(config_data, ["batch_size"], 2048))
    num_epochs = int(_pop_typed_config_value(config_data, ["num_epochs"], 1))
    learning_rate = float(
        _pop_typed_config_value(config_data, ["learning_rate", "lr"], 1e-4)
    )
    model_path = _pop_typed_config_value(config_data, ["model_path"], None)
    use_cpu = bool(_pop_typed_config_value(config_data, ["use_cpu"], False))
    fp16 = bool(_pop_typed_config_value(config_data, ["fp16"], True))

    if checkpoint:
        generic_config = ModelFactory.create_finetune_config(
            model_type=model_type_lower,
            model_path=model_path,
            context_length=context_length,
            forecast_length=forecast_length,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_cpu=use_cpu,
            fp16=fp16,
            extra_config=config_data,
        )
        model = ModelFactory.load_model(model_type_lower, checkpoint, generic_config)
        config = model.config
        _apply_checkpoint_overrides(config, kwargs)
        return model, config

    if model_type_lower in ZERO_SHOT_INFERENCE_MODEL_TYPES:
        generic_config = ModelFactory.create_zero_shot_config(
            model_type=model_type_lower,
            model_path=model_path,
            context_length=context_length,
            forecast_length=forecast_length,
            batch_size=batch_size,
            use_cpu=use_cpu,
            fp16=fp16,
            extra_config=config_data,
        )
    else:
        generic_config = ModelFactory.create_finetune_config(
            model_type=model_type_lower,
            model_path=model_path,
            context_length=context_length,
            forecast_length=forecast_length,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_cpu=use_cpu,
            fp16=fp16,
            extra_config=config_data,
        )

    model = ModelFactory.create_model(generic_config)
    return model, model.config


class ModelFactory:
    """Factory for creating workflow model instances by model type."""

    @staticmethod
    def get_default_model_path(model_type: str) -> str:
        defaults = {
            "sundial": "thuml/sundial-base-128m",
            "ttm": "ibm-granite/granite-timeseries-ttm-r2",
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
        if model_type == "sundial":
            from ...models.sundial import SundialConfig, SundialForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "sundial",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    **extra,
                },
            )
            return SundialForecaster(SundialConfig(**runtime_config))
        if model_type == "ttm":
            from ...models.ttm import TTMConfig, TTMForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "ttm",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TTMForecaster(TTMConfig(**runtime_config))
        if model_type == "chronos2":
            try:
                from ...models.chronos2 import Chronos2Config, Chronos2Forecaster
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
                from ...models.moment import MomentConfig, MomentForecaster
            except ImportError as e:
                raise ImportError(
                    f"MOMENT model not available. Install moment dependencies: {e}"
                ) from e
            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "moment",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return MomentForecaster(MomentConfig(**runtime_config))
        if model_type == "timesfm":
            try:
                from ...models.timesfm import TimesFMConfig, TimesFMForecaster
            except ImportError as e:
                raise ImportError(
                    f"TimesFM model not available. Install with: pip install transformers>=5.2.0: {e}"
                ) from e
            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "timesfm",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TimesFMForecaster(TimesFMConfig(**runtime_config))
        if model_type == "timegrad":
            try:
                from ...models.timegrad import TimeGradConfig, TimeGradForecaster
            except ImportError as e:
                raise ImportError(
                    "TimeGrad model not available. Install with: "
                    f"source scripts/setup_model_env.sh timegrad\n{e}"
                ) from e
            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "timegrad",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TimeGradForecaster(TimeGradConfig(**runtime_config))
        if model_type == "tide":
            try:
                from ...models.tide import TiDEConfig, TiDEForecaster
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
                from ...models.toto import TotoConfig, TotoForecaster
            except ImportError as e:
                raise ImportError(
                    f"Toto model not available. Install with: source scripts/setup_model_env.sh toto\n{e}"
                ) from e
            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "toto",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TotoForecaster(TotoConfig(**runtime_config))
        if model_type == "moirai":
            try:
                from ...models.moirai import MoiraiConfig, MoiraiForecaster
            except ImportError as e:
                raise ImportError(
                    "Moirai model not available. Install with: "
                    f"pip install 'nocturnal-hypo-gly-prob-forecast[moirai]'\n{e}"
                ) from e
            moirai_kwargs = dict(config.extra_config) if config.extra_config else {}
            if "lr" in moirai_kwargs and "learning_rate" not in moirai_kwargs:
                moirai_kwargs["learning_rate"] = moirai_kwargs.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type,
                config_data={
                    "model_type": "moirai",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **moirai_kwargs,
                },
            )
            return MoiraiForecaster(MoiraiConfig(**runtime_config))
        if model_type == "naive_baseline":
            from ...models.naive_baseline import (
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
            from ...models.statistical import StatisticalConfig, StatisticalForecaster

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
            from ...models.deepar import DeepARConfig, DeepARForecaster

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
            from ...models.patchtst import PatchTSTConfig, PatchTSTForecaster

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
            from ...models.tft import TFTConfig, TFTForecaster

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
                from ...models.tsmixer import TSMixerConfig, TSMixerForecaster
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
            "Supported types: sundial, ttm, chronos2, moment, timesfm, timegrad, tide, toto, moirai, "
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
        if model_type_lower == "sundial":
            from ...models.sundial import SundialConfig, SundialForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type_lower,
                config_data={
                    "model_type": "sundial",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    **extra,
                },
            )
            return SundialForecaster.load(
                model_path,
                SundialConfig(**runtime_config),
            )
        if model_type_lower == "ttm":
            from ...models.ttm import TTMConfig, TTMForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type_lower,
                config_data={
                    "model_type": "ttm",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TTMForecaster.load(
                model_path,
                TTMConfig(**runtime_config),
            )
        if model_type_lower == "chronos2":
            from ...models.chronos2 import Chronos2Forecaster

            return Chronos2Forecaster.load(model_path)
        if model_type_lower == "moment":
            from ...models.moment import MomentConfig, MomentForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type_lower,
                config_data={
                    "model_type": "moment",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return MomentForecaster.load(
                model_path,
                MomentConfig(**runtime_config),
            )
        if model_type_lower == "timesfm":
            from ...models.timesfm import TimesFMConfig, TimesFMForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            runtime_config = build_model_runtime_config(
                model_type=model_type_lower,
                config_data={
                    "model_type": "timesfm",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TimesFMForecaster.load(
                model_path,
                TimesFMConfig(**runtime_config),
            )
        if model_type_lower == "timegrad":
            from ...models.timegrad import TimeGradConfig, TimeGradForecaster

            extra = dict(config.extra_config) if config.extra_config else {}
            if "lr" in extra and "learning_rate" not in extra:
                extra["learning_rate"] = extra.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type_lower,
                config_data={
                    "model_type": "timegrad",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **extra,
                },
            )
            return TimeGradForecaster.load(
                model_path,
                TimeGradConfig(**runtime_config),
            )
        if model_type_lower == "tide":
            from ...models.tide import TiDEForecaster

            return TiDEForecaster.load(model_path)
        if model_type_lower == "toto":
            from ...models.toto import TotoForecaster

            return TotoForecaster.load(model_path)
        if model_type_lower == "moirai":
            from ...models.moirai import MoiraiConfig, MoiraiForecaster

            moirai_kwargs = dict(config.extra_config) if config.extra_config else {}
            if "lr" in moirai_kwargs and "learning_rate" not in moirai_kwargs:
                moirai_kwargs["learning_rate"] = moirai_kwargs.pop("lr")
            runtime_config = build_model_runtime_config(
                model_type=model_type_lower,
                config_data={
                    "model_type": "moirai",
                    "model_path": config.model_path,
                    "context_length": config.context_length,
                    "forecast_length": config.forecast_length,
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "training_mode": config.training_mode,
                    "freeze_backbone": config.freeze_backbone,
                    "use_cpu": config.use_cpu,
                    "fp16": config.fp16,
                    "learning_rate": config.learning_rate,
                    **moirai_kwargs,
                },
            )
            return MoiraiForecaster.load(
                model_path,
                MoiraiConfig(**runtime_config),
            )
        if model_type_lower == "naive_baseline":
            from ...models.naive_baseline import NaiveBaselineForecaster

            return NaiveBaselineForecaster.load(model_path)
        if model_type_lower == "statistical":
            from ...models.statistical import StatisticalForecaster

            return StatisticalForecaster.load(model_path)
        if model_type_lower == "deepar":
            from ...models.deepar import DeepARForecaster

            return DeepARForecaster.load(model_path)
        if model_type_lower == "patchtst":
            from ...models.patchtst import PatchTSTForecaster

            return PatchTSTForecaster.load(model_path)
        if model_type_lower == "tft":
            from ...models.tft import TFTForecaster

            return TFTForecaster.load(model_path)
        if model_type_lower == "tsmixer":
            from ...models.tsmixer import TSMixerForecaster

            return TSMixerForecaster.load(model_path)
        raise ValueError(
            f"Unsupported model type for loading: {model_type}. "
            "Supported types: sundial, ttm, chronos2, moment, timesfm, timegrad, tide, toto, moirai, "
            "naive_baseline, statistical, deepar, patchtst, tft, tsmixer"
        )
