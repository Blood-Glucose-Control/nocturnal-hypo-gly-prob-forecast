"""Config construction and default-parity tests for TimesFM 3.0.

These tests only touch the pure-Python config surface and therefore run in the
shared ``.noctprob-venv`` without the heavy ``timesfm3`` / ``peft`` runtime deps.
"""

from src.models.base import TrainingBackend
from src.models.timesfm3 import TimesFM3Config


def test_timesfm3_config_defaults() -> None:
    """Default construction yields the proven zero-shot + LoRA recipe values."""
    config = TimesFM3Config()

    # Identity / backend.
    assert config.model_type == "timesfm3"
    assert config.model_path == "google/timesfm-3.0-pytorch"
    assert config.training_backend == TrainingBackend.PYTORCH

    # Windowing / inference.
    assert config.context_length == 512
    assert config.forecast_length == 96
    assert config.target_col == "bg_mM"  # mmol/L, not mg/dL
    assert config.covariate_cols == ["iob"]
    assert config.make_positive is True
    assert config.quantiles == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # LoRA adapter hyperparameters.
    assert config.lora_r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert config.lora_target_modules == "all-linear"

    # Fine-tuning / early stopping.
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 0.01
    assert config.batch_size == 32
    assert config.early_stopping_patience == 3
    assert config.val_fraction == 0.2
    assert config.max_train_windows == 2048
    assert config.grad_clip_norm == 1.0
    assert config.covariate_jitter == 1e-3


def test_timesfm3_config_overrides() -> None:
    """TimesFM 3.0 specific and base fields can be overridden via kwargs."""
    config = TimesFM3Config(
        context_length=64,
        forecast_length=16,
        covariate_cols=[],
        make_positive=False,
        lora_r=4,
        lora_alpha=8,
        learning_rate=5e-5,
        num_epochs=2,
        batch_size=8,
        val_fraction=0.0,
        device="cpu",
    )

    assert config.context_length == 64
    assert config.forecast_length == 16
    assert config.covariate_cols == []
    assert config.make_positive is False
    assert config.lora_r == 4
    assert config.lora_alpha == 8
    assert config.learning_rate == 5e-5
    assert config.num_epochs == 2
    assert config.batch_size == 8
    assert config.val_fraction == 0.0
    assert config.device == "cpu"
    # Still a timesfm3 config regardless of overrides.
    assert config.model_type == "timesfm3"


def test_timesfm3_config_to_dict_roundtrip() -> None:
    """``to_dict`` captures custom fields and ``from_dict`` reconstructs them."""
    config = TimesFM3Config(
        context_length=64,
        forecast_length=16,
        covariate_cols=["iob", "cob"],
        quantiles=[0.25, 0.5, 0.75],
    )
    payload = config.to_dict()

    # Custom fields survive serialization.
    for key in (
        "target_col",
        "covariate_cols",
        "make_positive",
        "quantiles",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "val_fraction",
        "grad_clip_norm",
        "covariate_jitter",
    ):
        assert key in payload
    assert payload["model_type"] == "timesfm3"
    assert payload["covariate_cols"] == ["iob", "cob"]
    assert payload["quantiles"] == [0.25, 0.5, 0.75]

    restored = TimesFM3Config.from_dict(payload)
    assert restored.model_type == "timesfm3"
    assert restored.context_length == 64
    assert restored.forecast_length == 16
    assert restored.covariate_cols == ["iob", "cob"]
    assert restored.quantiles == [0.25, 0.5, 0.75]
    assert restored.training_backend == TrainingBackend.PYTORCH
