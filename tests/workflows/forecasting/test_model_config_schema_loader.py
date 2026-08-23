"""Tests for schema-validated model config loading."""

import sys
import types
from pathlib import Path

import pytest

from src.config.schemas import (
    build_model_runtime_config,
    get_model_config_schema,
    get_registered_model_config_types,
)
from src.workflows.forecasting.modeling import (
    GenericModelConfig,
    ModelFactory,
    load_model_config_from_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TSMIXER_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "models" / "tsmixer" / "00_iob_cob_smoke.yaml"
)
TIDE_SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "tide" / "00_bg_only.yaml"
TTM_SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "ttm" / "00_zs_cgm_only.yaml"
SUNDIAL_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "models" / "sundial" / "00_baseline.yaml"
)
MOIRAI_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "models" / "moirai" / "bg_only_smoke_test.yaml"
)
MOMENT_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "models" / "moment" / "00_baseline.yaml"
)
TIMEGRAD_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "models" / "timegrad" / "cgm_only.yaml"
)
TOTO_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "models" / "toto" / "bg_only_smoke_test.yaml"
)
TIMESFM_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs" / "models" / "timesfm" / "00_long_run.yaml"
)
AUTOGLUON_SCHEMA_SMOKE_CONFIGS = {
    "chronos2": REPO_ROOT / "configs" / "models" / "chronos2" / "00_bg_only.yaml",
    "deepar": REPO_ROOT / "configs" / "models" / "deepar" / "00_baseline.yaml",
    "naive_baseline": REPO_ROOT
    / "configs"
    / "models"
    / "naive_baseline"
    / "00_naive.yaml",
    "patchtst": REPO_ROOT / "configs" / "models" / "patchtst" / "00_baseline.yaml",
    "statistical": REPO_ROOT
    / "configs"
    / "models"
    / "statistical"
    / "00_autoarima_bg_only.yaml",
    "tft": REPO_ROOT / "configs" / "models" / "tft" / "00_bg_baseline.yaml",
    "tide": TIDE_SMOKE_CONFIG_PATH,
}


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def _install_fake_tsmixer_module(monkeypatch: pytest.MonkeyPatch) -> type:
    fake_models_pkg = types.ModuleType("src.models")
    fake_models_pkg.__path__ = []  # type: ignore[attr-defined]

    class FakeTSMixerConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            if getattr(self, "min_segment_length", None) is None:
                self.min_segment_length = self.context_length + self.forecast_length

    class FakeTSMixerForecaster:
        def __init__(self, config):
            self.config = config

    fake_tsmixer_module = types.ModuleType("src.models.tsmixer")
    fake_tsmixer_module.TSMixerConfig = FakeTSMixerConfig
    fake_tsmixer_module.TSMixerForecaster = FakeTSMixerForecaster

    monkeypatch.setitem(sys.modules, "src.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "src.models.tsmixer", fake_tsmixer_module)
    return FakeTSMixerConfig


def _install_fake_deepar_module(monkeypatch: pytest.MonkeyPatch) -> type:
    fake_models_pkg = types.ModuleType("src.models")
    fake_models_pkg.__path__ = []  # type: ignore[attr-defined]

    class FakeDeepARConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeDeepARForecaster:
        def __init__(self, config):
            self.config = config

    fake_deepar_module = types.ModuleType("src.models.deepar")
    fake_deepar_module.DeepARConfig = FakeDeepARConfig
    fake_deepar_module.DeepARForecaster = FakeDeepARForecaster

    monkeypatch.setitem(sys.modules, "src.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "src.models.deepar", fake_deepar_module)
    return FakeDeepARConfig


def _install_fake_chronos2_module(monkeypatch: pytest.MonkeyPatch) -> type:
    fake_models_pkg = types.ModuleType("src.models")
    fake_models_pkg.__path__ = []  # type: ignore[attr-defined]

    class FakeChronos2Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeChronos2Forecaster:
        def __init__(self, config):
            self.config = config

    fake_chronos2_module = types.ModuleType("src.models.chronos2")
    fake_chronos2_module.Chronos2Config = FakeChronos2Config
    fake_chronos2_module.Chronos2Forecaster = FakeChronos2Forecaster

    monkeypatch.setitem(sys.modules, "src.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, "src.models.chronos2", fake_chronos2_module)
    return FakeChronos2Config


def _install_fake_model_family_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    config_class_name: str,
    forecaster_class_name: str,
) -> type:
    fake_models_pkg = types.ModuleType("src.models")
    fake_models_pkg.__path__ = []  # type: ignore[attr-defined]

    class FakeConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeForecaster:
        def __init__(self, config):
            self.config = config

    fake_module = types.ModuleType(f"src.models.{module_name}")
    setattr(fake_module, config_class_name, FakeConfig)
    setattr(fake_module, forecaster_class_name, FakeForecaster)

    monkeypatch.setitem(sys.modules, "src.models", fake_models_pkg)
    monkeypatch.setitem(sys.modules, f"src.models.{module_name}", fake_module)
    return FakeConfig


def test_tsmixer_model_config_validates_and_loads(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "tsmixer_valid.yaml",
        """
model_type: tsmixer
training_mode: from_scratch
context_length: 128
forecast_length: 96
hidden_size: 32
ff_size: 32
num_blocks: 1
dropout: 0.1
learning_rate: 0.001
covariate_cols: [iob, cob]
target_col: bg_mM
patient_col: p_num
time_col: datetime
interval_mins: 5
imputation_threshold_mins: 45
""".strip(),
    )

    loaded = load_model_config_from_yaml(config_path, model_type="tsmixer")
    assert loaded["model_type"] == "tsmixer"
    assert loaded["context_length"] == 128
    assert loaded["covariate_cols"] == ["iob", "cob"]


@pytest.mark.parametrize(
    ("model_type", "config_path"),
    tuple(AUTOGLUON_SCHEMA_SMOKE_CONFIGS.items()),
)
def test_autogluon_model_configs_validate_via_schema_loader(
    model_type: str, config_path: Path
) -> None:
    loaded = load_model_config_from_yaml(str(config_path), model_type=model_type)

    assert loaded["model_type"] == model_type
    assert loaded["context_length"] > 0
    assert loaded["forecast_length"] > 0


def test_tsmixer_model_config_normalizes_lr_alias(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "tsmixer_lr_alias.yaml",
        """
model_type: tsmixer
context_length: 128
forecast_length: 96
lr: 0.004
""".strip(),
    )

    loaded = load_model_config_from_yaml(config_path, model_type="tsmixer")
    assert loaded["learning_rate"] == pytest.approx(0.004)
    assert "lr" not in loaded


def test_ttm_model_config_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(TTM_SMOKE_CONFIG_PATH),
        model_type="ttm",
    )

    assert loaded["model_type"] == "ttm"
    assert loaded["training_mode"] == "zero_shot"
    assert loaded["target_features"] == ["bg_mM"]


def test_sundial_model_config_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(SUNDIAL_SMOKE_CONFIG_PATH),
        model_type="sundial",
    )

    assert loaded["model_type"] == "sundial"
    assert loaded["num_samples"] == 50
    assert loaded["training_mode"] == "zero_shot"


def test_moirai_model_config_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(MOIRAI_SMOKE_CONFIG_PATH),
        model_type="moirai",
    )

    assert loaded["model_type"] == "moirai"
    assert loaded["patch_size"] == "auto"
    assert loaded["num_samples"] == 50
    assert loaded["past_covariate_dim"] == 0


def test_moment_model_config_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(MOMENT_SMOKE_CONFIG_PATH),
        model_type="moment",
    )

    assert loaded["model_type"] == "moment"
    assert loaded["model_path"] == "AutonLab/MOMENT-1-small"
    assert loaded["use_wrapper_normalization"] is False
    assert loaded["covariate_cols"] == []


def test_timegrad_model_config_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(TIMEGRAD_SMOKE_CONFIG_PATH),
        model_type="timegrad",
    )

    assert loaded["model_type"] == "timegrad"
    assert loaded["training_mode"] == "from_scratch"
    assert loaded["target_features"] == ["bg_mM"]
    assert loaded["split_config"] == {"train": 0.9, "val": 0.05, "test": 0.05}


def test_toto_model_config_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(TOTO_SMOKE_CONFIG_PATH),
        model_type="toto",
    )

    assert loaded["model_type"] == "toto"
    assert loaded["training_mode"] == "fine_tune"
    assert loaded["lr"] == pytest.approx(1.0e-4)
    assert loaded["max_steps"] == 100


def test_timesfm_model_config_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(TIMESFM_SMOKE_CONFIG_PATH),
        model_type="timesfm",
    )

    assert loaded["model_type"] == "timesfm"
    assert loaded["training_mode"] == "fine_tune"
    assert loaded["forecast_length"] == 96
    assert loaded["torch_dtype"] == "bfloat16"


@pytest.mark.parametrize("model_type", ["deepar", "patchtst", "tft", "tide"])
def test_autogluon_model_config_normalizes_learning_rate_alias(model_type: str) -> None:
    runtime_config = build_model_runtime_config(
        model_type=model_type,
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "learning_rate": 0.003,
        },
    )

    assert runtime_config["lr"] == pytest.approx(0.003)
    assert "learning_rate" not in runtime_config


def test_ttm_runtime_adapter_normalizes_learning_rate_alias() -> None:
    runtime_config = build_model_runtime_config(
        model_type="ttm",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "lr": 0.0025,
        },
    )

    assert runtime_config["learning_rate"] == pytest.approx(0.0025)
    assert "lr" not in runtime_config


def test_ttm_runtime_adapter_reports_unknown_field() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="ttm",
            config_data={"context_length": 128, "forecast_length": 96, "unknown": 1},
        )

    assert "unknown" in str(exc_info.value)


def test_sundial_runtime_adapter_enforces_zero_shot_mode() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="sundial",
            config_data={
                "context_length": 128,
                "forecast_length": 96,
                "training_mode": "fine_tune",
            },
        )

    assert "training_mode" in str(exc_info.value)
    assert "zero_shot" in str(exc_info.value)


def test_moirai_runtime_adapter_normalizes_learning_rate_alias() -> None:
    runtime_config = build_model_runtime_config(
        model_type="moirai",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "lr": 0.0007,
            "past_covariate_dim": 0,
            "covariate_cols": [],
        },
    )

    assert runtime_config["learning_rate"] == pytest.approx(0.0007)
    assert "lr" not in runtime_config


def test_moment_runtime_adapter_normalizes_learning_rate_alias() -> None:
    runtime_config = build_model_runtime_config(
        model_type="moment",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "lr": 0.0009,
        },
    )

    assert runtime_config["learning_rate"] == pytest.approx(0.0009)
    assert "lr" not in runtime_config


def test_timegrad_runtime_adapter_normalizes_learning_rate_alias() -> None:
    runtime_config = build_model_runtime_config(
        model_type="timegrad",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "lr": 0.0008,
        },
    )

    assert runtime_config["learning_rate"] == pytest.approx(0.0008)
    assert "lr" not in runtime_config


def test_toto_runtime_adapter_maps_learning_rate_alias_to_lr() -> None:
    runtime_config = build_model_runtime_config(
        model_type="toto",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "learning_rate": 0.0006,
        },
    )

    assert runtime_config["lr"] == pytest.approx(0.0006)
    assert "learning_rate" not in runtime_config


def test_timesfm_runtime_adapter_normalizes_checkpoint_and_horizon() -> None:
    runtime_config = build_model_runtime_config(
        model_type="timesfm",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "model_path": "google/timesfm-2.0-500m-pytorch",
            "learning_rate": 0.0002,
        },
    )

    assert runtime_config["checkpoint_path"] == "google/timesfm-2.0-500m-pytorch"
    assert runtime_config["model_path"] == "google/timesfm-2.0-500m-pytorch"
    assert runtime_config["horizon_length"] == 96
    assert runtime_config["learning_rate"] == pytest.approx(0.0002)


def test_timesfm_runtime_adapter_reports_unknown_field() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="timesfm",
            config_data={"context_length": 128, "forecast_length": 96, "unknown": 1},
        )

    assert "unknown" in str(exc_info.value)


def test_moirai_runtime_adapter_enforces_covariate_dim_parity() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="moirai",
            config_data={
                "context_length": 128,
                "forecast_length": 96,
                "past_covariate_dim": 1,
                "covariate_cols": [],
            },
        )

    assert "past_covariate_dim" in str(exc_info.value)
    assert "covariate_cols" in str(exc_info.value)


def test_tide_runtime_adapter_enforces_encoder_decoder_dim_parity() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="tide",
            config_data={
                "context_length": 128,
                "forecast_length": 96,
                "encoder_hidden_dim": 256,
                "decoder_hidden_dim": 128,
            },
        )

    assert "encoder_hidden_dim" in str(exc_info.value)
    assert "decoder_hidden_dim" in str(exc_info.value)


def test_tide_runtime_adapter_enforces_from_scratch_training_mode() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="tide",
            config_data={
                "context_length": 128,
                "forecast_length": 96,
                "training_mode": "fine_tune",
            },
        )

    assert "training_mode" in str(exc_info.value)


def test_tide_runtime_adapter_enforces_mean_scaling() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="tide",
            config_data={
                "context_length": 128,
                "forecast_length": 96,
                "scaling": "std",
            },
        )

    assert "scaling" in str(exc_info.value)


def test_chronos2_runtime_adapter_default_covariates_match_model_default() -> None:
    runtime_config = build_model_runtime_config(
        model_type="chronos2",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
        },
    )

    assert runtime_config["covariate_cols"] == ["iob"]


def test_chronos2_runtime_adapter_reports_field_specific_numeric_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config(
            model_type="chronos2",
            config_data={
                "context_length": 128,
                "forecast_length": 96,
                "fine_tune_lr": "not-a-number",
            },
        )

    assert "fine_tune_lr must be a numeric value" in str(exc_info.value)


def test_tide_runtime_adapter_default_parity_matches_model_defaults() -> None:
    runtime_config = build_model_runtime_config(
        model_type="tide",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
        },
    )

    assert runtime_config["training_mode"] == "from_scratch"
    assert runtime_config["scaling"] == "mean"
    assert runtime_config["lr"] == pytest.approx(1.0e-3)


def test_tsmixer_model_config_reports_schema_errors(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "tsmixer_invalid.yaml",
        """
model_type: tsmixer
context_length: "128"
unknown_field: true
""".strip(),
    )

    with pytest.raises(ValueError) as exc_info:
        load_model_config_from_yaml(config_path, model_type="tsmixer")

    message = str(exc_info.value)
    assert config_path in message
    assert "context_length" in message
    assert "unknown_field" in message


def test_chronos2_model_config_reports_joint_target_schema_errors(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "chronos2_invalid_joint_target.yaml",
        """
model_type: chronos2
context_length: 512
forecast_length: 96
joint_target_cols: [iob]
target_col: bg_mM
""".strip(),
    )

    with pytest.raises(ValueError) as exc_info:
        load_model_config_from_yaml(config_path, model_type="chronos2")

    assert "joint_target_cols" in str(exc_info.value)


def test_tsmixer_runtime_adapter_builds_runtime_config() -> None:
    runtime_config = build_model_runtime_config(
        model_type="tsmixer",
        config_data={
            "context_length": 128,
            "forecast_length": 96,
            "lr": 0.002,
            "covariate_cols": ["iob", "cob"],
            "use_cpu": True,
        },
    )

    assert runtime_config["learning_rate"] == pytest.approx(0.002)
    assert runtime_config["covariate_cols"] == ["iob", "cob"]
    assert runtime_config["use_cpu"] is True
    assert runtime_config["context_length"] == 128
    assert runtime_config["forecast_length"] == 96


def test_tsmixer_smoke_profile_validates_via_schema_loader() -> None:
    loaded = load_model_config_from_yaml(
        str(TSMIXER_SMOKE_CONFIG_PATH),
        model_type="tsmixer",
    )

    assert loaded["model_type"] == "tsmixer"
    assert loaded["training_mode"] == "from_scratch"
    assert loaded["covariate_cols"] == ["iob", "cob"]
    assert loaded["quantile_levels"] == [0.1, 0.5, 0.9]


def test_tsmixer_smoke_profile_builds_runtime_payload() -> None:
    loaded = load_model_config_from_yaml(
        str(TSMIXER_SMOKE_CONFIG_PATH),
        model_type="tsmixer",
    )
    runtime_config = build_model_runtime_config("tsmixer", loaded)

    assert runtime_config["learning_rate"] == pytest.approx(0.001)
    assert runtime_config["num_epochs"] == 1
    assert runtime_config["batch_size"] == 32
    assert runtime_config["target_col"] == "bg_mM"
    assert runtime_config["patient_col"] == "p_num"
    assert runtime_config["time_col"] == "datetime"


def test_model_config_registry_exposes_tsmixer_schema_and_adapter() -> None:
    registered = get_registered_model_config_types()
    for model_type in [
        "chronos2",
        "deepar",
        "moment",
        "moirai",
        "naive_baseline",
        "patchtst",
        "statistical",
        "sundial",
        "tft",
        "tide",
        "timesfm",
        "timegrad",
        "toto",
        "ttm",
        "tsmixer",
    ]:
        assert model_type in registered
    assert get_model_config_schema("sundial") is not None
    assert get_model_config_schema("tsmixer") is not None
    assert get_model_config_schema("deepar") is not None
    assert get_model_config_schema("ttm") is not None
    assert get_model_config_schema("moment") is not None
    assert get_model_config_schema("timesfm") is not None
    assert get_model_config_schema("timegrad") is not None
    assert get_model_config_schema("toto") is not None


def test_tsmixer_factory_path_uses_schema_adapter_for_lr_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_tsmixer_module(monkeypatch)

    config = GenericModelConfig(
        model_type="tsmixer",
        model_path="",
        context_length=128,
        forecast_length=96,
        batch_size=16,
        num_epochs=2,
        learning_rate=1e-4,
        extra_config={"lr": 0.003, "covariate_cols": ["iob"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.learning_rate == pytest.approx(0.003)
    assert model.config.covariate_cols == ["iob"]


def test_tsmixer_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tsmixer_module(monkeypatch)

    config = GenericModelConfig(
        model_type="tsmixer",
        model_path="",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_ttm_factory_path_uses_schema_adapter_for_lr_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="ttm",
        config_class_name="TTMConfig",
        forecaster_class_name="TTMForecaster",
    )

    config = GenericModelConfig(
        model_type="ttm",
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=128,
        forecast_length=96,
        batch_size=32,
        num_epochs=3,
        learning_rate=1e-4,
        extra_config={"lr": 0.002, "target_features": ["bg_mM"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.learning_rate == pytest.approx(0.002)
    assert model.config.target_features == ["bg_mM"]


def test_ttm_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="ttm",
        config_class_name="TTMConfig",
        forecaster_class_name="TTMForecaster",
    )

    config = GenericModelConfig(
        model_type="ttm",
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_sundial_factory_path_uses_schema_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="sundial",
        config_class_name="SundialConfig",
        forecaster_class_name="SundialForecaster",
    )

    config = GenericModelConfig(
        model_type="sundial",
        model_path="thuml/sundial-base-128m",
        context_length=128,
        forecast_length=72,
        training_mode="zero_shot",
        extra_config={"num_samples": 77},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.num_samples == 77
    assert model.config.training_mode == "zero_shot"


def test_sundial_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="sundial",
        config_class_name="SundialConfig",
        forecaster_class_name="SundialForecaster",
    )

    config = GenericModelConfig(
        model_type="sundial",
        model_path="thuml/sundial-base-128m",
        context_length=128,
        forecast_length=96,
        training_mode="zero_shot",
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_moirai_factory_path_uses_schema_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="moirai",
        config_class_name="MoiraiConfig",
        forecaster_class_name="MoiraiForecaster",
    )

    config = GenericModelConfig(
        model_type="moirai",
        model_path="Salesforce/moirai-1.0-R-small",
        context_length=128,
        forecast_length=96,
        batch_size=32,
        num_epochs=2,
        learning_rate=1e-4,
        training_mode="fine_tune",
        extra_config={
            "lr": 0.0009,
            "past_covariate_dim": 0,
            "covariate_cols": [],
        },
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.learning_rate == pytest.approx(0.0009)
    assert model.config.past_covariate_dim == 0
    assert model.config.covariate_cols == []


def test_moirai_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="moirai",
        config_class_name="MoiraiConfig",
        forecaster_class_name="MoiraiForecaster",
    )

    config = GenericModelConfig(
        model_type="moirai",
        model_path="Salesforce/moirai-1.0-R-small",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_moment_factory_path_uses_schema_adapter_for_lr_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="moment",
        config_class_name="MomentConfig",
        forecaster_class_name="MomentForecaster",
    )

    config = GenericModelConfig(
        model_type="moment",
        model_path="AutonLab/MOMENT-1-small",
        context_length=128,
        forecast_length=96,
        batch_size=16,
        num_epochs=2,
        learning_rate=1e-4,
        extra_config={"lr": 0.0007, "covariate_cols": ["iob"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.learning_rate == pytest.approx(0.0007)
    assert model.config.covariate_cols == ["iob"]


def test_moment_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="moment",
        config_class_name="MomentConfig",
        forecaster_class_name="MomentForecaster",
    )

    config = GenericModelConfig(
        model_type="moment",
        model_path="AutonLab/MOMENT-1-small",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_timegrad_factory_path_uses_schema_adapter_for_lr_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="timegrad",
        config_class_name="TimeGradConfig",
        forecaster_class_name="TimeGradForecaster",
    )

    config = GenericModelConfig(
        model_type="timegrad",
        model_path="",
        context_length=128,
        forecast_length=96,
        batch_size=16,
        num_epochs=2,
        learning_rate=1e-4,
        extra_config={"lr": 0.0012, "target_features": ["bg_mM"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.learning_rate == pytest.approx(0.0012)
    assert model.config.target_features == ["bg_mM"]


def test_timegrad_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="timegrad",
        config_class_name="TimeGradConfig",
        forecaster_class_name="TimeGradForecaster",
    )

    config = GenericModelConfig(
        model_type="timegrad",
        model_path="",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_timesfm_factory_path_uses_schema_adapter_for_checkpoint_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="timesfm",
        config_class_name="TimesFMConfig",
        forecaster_class_name="TimesFMForecaster",
    )

    config = GenericModelConfig(
        model_type="timesfm",
        model_path="google/timesfm-2.0-500m-pytorch",
        context_length=128,
        forecast_length=96,
        batch_size=16,
        num_epochs=2,
        learning_rate=1e-4,
        extra_config={"checkpoint_path": "google/timesfm-2.0-500m-pytorch"},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.checkpoint_path == "google/timesfm-2.0-500m-pytorch"
    assert model.config.model_path == "google/timesfm-2.0-500m-pytorch"
    assert model.config.horizon_length == 96


def test_timesfm_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="timesfm",
        config_class_name="TimesFMConfig",
        forecaster_class_name="TimesFMForecaster",
    )

    config = GenericModelConfig(
        model_type="timesfm",
        model_path="google/timesfm-2.0-500m-pytorch",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_toto_factory_path_uses_schema_adapter_for_lr_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="toto",
        config_class_name="TotoConfig",
        forecaster_class_name="TotoForecaster",
    )

    config = GenericModelConfig(
        model_type="toto",
        model_path="Datadog/Toto-Open-Base-1.0",
        context_length=128,
        forecast_length=96,
        batch_size=16,
        num_epochs=2,
        learning_rate=1e-4,
        extra_config={"lr": 0.0005, "covariate_cols": ["iob"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.lr == pytest.approx(0.0005)
    assert model.config.covariate_cols == ["iob"]


def test_toto_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="toto",
        config_class_name="TotoConfig",
        forecaster_class_name="TotoForecaster",
    )

    config = GenericModelConfig(
        model_type="toto",
        model_path="Datadog/Toto-Open-Base-1.0",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_deepar_factory_path_uses_schema_adapter_for_learning_rate_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_deepar_module(monkeypatch)

    config = GenericModelConfig(
        model_type="deepar",
        model_path="",
        context_length=128,
        forecast_length=96,
        extra_config={"learning_rate": 0.002, "covariate_cols": ["iob"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.lr == pytest.approx(0.002)
    assert model.config.covariate_cols == ["iob"]


def test_deepar_factory_path_reports_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_deepar_module(monkeypatch)

    config = GenericModelConfig(
        model_type="deepar",
        model_path="",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_chronos2_factory_path_uses_schema_adapter_without_num_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_chronos2_module(monkeypatch)

    config = GenericModelConfig(
        model_type="chronos2",
        model_path="autogluon/chronos-2",
        context_length=128,
        forecast_length=96,
        num_epochs=7,
        extra_config={"fine_tune_steps": 200, "covariate_cols": ["iob", "cob"]},
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.fine_tune_steps == 200
    assert model.config.covariate_cols == ["iob", "cob"]
    assert not hasattr(model.config, "num_epochs")


def test_chronos2_factory_path_uses_schema_covariate_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_chronos2_module(monkeypatch)

    config = GenericModelConfig(
        model_type="chronos2",
        model_path="autogluon/chronos-2",
        context_length=128,
        forecast_length=96,
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    assert model.config.covariate_cols == ["iob"]


@pytest.mark.parametrize(
    (
        "model_type",
        "module_name",
        "config_class_name",
        "forecaster_class_name",
        "extra_config",
        "expected_model_name",
        "expected_covariates",
        "expected_lr",
    ),
    [
        pytest.param(
            "naive_baseline",
            "naive_baseline",
            "NaiveBaselineConfig",
            "NaiveBaselineForecaster",
            {},
            "Naive",
            [],
            None,
            id="naive_baseline-default-model-name",
        ),
        pytest.param(
            "statistical",
            "statistical",
            "StatisticalConfig",
            "StatisticalForecaster",
            {},
            "AutoARIMA",
            [],
            None,
            id="statistical-default-model-name",
        ),
        pytest.param(
            "patchtst",
            "patchtst",
            "PatchTSTConfig",
            "PatchTSTForecaster",
            {"learning_rate": 0.002, "covariate_cols": ["iob"]},
            None,
            ["iob"],
            pytest.approx(0.002),
            id="patchtst-learning-rate-alias",
        ),
        pytest.param(
            "tft",
            "tft",
            "TFTConfig",
            "TFTForecaster",
            {"learning_rate": 0.003, "covariate_cols": ["iob"]},
            None,
            ["iob"],
            pytest.approx(0.003),
            id="tft-learning-rate-alias",
        ),
        pytest.param(
            "tide",
            "tide",
            "TiDEConfig",
            "TiDEForecaster",
            {"learning_rate": 0.004, "covariate_cols": ["iob"]},
            None,
            ["iob"],
            pytest.approx(0.004),
            id="tide-learning-rate-alias",
        ),
    ],
)
def test_autogluon_family_factory_paths_use_schema_adapter(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    module_name: str,
    config_class_name: str,
    forecaster_class_name: str,
    extra_config: dict[str, object],
    expected_model_name: str | None,
    expected_covariates: list[str],
    expected_lr: object | None,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name=module_name,
        config_class_name=config_class_name,
        forecaster_class_name=forecaster_class_name,
    )

    config = GenericModelConfig(
        model_type=model_type,
        model_path="",
        context_length=128,
        forecast_length=96,
        extra_config=extra_config,
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model.config, fake_config_class)
    if expected_model_name is not None:
        assert model.config.model_name == expected_model_name
    assert model.config.covariate_cols == expected_covariates
    if expected_lr is not None:
        assert model.config.lr == expected_lr


@pytest.mark.parametrize(
    ("model_type", "module_name", "config_class_name", "forecaster_class_name"),
    [
        pytest.param(
            "naive_baseline",
            "naive_baseline",
            "NaiveBaselineConfig",
            "NaiveBaselineForecaster",
            id="naive_baseline",
        ),
        pytest.param(
            "statistical",
            "statistical",
            "StatisticalConfig",
            "StatisticalForecaster",
            id="statistical",
        ),
        pytest.param(
            "patchtst",
            "patchtst",
            "PatchTSTConfig",
            "PatchTSTForecaster",
            id="patchtst",
        ),
        pytest.param(
            "tft",
            "tft",
            "TFTConfig",
            "TFTForecaster",
            id="tft",
        ),
        pytest.param(
            "tide",
            "tide",
            "TiDEConfig",
            "TiDEForecaster",
            id="tide",
        ),
    ],
)
def test_autogluon_family_factory_paths_report_unknown_runtime_field(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    module_name: str,
    config_class_name: str,
    forecaster_class_name: str,
) -> None:
    _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name=module_name,
        config_class_name=config_class_name,
        forecaster_class_name=forecaster_class_name,
    )

    config = GenericModelConfig(
        model_type=model_type,
        model_path="",
        context_length=128,
        forecast_length=96,
        extra_config={"unknown_field": True},
    )

    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)

    assert "unknown_field" in str(exc_info.value)


def test_tide_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="tide",
        config_class_name="TiDEConfig",
        forecaster_class_name="TiDEForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(TIDE_SMOKE_CONFIG_PATH),
        model_type="tide",
    )
    config = ModelFactory.create_finetune_config(
        model_type="tide",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.training_mode == "from_scratch"
    assert model.config.scaling == "mean"
    assert model.config.lr == pytest.approx(9.31e-4)
    assert model.config.batch_size == 256
    assert model.config.forecast_length == 96


def test_tide_create_finetune_config_defaults_to_from_scratch_mode() -> None:
    config = ModelFactory.create_finetune_config(model_type="tide")
    assert config.training_mode == "from_scratch"


def test_runtime_adapter_reports_registered_types_for_unknown_model() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_model_runtime_config("unknown_model", {})

    message = str(exc_info.value)
    assert "unknown_model" in message
    assert "Registered adapter types" in message
    assert "tsmixer" in message


def test_ttm_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="ttm",
        config_class_name="TTMConfig",
        forecaster_class_name="TTMForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(TTM_SMOKE_CONFIG_PATH),
        model_type="ttm",
    )
    config = ModelFactory.create_finetune_config(
        model_type="ttm",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.training_mode == "zero_shot"
    assert model.config.freeze_backbone is True
    assert model.config.num_epochs == 0
    assert model.config.target_features == ["bg_mM"]


def test_tsmixer_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_tsmixer_module(monkeypatch)

    overrides = load_model_config_from_yaml(
        str(TSMIXER_SMOKE_CONFIG_PATH),
        model_type="tsmixer",
    )
    config = ModelFactory.create_finetune_config(
        model_type="tsmixer",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.covariate_cols == ["iob", "cob"]
    assert model.config.num_epochs == 1
    assert model.config.batch_size == 32
    assert model.config.learning_rate == pytest.approx(0.001)


def test_sundial_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="sundial",
        config_class_name="SundialConfig",
        forecaster_class_name="SundialForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(SUNDIAL_SMOKE_CONFIG_PATH),
        model_type="sundial",
    )
    config = ModelFactory.create_zero_shot_config(
        model_type="sundial",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.num_samples == 50
    assert model.config.training_mode == "zero_shot"
    assert model.config.context_length == 512
    assert model.config.forecast_length == 96


def test_moirai_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="moirai",
        config_class_name="MoiraiConfig",
        forecaster_class_name="MoiraiForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(MOIRAI_SMOKE_CONFIG_PATH),
        model_type="moirai",
    )
    config = ModelFactory.create_finetune_config(
        model_type="moirai",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.model_type == "moirai"
    assert model.config.patch_size == "auto"
    assert model.config.num_samples == 50
    assert model.config.past_covariate_dim == 0
    assert model.config.covariate_cols == []


def test_moment_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="moment",
        config_class_name="MomentConfig",
        forecaster_class_name="MomentForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(MOMENT_SMOKE_CONFIG_PATH),
        model_type="moment",
    )
    config = ModelFactory.create_finetune_config(
        model_type="moment",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.model_type == "moment"
    assert model.config.model_path == "AutonLab/MOMENT-1-small"
    assert model.config.training_mode == "fine_tune"
    assert model.config.covariate_cols == []


def test_timegrad_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="timegrad",
        config_class_name="TimeGradConfig",
        forecaster_class_name="TimeGradForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(TIMEGRAD_SMOKE_CONFIG_PATH),
        model_type="timegrad",
    )
    config = ModelFactory.create_finetune_config(
        model_type="timegrad",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.model_type == "timegrad"
    assert model.config.training_mode == "from_scratch"
    assert model.config.target_features == ["bg_mM"]
    assert model.config.diff_steps == 10


def test_toto_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="toto",
        config_class_name="TotoConfig",
        forecaster_class_name="TotoForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(TOTO_SMOKE_CONFIG_PATH),
        model_type="toto",
    )
    config = ModelFactory.create_finetune_config(
        model_type="toto",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.model_type == "toto"
    assert model.config.training_mode == "fine_tune"
    assert model.config.max_steps == 100
    assert model.config.lr == pytest.approx(1.0e-4)


def test_timesfm_factory_path_supports_real_smoke_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config_class = _install_fake_model_family_module(
        monkeypatch=monkeypatch,
        module_name="timesfm",
        config_class_name="TimesFMConfig",
        forecaster_class_name="TimesFMForecaster",
    )

    overrides = load_model_config_from_yaml(
        str(TIMESFM_SMOKE_CONFIG_PATH),
        model_type="timesfm",
    )
    config = ModelFactory.create_finetune_config(
        model_type="timesfm",
        extra_config=overrides,
    )
    model = ModelFactory.create_model(config)

    assert isinstance(model.config, fake_config_class)
    assert model.config.model_type == "timesfm"
    assert model.config.training_mode == "fine_tune"
    assert model.config.learning_rate == pytest.approx(1.0e-5)
    assert model.config.horizon_length == model.config.forecast_length
