"""Tests for forecasting pipeline config precedence behavior."""
# pyright: reportMissingImports=false

from pathlib import Path

import pandas as pd

from src.workflows.forecasting import pipeline


def test_step5_preserves_yaml_precedence_when_cli_overrides_are_none(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_gpu_info(_logger):
        del _logger
        return {"gpu_available": False}

    class FakeConfig:
        model_type = "ttm"
        context_length = 512
        forecast_length = 96
        model_path = "fake-path"
        training_mode = "fine_tune"
        freeze_backbone = False
        num_epochs = 35

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def fit(self, train_data, output_dir):
            del train_data, output_dir
            return {"loss": 0.1}

        def save(self, save_path):
            Path(save_path).write_text("ok", encoding="utf-8")

    def fake_create_finetune_config(**kwargs):
        captured["num_epochs"] = kwargs["num_epochs"]
        captured["batch_size"] = kwargs["batch_size"]
        captured["extra_config"] = kwargs["extra_config"]
        return FakeConfig()

    def fake_create_model(config):
        return FakeModel(config)

    def fake_phase_evaluate_and_plot(**_kwargs):
        del _kwargs
        return None

    monkeypatch.setattr(pipeline, "runtime_get_gpu_info", fake_gpu_info)
    monkeypatch.setattr(
        pipeline.ModelFactory,
        "create_finetune_config",
        fake_create_finetune_config,
    )
    monkeypatch.setattr(pipeline.ModelFactory, "create_model", fake_create_model)
    monkeypatch.setattr(
        pipeline, "phase_evaluate_and_plot", fake_phase_evaluate_and_plot
    )

    combined = pd.DataFrame(
        {
            "patient_id": [1],
            "id": [1],
            "datetime": [pd.Timestamp("2025-01-01")],
            "bg_mM": [6.0],
        }
    )

    _, config, _, _ = pipeline.step5_train_model(
        model_type="ttm",
        combined_data=combined,
        dataset_names=["demo"],
        training_columns=list(combined.columns),
        config_dir="configs/data/holdout_10pct",
        output_dir=tmp_path.as_posix(),
        num_epochs=None,
        batch_size=None,
        model_config_overrides={"num_epochs": 35, "batch_size": 256},
    )

    assert captured["num_epochs"] is None
    assert captured["batch_size"] is None
    assert config.num_epochs == 35
