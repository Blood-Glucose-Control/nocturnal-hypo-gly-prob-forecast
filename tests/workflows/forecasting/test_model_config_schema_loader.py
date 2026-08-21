"""Tests for schema-validated model config loading."""

from pathlib import Path

import pytest

from src.workflows.forecasting.modeling import load_model_config_from_yaml


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


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
