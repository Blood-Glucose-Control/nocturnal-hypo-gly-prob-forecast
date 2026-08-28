"""Tests for forecasting workflow request schema validation."""
# pyright: reportMissingImports=false

from pathlib import Path

import pytest

from src.config.schemas.workflow_configs import (
    get_model_feature_override_columns,
    load_forecasting_eval_sweep_spec_from_yaml,
    load_forecasting_train_sweep_spec_from_yaml,
    validate_forecasting_workflow_request,
)


def test_forecasting_workflow_request_schema_valid_payload() -> None:
    validated = validate_forecasting_workflow_request(
        {
            "model_type": "tsmixer",
            "datasets": ["aleppo_2017"],
            "config_dir": "configs/data/holdout_10pct",
            "output_dir": None,
            "skip_training": False,
            "skip_steps": [4, 7],
            "epochs": 2,
            "batch_size": 32,
            "model_config_path": "configs/models/tsmixer/00_iob_cob_smoke.yaml",
        }
    )

    assert validated.model_type == "tsmixer"
    assert validated.datasets == ["aleppo_2017"]
    assert validated.skip_steps == [4, 7]
    assert validated.epochs == 2
    assert validated.batch_size == 32
    assert validated.model_config_path == "configs/models/tsmixer/00_iob_cob_smoke.yaml"


def test_forecasting_workflow_request_schema_rejects_invalid_skip_step() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_forecasting_workflow_request(
            {
                "model_type": "tsmixer",
                "datasets": ["aleppo_2017"],
                "skip_steps": [0],
            }
        )

    assert "skip_steps" in str(exc_info.value)


def test_forecasting_workflow_request_schema_rejects_empty_datasets() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_forecasting_workflow_request(
            {
                "model_type": "tsmixer",
                "datasets": [],
            }
        )

    assert "datasets" in str(exc_info.value)


def test_model_feature_override_columns_extracts_valid_lists() -> None:
    columns = get_model_feature_override_columns(
        {
            "input_features": ["iob", "cob"],
            "target_features": ["bg_mM"],
            "batch_size": 32,
        }
    )

    assert columns == ["iob", "cob", "bg_mM"]


def test_model_feature_override_columns_rejects_invalid_types() -> None:
    with pytest.raises(ValueError) as exc_info:
        get_model_feature_override_columns(
            {
                "input_features": "iob",
                "target_features": ["bg_mM"],
            }
        )

    assert "input_features" in str(exc_info.value)


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_forecasting_train_sweep_spec_loader_validates_jobs(tmp_path: Path) -> None:
    spec = _write_yaml(
        tmp_path / "train_sweep.yaml",
        """
jobs:
  - model_config: " configs/models/ttm/fine_tune.yaml "
    datasets:
      - " aleppo_2017 "
      - " lynch_2022 "
        """,
    )

    parsed = load_forecasting_train_sweep_spec_from_yaml(spec)

    assert len(parsed.jobs) == 1
    assert parsed.jobs[0].model_config_path == "configs/models/ttm/fine_tune.yaml"
    assert parsed.jobs[0].datasets == ["aleppo_2017", "lynch_2022"]


def test_forecasting_train_sweep_spec_loader_rejects_empty_dataset_entry(
    tmp_path: Path,
) -> None:
    spec = _write_yaml(
        tmp_path / "train_sweep_invalid.yaml",
        """
jobs:
  - model_config: configs/models/ttm/fine_tune.yaml
    datasets:
      - " "
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_forecasting_train_sweep_spec_from_yaml(spec)

    assert "datasets[0]" in str(exc_info.value)


def test_forecasting_eval_sweep_spec_loader_applies_defaults_and_overrides(
    tmp_path: Path,
) -> None:
    spec = _write_yaml(
        tmp_path / "eval_sweep.yaml",
        """
probabilistic: false
no_dilate: true
forecast_length: 120
output_dir_template: " runs/{dataset} "
jobs:
  - model_config: " configs/models/ttm/fine_tune.yaml "
    context_length: 512
    finetuned_datasets: [" aleppo_2017 "]
  - model_config: "configs/models/ttm/zero_shot.yaml"
    context_length: 256
    zeroshot_datasets: ["lynch_2022"]
    forecast_length: 96
    probabilistic: true
    no_dilate: false
        """,
    )

    parsed = load_forecasting_eval_sweep_spec_from_yaml(spec)

    assert parsed.probabilistic is False
    assert parsed.no_dilate is True
    assert parsed.forecast_length == 120
    assert parsed.output_dir_template == "runs/{dataset}"
    assert parsed.jobs[0].model_config_path == "configs/models/ttm/fine_tune.yaml"
    assert parsed.jobs[0].finetuned_datasets == ["aleppo_2017"]
    assert parsed.jobs[0].forecast_length is None
    assert parsed.jobs[1].forecast_length == 96
    assert parsed.jobs[1].probabilistic is True
    assert parsed.jobs[1].no_dilate is False


def test_forecasting_eval_sweep_spec_loader_rejects_blank_output_template(
    tmp_path: Path,
) -> None:
    spec = _write_yaml(
        tmp_path / "eval_sweep_invalid.yaml",
        """
output_dir_template: " "
jobs:
  - model_config: configs/models/ttm/fine_tune.yaml
    context_length: 512
        """,
    )

    with pytest.raises(ValueError) as exc_info:
        load_forecasting_eval_sweep_spec_from_yaml(spec)

    assert "output_dir_template" in str(exc_info.value)
