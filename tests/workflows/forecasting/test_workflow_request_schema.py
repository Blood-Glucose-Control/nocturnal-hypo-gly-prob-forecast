"""Tests for forecasting workflow request schema validation."""

import pytest

from src.config.schemas import validate_forecasting_workflow_request


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
            "model_config": "configs/models/tsmixer/00_iob_cob_smoke.yaml",
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
