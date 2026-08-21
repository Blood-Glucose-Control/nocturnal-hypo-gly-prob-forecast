"""Tests for schema-validated holdout config loading and registry wiring."""

from pathlib import Path

import pytest

from src.config.schemas.data_configs import (
    build_holdout_runtime_config,
    load_holdout_runtime_config_from_yaml,
)
from src.data.versioning.dataset_registry import DatasetRegistry
from src.data.versioning.holdout_config import HoldoutConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_holdout_yaml_schema_loader_matches_legacy_runtime_for_fixture() -> None:
    fixture = REPO_ROOT / "configs" / "data" / "holdout_10pct" / "aleppo_2017.yaml"

    legacy = HoldoutConfig.load(fixture)
    schema_loaded = load_holdout_runtime_config_from_yaml(fixture)

    assert schema_loaded.to_dict() == legacy.to_dict()


def test_build_holdout_runtime_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_holdout_runtime_config(
            {
                "dataset_name": "demo",
                "holdout_type": "hybrid",
                "temporal_config": {
                    "holdout_percentage": 0.1,
                    "min_train_samples": 100,
                    "min_holdout_samples": 20,
                },
                "patient_config": {
                    "holdout_patients": [],
                    "holdout_percentage": 0.2,
                    "min_train_patients": 2,
                    "min_holdout_patients": 1,
                    "random_seed": 42,
                },
                "unexpected_root_key": True,
            }
        )

    assert "unexpected_root_key" in str(exc_info.value)


def test_build_holdout_runtime_config_requires_hybrid_sections() -> None:
    with pytest.raises(ValueError) as exc_info:
        build_holdout_runtime_config(
            {
                "dataset_name": "demo",
                "holdout_type": "hybrid",
                "temporal_config": {
                    "holdout_percentage": 0.1,
                    "min_train_samples": 100,
                    "min_holdout_samples": 20,
                },
            }
        )

    assert "patient_config" in str(exc_info.value)


def test_dataset_registry_uses_schema_validation_for_holdout_configs(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "demo_dataset.yaml",
        """
dataset_name: demo_dataset
holdout_type: hybrid
temporal_config:
  holdout_percentage: 0.1
  min_train_samples: 100
  min_holdout_samples: 20
patient_config:
  holdout_patients: []
  holdout_percentage: 0.2
  min_train_patients: 2
  min_holdout_patients: 1
  random_seed: 42
unexpected_root_key: true
""".strip(),
    )

    registry = DatasetRegistry(holdout_config_dir=tmp_path)
    config = registry.get_holdout_config("demo_dataset")

    assert config is None
