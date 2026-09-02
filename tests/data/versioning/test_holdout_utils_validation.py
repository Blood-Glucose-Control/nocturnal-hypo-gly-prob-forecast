"""Regression tests for holdout validation edge cases."""

import pandas as pd

from src.data.versioning.holdout_config import (
    HoldoutConfig,
    HoldoutType,
    PatientHoldoutConfig,
    TemporalHoldoutConfig,
)
from src.data.versioning.holdout_utils import validate_holdout_config


class _RegistryStub:
    def __init__(
        self,
        config: HoldoutConfig,
        train_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
    ) -> None:
        self._config = config
        self._train_data = train_data
        self._holdout_data = holdout_data

    def get_holdout_config(self, dataset_name: str) -> HoldoutConfig:
        return self._config

    def load_dataset_with_split(
        self, dataset_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._train_data.copy(), self._holdout_data.copy()


def _temporal_config(
    min_train_samples: int = 1, min_holdout_samples: int = 1
) -> HoldoutConfig:
    return HoldoutConfig(
        dataset_name="demo",
        holdout_type=HoldoutType.TEMPORAL,
        temporal_config=TemporalHoldoutConfig(
            holdout_percentage=0.2,
            min_train_samples=min_train_samples,
            min_holdout_samples=min_holdout_samples,
        ),
    )


def _patient_config(
    min_train_patients: int = 1, min_holdout_patients: int = 1
) -> HoldoutConfig:
    return HoldoutConfig(
        dataset_name="demo",
        holdout_type=HoldoutType.PATIENT_BASED,
        patient_config=PatientHoldoutConfig(
            holdout_patients=[],
            holdout_percentage=0.5,
            min_train_patients=min_train_patients,
            min_holdout_patients=min_holdout_patients,
            random_seed=42,
        ),
    )


def _hybrid_config() -> HoldoutConfig:
    return HoldoutConfig(
        dataset_name="demo",
        holdout_type=HoldoutType.HYBRID,
        temporal_config=TemporalHoldoutConfig(
            holdout_percentage=0.2,
            min_train_samples=1,
            min_holdout_samples=1,
        ),
        patient_config=PatientHoldoutConfig(
            holdout_patients=["3"],
            holdout_percentage=0.5,
            min_train_patients=1,
            min_holdout_patients=1,
            random_seed=42,
        ),
    )


def test_validate_holdout_config_flags_inconsistent_patient_id_columns() -> None:
    train_data = pd.DataFrame(
        {
            "patient_id": [1] * 120,
            "datetime": pd.date_range("2024-01-01", periods=120, freq="h"),
            "bg": [100.0] * 120,
        }
    )
    holdout_data = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-06", periods=30, freq="h"),
            "bg": [110.0] * 30,
        }
    )
    registry = _RegistryStub(_temporal_config(), train_data, holdout_data)

    results = validate_holdout_config("demo", registry, verbose=False)

    assert any("Inconsistent split columns" in err for err in results["errors"])


def test_validate_holdout_config_detects_temporal_ordering_for_numeric_patients() -> (
    None
):
    train_data = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-05T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                ]
            ),
            "bg": [100.0, 101.0, 99.0, 98.0],
        }
    )
    holdout_data = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "datetime": pd.to_datetime(["2024-01-04T00:00:00", "2024-01-03T00:00:00"]),
            "bg": [102.0, 97.0],
        }
    )
    registry = _RegistryStub(_temporal_config(), train_data, holdout_data)

    results = validate_holdout_config("demo", registry, verbose=False)

    assert any("Temporal ordering issue" in err for err in results["errors"])


def test_validate_holdout_config_accepts_datetime_temporal_column() -> None:
    train_data = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "datetime": pd.to_datetime(["2024-01-05T00:00:00", "2024-01-06T00:00:00"]),
            "bg": [100.0, 101.0],
        }
    )
    holdout_data = pd.DataFrame(
        {
            "patient_id": [1],
            "datetime": pd.to_datetime(["2024-01-04T00:00:00"]),
            "bg": [102.0],
        }
    )
    registry = _RegistryStub(_temporal_config(), train_data, holdout_data)

    results = validate_holdout_config("demo", registry, verbose=False)

    assert any("Temporal ordering issue" in err for err in results["errors"])


def test_validate_holdout_config_uses_temporal_config_thresholds() -> None:
    train_data = pd.DataFrame(
        {
            "patient_id": [1] * 50,
            "datetime": pd.date_range("2024-01-01", periods=50, freq="h"),
            "bg": [100.0] * 50,
        }
    )
    holdout_data = pd.DataFrame(
        {
            "patient_id": [1] * 10,
            "datetime": pd.date_range("2024-01-03", periods=10, freq="h"),
            "bg": [101.0] * 10,
        }
    )
    registry = _RegistryStub(
        _temporal_config(min_train_samples=40, min_holdout_samples=5),
        train_data,
        holdout_data,
    )

    results = validate_holdout_config("demo", registry, verbose=False)

    assert not any("below configured minimum" in err for err in results["errors"])


def test_validate_holdout_config_rejects_time_column_without_datetime() -> None:
    train_data = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "time": pd.date_range("2024-01-01", periods=2, freq="h"),
            "bg": [100.0, 101.0],
        }
    )
    holdout_data = pd.DataFrame(
        {
            "patient_id": [1],
            "time": pd.date_range("2024-01-02", periods=1, freq="h"),
            "bg": [102.0],
        }
    )
    registry = _RegistryStub(_temporal_config(), train_data, holdout_data)

    results = validate_holdout_config("demo", registry, verbose=False)

    assert any("missing required 'datetime' column" in err for err in results["errors"])


def test_validate_holdout_config_uses_patient_config_thresholds() -> None:
    train_data = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "datetime": pd.date_range("2024-01-01", periods=4, freq="h"),
            "bg": [100.0, 101.0, 99.0, 98.0],
        }
    )
    holdout_data = pd.DataFrame(
        {
            "patient_id": [3, 3],
            "datetime": pd.date_range("2024-01-02", periods=2, freq="h"),
            "bg": [102.0, 103.0],
        }
    )
    registry = _RegistryStub(
        _patient_config(min_train_patients=3, min_holdout_patients=2),
        train_data,
        holdout_data,
    )

    results = validate_holdout_config("demo", registry, verbose=False)

    assert any(
        "Training patient count below configured minimum" in err
        for err in results["errors"]
    )
    assert any(
        "Holdout patient count below configured minimum" in err
        for err in results["errors"]
    )


def test_validate_holdout_config_patient_split_skips_temporal_datetime_requirement() -> (
    None
):
    train_data = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "bg": [100.0, 101.0, 99.0, 98.0],
        }
    )
    holdout_data = pd.DataFrame(
        {
            "patient_id": [3, 3],
            "bg": [102.0, 103.0],
        }
    )
    registry = _RegistryStub(_patient_config(), train_data, holdout_data)

    results = validate_holdout_config("demo", registry, verbose=False)

    assert not any(
        "missing required 'datetime' column" in err for err in results["errors"]
    )


def test_validate_holdout_config_hybrid_runs_temporal_checks() -> None:
    train_data = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "bg": [100.0, 99.0],
        }
    )
    holdout_data = pd.DataFrame(
        {
            "patient_id": [3],
            "bg": [102.0],
        }
    )
    registry = _RegistryStub(_hybrid_config(), train_data, holdout_data)

    results = validate_holdout_config("demo", registry, verbose=False)

    assert any("missing required 'datetime' column" in err for err in results["errors"])
