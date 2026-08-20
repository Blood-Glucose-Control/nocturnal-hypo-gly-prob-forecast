"""Unit tests for TimesFM patient-level train/validation splitting."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="requires torch",
)


def test_single_patient_split_keeps_patient_in_train():
    from src.models.timesfm.model import _split_train_val_patients

    train_pids, val_pids = _split_train_val_patients(["bro_92"], 0.2, seed=42)
    assert train_pids == {"bro_92"}
    assert val_pids == set()


def test_multi_patient_split_preserves_train_and_val_sets():
    from src.models.timesfm.model import _split_train_val_patients

    train_pids, val_pids = _split_train_val_patients(["p1", "p2", "p3"], 0.34, seed=42)
    assert train_pids
    assert val_pids
    assert train_pids.isdisjoint(val_pids)
    assert train_pids.union(val_pids) == {"p1", "p2", "p3"}
