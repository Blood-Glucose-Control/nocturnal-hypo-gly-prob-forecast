from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_BASE_PATH = REPO_ROOT / "src/data/diabetes_datasets/dataset_base.py"
BROWN_LOADER_PATH = REPO_ROOT / "src/data/diabetes_datasets/brown_2019/brown_2019.py"
LYNCH_LOADER_PATH = REPO_ROOT / "src/data/diabetes_datasets/lynch_2022/lynch_2022.py"
TAMB_LOADER_PATH = (
    REPO_ROOT / "src/data/diabetes_datasets/tamborlane_2008/tamborlane_2008.py"
)


def _load_dataset_base_class():
    module = importlib.import_module("src.data.diabetes_datasets.dataset_base")
    return module.DatasetBase


def _sample_cached_processed_data() -> dict[str, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "patient_id": ["p001"] * 4,
            "bg_mM": [6.1, 6.0, 5.9, 6.2],
            "dose_units": [0.0, 0.5, 0.0, 1.0],
            "carbohydrate_g": [0.0, 10.0, 0.0, 0.0],
        },
        index=pd.date_range("2024-01-01 00:00:00", periods=4, freq="5min"),
    )
    data.index.name = "datetime"
    return {"p001": data}


def _sample_noncanonical_processed_data() -> dict[str, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 00:00:00", periods=4, freq="5min"),
            "patient_id": ["p001"] * 4,
            "bg_mM": [6.1, 6.0, 5.9, 6.2],
            "dose_units": [0.0, 0.5, 0.0, 1.0],
        }
    )
    return {"p001": data}


def _build_dummy_loader_class():
    dataset_base_class = _load_dataset_base_class()

    class DummyLoader(dataset_base_class):
        def __init__(self, keep_columns: list[str] | None):
            super().__init__()
            self.keep_columns = keep_columns
            self.use_cached = True

        @property
        def dataset_name(self):
            return "dummy_dataset"

        @property
        def description(self):
            return "dummy"

        def load_raw(self):
            return None

        def _process_and_cache_data(self):
            raise AssertionError(
                "_process_and_cache_data should not be called when cached data is available."
            )

        def _load_cached_processed_data(self):
            cached = _sample_cached_processed_data()
            return self._apply_keep_columns_filter(
                {pid: df.copy(deep=True) for pid, df in cached.items()}
            )

    return DummyLoader


def _get_patient_df(loader) -> pd.DataFrame:
    assert loader.processed_data is not None
    return next(iter(loader.processed_data.values()))


def test_base_keep_columns_filters_processed_data_without_mutation():
    dummy_loader_cls = _build_dummy_loader_class()
    requested_columns = ["datetime", "bg_mM", "does_not_exist"]
    original_request = requested_columns.copy()

    loader = dummy_loader_cls(keep_columns=requested_columns)
    loader.load_data()

    assert requested_columns == original_request
    assert loader.keep_columns == original_request

    patient_df = _get_patient_df(loader)
    assert list(patient_df.columns) == ["patient_id", "bg_mM"]
    assert (patient_df["patient_id"] == "p001").all()
    assert patient_df.index.name == "datetime"


def test_base_keep_columns_none_returns_full_processed_schema():
    dummy_loader_cls = _build_dummy_loader_class()

    loader = dummy_loader_cls(keep_columns=None)
    loader.load_data()

    patient_df = _get_patient_df(loader)
    assert list(patient_df.columns) == [
        "patient_id",
        "bg_mM",
        "dose_units",
        "carbohydrate_g",
    ]
    assert patient_df.index.name == "datetime"


def test_base_keep_columns_raises_for_non_datetime_index():
    dataset_base_class = _load_dataset_base_class()

    class InvalidShapeLoader(dataset_base_class):
        def __init__(self):
            super().__init__()
            self.keep_columns = ["bg_mM"]
            self.use_cached = True

        @property
        def dataset_name(self):
            return "dummy_dataset"

        @property
        def description(self):
            return "dummy"

        def load_raw(self):
            return None

        def _process_and_cache_data(self):
            raise AssertionError(
                "_process_and_cache_data should not be called when cached data is available."
            )

        def _load_cached_processed_data(self):
            return self._apply_keep_columns_filter(
                _sample_noncanonical_processed_data()
            )

    loader = InvalidShapeLoader()
    with pytest.raises(
        ValueError,
        match="must use DatetimeIndex before keep_columns filtering",
    ):
        loader.load_data()


def test_brown_loader_no_longer_filters_keep_columns_inside_processing():
    tree = ast.parse(BROWN_LOADER_PATH.read_text())
    source = BROWN_LOADER_PATH.read_text()

    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Brown2019DataLoader"
    )
    process_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_process_raw_data"
    )

    keep_columns_attribute_reads = [
        node
        for node in ast.walk(process_node)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "keep_columns"
    ]
    assert not keep_columns_attribute_reads
    assert "if self.keep_columns is not None" not in source


def test_loader_inits_do_not_mutate_keep_columns_argument():
    lynch_source = LYNCH_LOADER_PATH.read_text()
    tamb_source = TAMB_LOADER_PATH.read_text()
    base_source = DATASET_BASE_PATH.read_text()

    assert (
        'if keep_columns is not None and "datetime" not in keep_columns'
        not in lynch_source
    )
    assert "keep_columns.append(" not in tamb_source
    assert (
        'self.keep_columns = ["datetime", *first_patient_df.columns.tolist()]'
        not in base_source
    )


def test_load_cached_processed_data_forwards_keep_columns_with_required_columns():
    dataset_base_class = _load_dataset_base_class()

    class FakeCacheManager:
        def __init__(self):
            self.calls = []

        def load_processed_data(
            self,
            dataset_name,
            file_format="csv",
            keep_columns=None,
        ):
            self.calls.append(
                {
                    "dataset_name": dataset_name,
                    "file_format": file_format,
                    "keep_columns": keep_columns,
                }
            )
            return {"p001": pd.DataFrame({"bg_mM": [5.5]})}

    class LoaderWithRealCacheLoad(dataset_base_class):
        def __init__(self):
            super().__init__()
            self.keep_columns = ["dose_units"]
            self.cache_manager = FakeCacheManager()

        @property
        def dataset_name(self):
            return "dummy_dataset"

        @property
        def description(self):
            return "dummy"

        def load_raw(self):
            return None

        def _process_and_cache_data(self):
            raise AssertionError

    loader = LoaderWithRealCacheLoad()
    loader._load_cached_processed_data()

    assert loader.cache_manager.calls, "Expected load_processed_data to be called"
    keep_columns = loader.cache_manager.calls[0]["keep_columns"]
    assert keep_columns is not None
    assert keep_columns[:3] == ["datetime", "patient_id", "bg_mM"]
    assert "dose_units" in keep_columns


def test_load_data_applies_keep_columns_once_for_cached_path():
    dataset_base_class = _load_dataset_base_class()

    class CachedCountingLoader(dataset_base_class):
        def __init__(self):
            super().__init__()
            self.keep_columns = ["dose_units"]
            self.use_cached = True
            self.apply_calls = 0

        @property
        def dataset_name(self):
            return "dummy_dataset"

        @property
        def description(self):
            return "dummy"

        def load_raw(self):
            return None

        def _process_and_cache_data(self):
            raise AssertionError(
                "_process_and_cache_data should not run for cached path"
            )

        def _load_cached_processed_data(self):
            cached = _sample_cached_processed_data()
            return self._apply_keep_columns_filter(
                {pid: df.copy(deep=True) for pid, df in cached.items()}
            )

        def _apply_keep_columns_filter(self, patient_data):
            self.apply_calls += 1
            return super()._apply_keep_columns_filter(patient_data)

    loader = CachedCountingLoader()
    loader.load_data()
    assert loader.apply_calls == 1


def test_load_data_applies_keep_columns_once_for_processed_path():
    dataset_base_class = _load_dataset_base_class()

    class ProcessedCountingLoader(dataset_base_class):
        def __init__(self):
            super().__init__()
            self.keep_columns = ["dose_units"]
            self.use_cached = False
            self.apply_calls = 0

        @property
        def dataset_name(self):
            return "dummy_dataset"

        @property
        def description(self):
            return "dummy"

        def load_raw(self):
            return None

        def _process_and_cache_data(self):
            cached = _sample_cached_processed_data()
            return {pid: df.copy(deep=True) for pid, df in cached.items()}

        def _apply_keep_columns_filter(self, patient_data):
            self.apply_calls += 1
            return super()._apply_keep_columns_filter(patient_data)

    loader = ProcessedCountingLoader()
    loader.load_data()
    assert loader.apply_calls == 1
