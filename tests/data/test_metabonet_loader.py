from pathlib import Path

import pandas as pd
import pytest

from src.data.cache_manager import CacheManager
from src.data.diabetes_datasets.metabonet import metabonet as metabonet_module
from src.data.diabetes_datasets.metabonet.metabonet import MetabonetDataLoader


def _write_metabonet_split_files(raw_path: Path) -> None:
    raw_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "id": [101, 101, 202],
            "date": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00",
                "2024-01-02 00:00:00",
            ],
            "CGM": [180, 181, 190],
            "carbs": [10, 0, 15],
            "age": [30, 30, 40],
            "age_of_diagnosis": [12, 12, 20],
            "source_file": ["cohort_a.csv", "cohort_a.csv", "cohort_b.csv"],
            "treatment_group": ["A", "A", "B"],
            "ethnicity": ["eth_a", "eth_a", "eth_b"],
            "is_test": [False, False, False],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [101, 202],
            "date": ["2024-02-01 00:00:00", "2024-02-02 00:00:00"],
            "CGM": [175, 188],
            "row_id": ["seg_a", "seg_b"],
            "source_file": ["cohort_a.csv", "cohort_b.csv"],
        }
    )

    train_df.to_parquet(raw_path / "train.parquet", index=False)
    test_df.to_parquet(raw_path / "test.parquet", index=False)


def test_metabonet_loader_processes_train_cache_without_load_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_manager = CacheManager(cache_root=str(tmp_path))
    raw_path = cache_manager.get_absolute_path_by_type("metabonet", "raw")
    processed_path = cache_manager.get_absolute_path_by_type("metabonet", "processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": ["stale"]}).to_csv(
        processed_path / "540_static_covariates.csv", index=False
    )
    _write_metabonet_split_files(raw_path)

    monkeypatch.setattr(metabonet_module, "get_cache_manager", lambda: cache_manager)

    loader = MetabonetDataLoader(
        use_cached=True,
        load_all=False,
        eager_load_test_data=False,
    )

    assert loader.processed_data == {}

    partitions_path = processed_path / metabonet_module.PROCESSED_PATIENT_PARQUET_DIR
    cached_patient_partitions = sorted(
        path.name for path in partitions_path.iterdir() if path.is_dir()
    )
    assert len(cached_patient_partitions) == 2
    partition_patient_id_prefixes = sorted(
        partition_name[len("patient_id=") :].split("_", 1)[0]
        for partition_name in cached_patient_partitions
    )
    assert partition_patient_id_prefixes == ["101", "202"]
    for partition_name in cached_patient_partitions:
        partition_path = partitions_path / partition_name
        assert sorted(path.name for path in partition_path.glob("*.parquet"))
    assert (processed_path / metabonet_module.PROCESSED_COMPLETE_MARKER).read_text(
        encoding="utf-8"
    ) == "2"
    static_covariates_df = pd.read_csv(
        processed_path / metabonet_module.STATIC_COVARIATES_FILE
    )
    assert not (processed_path / "540_static_covariates.csv").exists()
    static_patient_id_prefixes = sorted(
        {
            patient_id.split("_", 1)[0]
            for patient_id in static_covariates_df["patient_id"].astype(str)
        }
    )
    assert static_patient_id_prefixes == ["101", "202"]
    assert "age" in static_covariates_df.columns
    assert "treatment_group" in static_covariates_df.columns
    assert "ethnicity" in static_covariates_df.columns
    assert "is_test" in static_covariates_df.columns


def test_metabonet_loader_load_all_materializes_processed_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_manager = CacheManager(cache_root=str(tmp_path))
    raw_path = cache_manager.get_absolute_path_by_type("metabonet", "raw")
    _write_metabonet_split_files(raw_path)

    monkeypatch.setattr(metabonet_module, "get_cache_manager", lambda: cache_manager)

    loader = MetabonetDataLoader(
        use_cached=False,
        load_all=True,
        eager_load_test_data=False,
    )

    assert loader.processed_data is not None
    assert len(loader.processed_data) == 2
    patient_id_prefixes = sorted(
        patient_id.split("_", 1)[0] for patient_id in loader.processed_data
    )
    assert patient_id_prefixes == ["101", "202"]
    patient_101_id = next(
        patient_id
        for patient_id in loader.processed_data
        if patient_id.split("_", 1)[0] == "101"
    )
    assert "bg_mM" in loader.processed_data[patient_101_id].columns
    assert "age" not in loader.processed_data[patient_101_id].columns
    assert "treatment_group" not in loader.processed_data[patient_101_id].columns


def test_metabonet_loader_load_test_data_returns_nested_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_manager = CacheManager(cache_root=str(tmp_path))
    raw_path = cache_manager.get_absolute_path_by_type("metabonet", "raw")
    _write_metabonet_split_files(raw_path)

    monkeypatch.setattr(metabonet_module, "get_cache_manager", lambda: cache_manager)

    loader = MetabonetDataLoader(
        use_cached=True,
        load_all=False,
        eager_load_test_data=False,
    )

    nested_test_data = loader.load_test_data(use_cached=False)

    assert len(nested_test_data) == 2
    patient_id_prefixes = sorted(
        patient_id.split("_", 1)[0] for patient_id in nested_test_data
    )
    assert patient_id_prefixes == ["101", "202"]
    patient_101_id = next(
        patient_id
        for patient_id in nested_test_data
        if patient_id.split("_", 1)[0] == "101"
    )
    assert set(nested_test_data[patient_101_id]) == {"seg_a"}
    assert "bg_mM" in nested_test_data[patient_101_id]["seg_a"].columns


def test_metabonet_loader_uses_fixed_static_covariate_column_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_manager = CacheManager(cache_root=str(tmp_path))
    raw_path = cache_manager.get_absolute_path_by_type("metabonet", "raw")
    raw_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "id": [101, 101, 202],
            "date": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00",
                "2024-01-02 00:00:00",
            ],
            "CGM": [180, 181, 190],
            "age": [30, 31, 40],
            "age_of_diagnosis": [12, 12, 20],
            "source_file": ["cohort_a.csv", "cohort_a.csv", "cohort_b.csv"],
            "treatment_group": ["A", "A", "B"],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [101, 202],
            "date": ["2024-02-01 00:00:00", "2024-02-02 00:00:00"],
            "CGM": [175, 188],
            "row_id": ["seg_a", "seg_b"],
            "source_file": ["cohort_a.csv", "cohort_b.csv"],
        }
    )
    train_df.to_parquet(raw_path / "train.parquet", index=False)
    test_df.to_parquet(raw_path / "test.parquet", index=False)

    monkeypatch.setattr(metabonet_module, "get_cache_manager", lambda: cache_manager)

    loader = MetabonetDataLoader(
        use_cached=False,
        load_all=True,
        eager_load_test_data=False,
    )

    patient_101_id = next(
        patient_id
        for patient_id in loader.processed_data
        if patient_id.split("_", 1)[0] == "101"
    )
    patient_df = loader.processed_data[patient_101_id]
    assert "age" not in patient_df.columns
    assert "age_of_diagnosis" not in patient_df.columns
    assert "treatment_group" not in patient_df.columns
    static_covariates_df = pd.read_csv(
        cache_manager.get_absolute_path_by_type("metabonet", "processed")
        / metabonet_module.STATIC_COVARIATES_FILE
    )
    patient_101_static = static_covariates_df.loc[
        static_covariates_df["patient_id"].astype(str) == patient_101_id
    ]
    assert patient_101_static["age"].iloc[0] == 30
    assert patient_101_static["age_of_diagnosis"].iloc[0] == 12
    assert patient_101_static["treatment_group"].iloc[0] == "A"


def test_metabonet_loader_keep_columns_always_includes_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_manager = CacheManager(cache_root=str(tmp_path))
    raw_path = cache_manager.get_absolute_path_by_type("metabonet", "raw")
    _write_metabonet_split_files(raw_path)

    monkeypatch.setattr(metabonet_module, "get_cache_manager", lambda: cache_manager)

    loader = MetabonetDataLoader(
        keep_columns=["carbs"],
        use_cached=False,
        load_all=True,
        eager_load_test_data=False,
    )

    patient_101_id = next(
        patient_id
        for patient_id in loader.processed_data
        if patient_id.split("_", 1)[0] == "101"
    )
    patient_df = loader.processed_data[patient_101_id]
    assert "carbs" in patient_df.columns
    assert "patient_id" in patient_df.columns
    assert "bg_mM" in patient_df.columns
    assert isinstance(patient_df.index, pd.DatetimeIndex)


def test_metabonet_loader_drops_all_null_static_columns_and_is_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_manager = CacheManager(cache_root=str(tmp_path))
    raw_path = cache_manager.get_absolute_path_by_type("metabonet", "raw")
    raw_path.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "id": [101, 101, 202],
            "date": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:05:00",
                "2024-01-02 00:00:00",
            ],
            "CGM": [180, 181, 190],
            "source_file": ["cohort_a.csv", "cohort_a.csv", "cohort_b.csv"],
            "is_test": [False, False, False],
            "ethnicity": [None, None, None],
            "treatment_group": [None, None, None],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [101, 202],
            "date": ["2024-02-01 00:00:00", "2024-02-02 00:00:00"],
            "CGM": [175, 188],
            "row_id": ["seg_a", "seg_b"],
            "source_file": ["cohort_a.csv", "cohort_b.csv"],
        }
    )
    train_df.to_parquet(raw_path / "train.parquet", index=False)
    test_df.to_parquet(raw_path / "test.parquet", index=False)

    monkeypatch.setattr(metabonet_module, "get_cache_manager", lambda: cache_manager)
    loader = MetabonetDataLoader(
        use_cached=False,
        load_all=True,
        eager_load_test_data=False,
    )

    patient_101_id = next(
        patient_id
        for patient_id in loader.processed_data
        if patient_id.split("_", 1)[0] == "101"
    )
    patient_df = loader.processed_data[patient_101_id]
    assert "is_test" not in patient_df.columns
    assert "ethnicity" not in patient_df.columns
    assert "treatment_group" not in patient_df.columns

    static_covariates_df = pd.read_csv(
        cache_manager.get_absolute_path_by_type("metabonet", "processed")
        / metabonet_module.STATIC_COVARIATES_FILE
    )
    patient_static = static_covariates_df.loc[
        static_covariates_df["patient_id"].astype(str) == patient_101_id
    ]
    assert str(patient_static["is_test"].iloc[0]).lower() in {"false", "0"}
    assert patient_static["ethnicity"].isna().all()
    assert patient_static["treatment_group"].isna().all()


def test_metabonet_loader_load_all_cached_path_respects_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_manager = CacheManager(cache_root=str(tmp_path))
    monkeypatch.setattr(metabonet_module, "get_cache_manager", lambda: cache_manager)

    checked_cache_marker = {"called": False}
    processed_called = {"called": False}

    def _fake_cache_exists(self: MetabonetDataLoader) -> bool:
        checked_cache_marker["called"] = True
        return False

    def _fake_process(self: MetabonetDataLoader):
        processed_called["called"] = True
        return {
            "synthetic_1": pd.DataFrame(
                {
                    "patient_id": ["synthetic_1"],
                    "bg_mM": [6.1],
                },
                index=pd.DatetimeIndex(
                    pd.to_datetime(["2024-01-01 00:00:00"]),
                    name="datetime",
                ),
            )
        }

    monkeypatch.setattr(
        MetabonetDataLoader,
        "_processed_cache_exists",
        _fake_cache_exists,
    )
    monkeypatch.setattr(
        MetabonetDataLoader,
        "_process_and_cache_data",
        _fake_process,
    )

    loader = MetabonetDataLoader(
        use_cached=True,
        load_all=True,
        eager_load_test_data=False,
    )

    assert checked_cache_marker["called"] is True
    assert processed_called["called"] is True
    assert loader.processed_data is not None
    assert "synthetic_1" in loader.processed_data


def test_reset_processed_patient_cache_removes_legacy_static_files(tmp_path: Path):
    loader = object.__new__(MetabonetDataLoader)
    processed_path = tmp_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    (processed_path / metabonet_module.PROCESSED_COMPLETE_MARKER).write_text(
        "2", encoding="utf-8"
    )
    (processed_path / metabonet_module.STATIC_COVARIATES_FILE).write_text(
        "patient_id\n", encoding="utf-8"
    )
    (processed_path / "540_static_covariates.csv").write_text(
        "patient_id\n", encoding="utf-8"
    )
    (processed_path / "123_static_covariates.csv").write_text(
        "patient_id\n", encoding="utf-8"
    )
    (processed_path / "101_full.csv").write_text("datetime,bg_mM\n", encoding="utf-8")
    (processed_path / "101_full.parquet").write_bytes(b"parquet")

    partitions_path = processed_path / metabonet_module.PROCESSED_PATIENT_PARQUET_DIR
    partition_dir = partitions_path / "patient_id=101_source-abc"
    partition_dir.mkdir(parents=True, exist_ok=True)
    (partition_dir / "part-000000.parquet").write_bytes(b"parquet")

    loader._reset_processed_patient_cache(processed_path)

    assert not (processed_path / metabonet_module.PROCESSED_COMPLETE_MARKER).exists()
    assert not (processed_path / metabonet_module.STATIC_COVARIATES_FILE).exists()
    assert not (processed_path / "540_static_covariates.csv").exists()
    assert not (processed_path / "123_static_covariates.csv").exists()
    assert not (processed_path / "101_full.csv").exists()
    assert not (processed_path / "101_full.parquet").exists()
    assert not partitions_path.exists()
