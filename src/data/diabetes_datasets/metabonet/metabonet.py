# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ...cache_manager import get_cache_manager
from ...dataset_configs import DatasetConfig, get_dataset_config
from ...models import DatasetSourceType
from ..dataset_base import DatasetBase, ProcessedPatientDataFrames
from .data_cleaner import (
    build_nested_test_data,
    normalize_metabonet_dataframe,
)

logger = logging.getLogger(__name__)
PROCESSED_COMPLETE_MARKER = ".metabonet_train_cache_complete"
STATIC_COVARIATES_FILE = "static_covariates.csv"
PIECEWISE_STATIC_COVARIATES_FILE = "piecewise_static_covariates.parquet"
PROCESSED_PATIENT_PARQUET_DIR = "patient_timeseries_parquet"
PATIENT_PARTITION_PREFIX = "patient_id="
PATIENT_PARQUET_SUFFIX = ".parquet"
STATIC_COVARIATE_COLUMNS = (
    "age",
    "age_of_diagnosis",
    "ethnicity",
    "extension_date",
    "gender",
    "is_test",
    "is_pregnant",
    "randomization_date",
    "source_file",
    "treatment_group",
)
PIECEWISE_STATIC_COVARIATE_COLUMNS = (
    "height",
    "weight",
    "cgm_device",
    "insulin_delivery_algorithm",
    "insulin_delivery_device",
    "insulin_delivery_modality",
    "insulin_type_basal",
    "insulin_type_bolus",
    "subject_split_across_traintest",
    "weight",
)


class MetabonetDataLoader(DatasetBase):
    """Loader scaffold for Metabonet contest train/test assets."""

    def __init__(
        self,
        keep_columns: list[str] | None = None,
        use_cached: bool = True,
        parallel: bool = True,
        max_workers: int = 14,
        train_file_name: str = "train.parquet",
        test_file_name: str = "test.parquet",
        test_segment_column: str | None = None,
        eager_load_test_data: bool = False,
        load_all: bool = False,
        split_static_covariates: bool = True,
    ) -> None:
        super().__init__()
        self.keep_columns = keep_columns
        self.use_cached = use_cached
        self.parallel = parallel
        self.max_workers = max_workers
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.test_segment_column = test_segment_column
        self.eager_load_test_data = eager_load_test_data
        self.load_all = load_all
        self.split_static_covariates = split_static_covariates

        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)
        self.test_data: dict[str, dict[str, pd.DataFrame]] = {}
        self.static_covariates: pd.DataFrame | None = None
        self.piecewise_static_covariates: pd.DataFrame | None = None

        logger.info(
            "Initializing %s with use_cached=%s.",
            self.__class__.__name__,
            self.use_cached,
        )
        self.load_data()

    @property
    def dataset_name(self) -> str:
        return DatasetSourceType.METABONET.value

    @property
    def description(self) -> str:
        return (
            "Metabonet contest dataset loader with explicit train/test split handling."
        )

    def load_data(self) -> None:
        if self.use_cached:
            if self.load_all:
                if self._processed_cache_exists():
                    cached_data = self._load_cached_processed_data()
                    if cached_data is None:
                        raise ValueError(
                            "Metabonet cache marker exists but cached processed data could not be loaded."
                        )
                    self.processed_data = cached_data
                    self.static_covariates = self._load_static_covariates_from_cache()
                    self.piecewise_static_covariates = (
                        self._load_piecewise_static_covariates_from_cache()
                    )
                else:
                    self.processed_data = self._apply_keep_columns_filter(
                        self._process_and_cache_data()
                    )
            else:
                if self._processed_cache_exists():
                    logger.info(
                        "Found cached processed data for %s; skipping in-memory load because load_all=False.",
                        self.dataset_name,
                    )
                    self.processed_data = {}
                    self.static_covariates = self._load_static_covariates_from_cache()
                    self.piecewise_static_covariates = (
                        self._load_piecewise_static_covariates_from_cache()
                    )
                else:
                    self.processed_data = self._process_and_cache_data()
        else:
            self.processed_data = self._apply_keep_columns_filter(
                self._process_and_cache_data()
            )

        if self.eager_load_test_data and not self.test_data:
            self.test_data = self.load_test_data(use_cached=self.use_cached)

    def load_raw(self) -> pd.DataFrame:
        self.cache_manager.ensure_raw_data(self.dataset_name, self.dataset_config)
        raw_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "raw"
        )
        return self._read_split_frame(
            raw_path, split_name="train", file_name=self.train_file_name
        )

    def load_test_data(
        self, *, use_cached: bool = True
    ) -> dict[str, dict[str, pd.DataFrame]]:
        if use_cached and self.cache_manager.nested_test_data_exists(
            self.dataset_name,
            dataset_type="test",
        ):
            cached_nested_data = self.cache_manager.load_nested_test_data(
                self.dataset_name,
                dataset_type="test",
            )
            if cached_nested_data is not None:
                return self._apply_keep_columns_to_nested_data(cached_nested_data)

        self.cache_manager.ensure_raw_data(self.dataset_name, self.dataset_config)
        raw_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "raw"
        )
        raw_test_df = self._load_raw_test_split(raw_path)
        normalized_test_df = normalize_metabonet_dataframe(
            raw_test_df,
            split_name="test",
            require_bg=False,
        )
        nested_test_data = build_nested_test_data(
            normalized_test_df,
            segment_column=self.test_segment_column,
        )
        self.cache_manager.save_nested_test_data(self.dataset_name, nested_test_data)
        return self._apply_keep_columns_to_nested_data(nested_test_data)

    def _load_cached_processed_data(self) -> ProcessedPatientDataFrames | None:
        import pyarrow.parquet as pq

        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        partition_dirs = self._list_cached_patient_partition_dirs(processed_path)
        if not partition_dirs:
            return None

        keep_columns = self._get_effective_keep_columns()
        requested_columns: list[str] | None = None
        if keep_columns is not None:
            requested_columns = list(
                dict.fromkeys(
                    [
                        "datetime",
                        *[column for column in keep_columns if column != "datetime"],
                    ]
                )
            )

        self._configure_pyarrow_threads()

        cached_data: ProcessedPatientDataFrames = {}
        for partition_dir in partition_dirs:
            parquet_files = sorted(partition_dir.glob(f"*{PATIENT_PARQUET_SUFFIX}"))
            if not parquet_files:
                continue

            patient_id = self._extract_patient_id_from_partition_dir(partition_dir)
            available_columns = pq.ParquetFile(parquet_files[0]).schema.names
            read_columns: list[str] | None = None
            if requested_columns is not None:
                read_columns = [
                    column
                    for column in requested_columns
                    if column in available_columns
                ]
                if "datetime" not in read_columns:
                    raise ValueError(
                        f"Cached Metabonet parquet for patient {patient_id} is missing required datetime column."
                    )
                missing_columns = [
                    column
                    for column in requested_columns
                    if column not in read_columns and column != "datetime"
                ]
                if missing_columns:
                    logger.warning(
                        "Patient %s missing requested columns %s; available=%s",
                        patient_id,
                        missing_columns,
                        available_columns,
                    )

            frame_parts = [
                pd.read_parquet(parquet_file, columns=read_columns)
                for parquet_file in parquet_files
            ]
            if not frame_parts:
                continue
            patient_df = pd.concat(frame_parts, ignore_index=True)

            try:
                datetime_index = pd.to_datetime(
                    patient_df["datetime"], format="mixed", errors="raise"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed parsing cached datetime values in {partition_dir}"
                ) from exc

            patient_df = patient_df.drop(columns=["datetime"])
            patient_df.index = pd.DatetimeIndex(datetime_index, name="datetime")
            cached_data[patient_id] = patient_df.sort_index()

        return cached_data if cached_data else None

    def _process_and_cache_data(self) -> ProcessedPatientDataFrames:
        raw_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "raw"
        )
        train_file_path = self._resolve_split_file_path(
            raw_path=raw_path,
            split_name="train",
            file_name=self.train_file_name,
        )

        if train_file_path.suffix != ".parquet":
            raw_train_df = self._load_file(train_file_path)
            normalized_train_df = normalize_metabonet_dataframe(
                raw_train_df,
                split_name="train",
                require_bg=True,
            )
            return self._split_train_by_patient_with_progress(normalized_train_df)

        return self._process_train_parquet_by_patient(train_file_path)

    def _load_raw_test_split(self, raw_path: Path) -> pd.DataFrame:
        return self._read_split_frame(
            raw_path,
            split_name="test",
            file_name=self.test_file_name,
        )

    def _split_train_by_patient_with_progress(
        self, normalized_train_df: pd.DataFrame
    ) -> ProcessedPatientDataFrames:
        grouped = normalized_train_df.groupby("patient_id", sort=False)
        total_patients = int(normalized_train_df["patient_id"].nunique())
        train_data: ProcessedPatientDataFrames = {}
        static_covariates_by_patient: dict[str, dict[str, object]] = {}
        piecewise_rows: list[dict[str, object]] = []
        piecewise_observations: dict[
            tuple[str, str], list[tuple[pd.Timestamp, object]]
        ] = {}

        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.mkdir(parents=True, exist_ok=True)
        self._reset_processed_patient_cache(processed_path)
        patient_partitions_path = self._get_patient_partitions_path(processed_path)
        patient_partitions_path.mkdir(parents=True, exist_ok=True)

        for patient_id, patient_df in tqdm(
            grouped,
            total=total_patients,
            desc="Processing Metabonet train patients",
            unit="patient",
        ):
            normalized_patient_id = str(patient_id)
            self._update_piecewise_covariate_segments(
                patient_df=patient_df,
                patient_id=normalized_patient_id,
                observation_map=piecewise_observations,
            )
            static_row, timeseries_df = self._split_static_covariates_from_patient_df(
                patient_df.copy(),
                normalized_patient_id,
            )
            existing_static_row = static_covariates_by_patient.get(
                normalized_patient_id
            )
            if existing_static_row is None:
                static_covariates_by_patient[normalized_patient_id] = static_row
            else:
                static_covariates_by_patient[normalized_patient_id] = (
                    self._merge_static_covariate_rows(
                        existing_static_row,
                        static_row,
                    )
                )
            self._write_patient_parquet_fragment(
                processed_path=processed_path,
                patient_id=normalized_patient_id,
                timeseries_df=timeseries_df,
                fragment_id="000000",
            )
            if self.load_all:
                train_data[normalized_patient_id] = timeseries_df

        self._finalize_piecewise_covariate_segments(
            observation_map=piecewise_observations,
            completed_rows=piecewise_rows,
        )
        self._save_static_covariates(list(static_covariates_by_patient.values()))
        self._save_piecewise_static_covariates(piecewise_rows)
        self._write_processed_completion_marker(total_patients)
        if self.load_all:
            return train_data
        return {}

    def _process_train_parquet_by_patient(
        self,
        train_file_path: Path,
    ) -> ProcessedPatientDataFrames:
        import pyarrow.parquet as pq

        thread_count = self._configure_pyarrow_threads()
        use_threads = thread_count > 1

        parquet_columns = pq.ParquetFile(train_file_path).schema.names
        if "patient_id" not in parquet_columns and "id" not in parquet_columns:
            raise ValueError(
                "Metabonet train parquet must include either 'patient_id' or 'id' column."
            )

        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.mkdir(parents=True, exist_ok=True)
        self._reset_processed_patient_cache(processed_path)
        patient_partitions_path = self._get_patient_partitions_path(processed_path)
        patient_partitions_path.mkdir(parents=True, exist_ok=True)

        static_covariates_by_patient: dict[str, dict[str, object]] = {}
        piecewise_rows: list[dict[str, object]] = []
        piecewise_observations: dict[
            tuple[str, str], list[tuple[pd.Timestamp, object]]
        ] = {}
        seen_patients: set[str] = set()
        patient_progress = tqdm(
            desc="Processing Metabonet train patients", unit="patient"
        )
        parquet_file = pq.ParquetFile(train_file_path)
        for batch_index, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=1_000_000,
                use_threads=use_threads,
            )
        ):
            batch_df = batch.to_pandas()
            normalized_batch_df = normalize_metabonet_dataframe(
                batch_df,
                split_name="train",
                require_bg=True,
            )
            for patient_id, patient_df in normalized_batch_df.groupby(
                "patient_id", sort=False
            ):
                normalized_patient_id = str(patient_id)
                self._update_piecewise_covariate_segments(
                    patient_df=patient_df,
                    patient_id=normalized_patient_id,
                    observation_map=piecewise_observations,
                )
                static_row, timeseries_df = (
                    self._split_static_covariates_from_patient_df(
                        patient_df.copy(),
                        normalized_patient_id,
                    )
                )
                existing_static_row = static_covariates_by_patient.get(
                    normalized_patient_id
                )
                if existing_static_row is None:
                    static_covariates_by_patient[normalized_patient_id] = static_row
                else:
                    static_covariates_by_patient[normalized_patient_id] = (
                        self._merge_static_covariate_rows(
                            existing_static_row,
                            static_row,
                        )
                    )

                self._write_patient_parquet_fragment(
                    processed_path=processed_path,
                    patient_id=normalized_patient_id,
                    timeseries_df=timeseries_df,
                    fragment_id=f"{batch_index:06d}",
                )
                if normalized_patient_id not in seen_patients:
                    seen_patients.add(normalized_patient_id)
                    patient_progress.update(1)
        patient_progress.close()

        if not seen_patients:
            raise ValueError("Metabonet train parquet has no patient IDs.")

        self._finalize_piecewise_covariate_segments(
            observation_map=piecewise_observations,
            completed_rows=piecewise_rows,
        )
        self._save_static_covariates(list(static_covariates_by_patient.values()))
        self._save_piecewise_static_covariates(piecewise_rows)
        self._write_processed_completion_marker(len(seen_patients))

        if self.load_all:
            cached_data = self._load_cached_processed_data()
            if cached_data is None:
                raise ValueError(
                    "Expected cached Metabonet train data after processing, but no cache was found."
                )
            return cached_data
        return {}

    def _apply_keep_columns_to_nested_data(
        self, nested_data: dict[str, dict[str, pd.DataFrame]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        keep_columns = self._get_effective_keep_columns()
        if keep_columns is None:
            return nested_data

        columns_to_keep = [column for column in keep_columns if column != "datetime"]
        filtered_nested_data: dict[str, dict[str, pd.DataFrame]] = {}
        for patient_id, patient_segments in nested_data.items():
            filtered_nested_data[patient_id] = {}
            for segment_id, segment_df in patient_segments.items():
                if not isinstance(segment_df.index, pd.DatetimeIndex):
                    raise ValueError(
                        "Metabonet test segment data must use DatetimeIndex before keep_columns filtering."
                    )
                available_columns = [
                    column for column in columns_to_keep if column in segment_df.columns
                ]
                filtered_nested_data[patient_id][segment_id] = (
                    segment_df[available_columns] if available_columns else segment_df
                )
        return filtered_nested_data

    def _read_split_frame(
        self,
        raw_path: Path,
        *,
        split_name: str,
        file_name: str,
    ) -> pd.DataFrame:
        resolved_file_path = self._resolve_split_file_path(
            raw_path=raw_path,
            split_name=split_name,
            file_name=file_name,
        )
        return self._load_file(resolved_file_path)

    def _resolve_split_file_path(
        self,
        *,
        raw_path: Path,
        split_name: str,
        file_name: str,
    ) -> Path:
        for candidate_name in self._candidate_file_names(file_name):
            candidate_path = raw_path / candidate_name
            if candidate_path.exists():
                return candidate_path

        raise FileNotFoundError(
            f"Metabonet {split_name} file not found in {raw_path}. "
            f"Tried: {self._candidate_file_names(file_name)}"
        )

    def _candidate_file_names(self, file_name: str) -> list[str]:
        candidates = [file_name]
        stem = Path(file_name).stem
        for extension in (".parquet", ".csv"):
            candidate_name = f"{stem}{extension}"
            if candidate_name not in candidates:
                candidates.append(candidate_name)
        return candidates

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path, low_memory=False)
        raise ValueError(
            f"Unsupported Metabonet file format for {file_path}. Expected .parquet or .csv."
        )

    def _processed_cache_exists(self) -> bool:
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        if not processed_path.exists():
            return False
        marker_path = processed_path / PROCESSED_COMPLETE_MARKER
        if not marker_path.exists():
            return False

        if self.split_static_covariates:
            if not (processed_path / STATIC_COVARIATES_FILE).exists():
                return False
            if not (processed_path / PIECEWISE_STATIC_COVARIATES_FILE).exists():
                return False

        patient_partition_dirs = self._list_cached_patient_partition_dirs(
            processed_path
        )
        if not patient_partition_dirs:
            return False

        try:
            expected_count = int(marker_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return False
        if expected_count <= 0:
            return False

        cached_patient_count = sum(
            1
            for partition_dir in patient_partition_dirs
            if any(partition_dir.glob(f"*{PATIENT_PARQUET_SUFFIX}"))
        )
        return cached_patient_count >= expected_count

    def _write_processed_completion_marker(self, patient_count: int) -> None:
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.mkdir(parents=True, exist_ok=True)
        marker_path = processed_path / PROCESSED_COMPLETE_MARKER
        marker_path.write_text(str(patient_count), encoding="utf-8")

    def _reset_processed_patient_cache(self, processed_path: Path) -> None:
        marker_path = processed_path / PROCESSED_COMPLETE_MARKER
        if marker_path.exists():
            marker_path.unlink()
        current_static_covariates = processed_path / STATIC_COVARIATES_FILE
        if current_static_covariates.exists():
            current_static_covariates.unlink()
        piecewise_static_covariates = processed_path / PIECEWISE_STATIC_COVARIATES_FILE
        if piecewise_static_covariates.exists():
            piecewise_static_covariates.unlink()
        for stale_static_covariates in processed_path.glob("*_static_covariates.csv"):
            if stale_static_covariates.name == STATIC_COVARIATES_FILE:
                continue
            stale_static_covariates.unlink()
        for patient_csv in processed_path.glob("*_full.csv"):
            patient_csv.unlink()
        for patient_parquet in processed_path.glob("*_full.parquet"):
            patient_parquet.unlink()
        patient_partitions_path = self._get_patient_partitions_path(processed_path)
        if patient_partitions_path.exists():
            shutil.rmtree(patient_partitions_path)

    def _get_patient_partitions_path(self, processed_path: Path) -> Path:
        return processed_path / PROCESSED_PATIENT_PARQUET_DIR

    def _get_patient_partition_dir(self, processed_path: Path, patient_id: str) -> Path:
        return (
            self._get_patient_partitions_path(processed_path)
            / f"{PATIENT_PARTITION_PREFIX}{patient_id}"
        )

    def _extract_patient_id_from_partition_dir(self, partition_dir: Path) -> str:
        if not partition_dir.name.startswith(PATIENT_PARTITION_PREFIX):
            raise ValueError(
                "Invalid Metabonet patient partition directory name: "
                f"{partition_dir.name}"
            )
        return partition_dir.name[len(PATIENT_PARTITION_PREFIX) :]

    def _list_cached_patient_partition_dirs(self, processed_path: Path) -> list[Path]:
        patient_partitions_path = self._get_patient_partitions_path(processed_path)
        if not patient_partitions_path.exists():
            return []
        return sorted(
            [
                partition_dir
                for partition_dir in patient_partitions_path.iterdir()
                if partition_dir.is_dir()
                and partition_dir.name.startswith(PATIENT_PARTITION_PREFIX)
            ],
            key=lambda path: path.name,
        )

    def _write_patient_parquet_fragment(
        self,
        *,
        processed_path: Path,
        patient_id: str,
        timeseries_df: pd.DataFrame,
        fragment_id: str,
    ) -> None:
        partition_dir = self._get_patient_partition_dir(processed_path, patient_id)
        partition_dir.mkdir(parents=True, exist_ok=True)
        fragment_path = partition_dir / f"part-{fragment_id}{PATIENT_PARQUET_SUFFIX}"
        timeseries_df.reset_index().to_parquet(fragment_path, index=False)

    def _split_static_covariates_from_patient_df(
        self,
        patient_df: pd.DataFrame,
        patient_id: str,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        static_row: dict[str, object] = {"patient_id": patient_id}
        if not self.split_static_covariates:
            return static_row, patient_df

        columns_to_drop: list[str] = []
        for column in STATIC_COVARIATE_COLUMNS:
            if column not in patient_df.columns:
                continue
            non_null_series = patient_df[column].dropna()
            static_row[column] = (
                non_null_series.iloc[0] if not non_null_series.empty else None
            )
            columns_to_drop.append(column)
        for column in PIECEWISE_STATIC_COVARIATE_COLUMNS:
            if column in patient_df.columns:
                columns_to_drop.append(column)

        return static_row, patient_df.drop(
            columns=list(dict.fromkeys(columns_to_drop)),
            errors="ignore",
        )

    def _save_static_covariates(self, static_rows: list[dict[str, object]]) -> None:
        if not static_rows:
            self.static_covariates = pd.DataFrame(columns=["patient_id"])
        else:
            self.static_covariates = pd.DataFrame(static_rows).drop_duplicates(
                subset=["patient_id"]
            )

        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.mkdir(parents=True, exist_ok=True)
        self.static_covariates.to_csv(
            processed_path / STATIC_COVARIATES_FILE,
            index=False,
        )

    def _load_static_covariates_from_cache(self) -> pd.DataFrame | None:
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        static_covariates_path = processed_path / STATIC_COVARIATES_FILE
        if not static_covariates_path.exists():
            return None
        return pd.read_csv(static_covariates_path, low_memory=False)

    def _save_piecewise_static_covariates(
        self, piecewise_rows: list[dict[str, object]]
    ) -> None:
        if not piecewise_rows:
            self.piecewise_static_covariates = pd.DataFrame(
                columns=[
                    "patient_id",
                    "covariate",
                    "start_datetime",
                    "end_datetime",
                    "value",
                ]
            )
        else:
            self.piecewise_static_covariates = pd.DataFrame(piecewise_rows)

        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        processed_path.mkdir(parents=True, exist_ok=True)
        self.piecewise_static_covariates.to_parquet(
            processed_path / PIECEWISE_STATIC_COVARIATES_FILE,
            index=False,
        )

    def _load_piecewise_static_covariates_from_cache(self) -> pd.DataFrame | None:
        processed_path = self.cache_manager.get_absolute_path_by_type(
            self.dataset_name, "processed"
        )
        piecewise_covariates_path = processed_path / PIECEWISE_STATIC_COVARIATES_FILE
        if not piecewise_covariates_path.exists():
            return None
        return pd.read_parquet(piecewise_covariates_path)

    def _update_piecewise_covariate_segments(
        self,
        *,
        patient_df: pd.DataFrame,
        patient_id: str,
        observation_map: dict[tuple[str, str], list[tuple[pd.Timestamp, object]]],
    ) -> None:
        if not self.split_static_covariates:
            return

        for covariate in PIECEWISE_STATIC_COVARIATE_COLUMNS:
            if covariate not in patient_df.columns:
                continue
            series = patient_df[covariate].dropna()
            if series.empty:
                continue

            state_key = (patient_id, covariate)
            covariate_observations = observation_map.setdefault(state_key, [])
            change_points = series[series.ne(series.shift())]
            covariate_observations.extend(
                [(timestamp, value) for timestamp, value in change_points.items()]
            )

    def _finalize_piecewise_covariate_segments(
        self,
        *,
        observation_map: dict[tuple[str, str], list[tuple[pd.Timestamp, object]]],
        completed_rows: list[dict[str, object]],
    ) -> None:
        if not self.split_static_covariates:
            return

        for (patient_id, covariate), observations in observation_map.items():
            sorted_observations = sorted(observations, key=lambda row: row[0])
            current_start: pd.Timestamp | None = None
            current_value: object | None = None
            for timestamp, value in sorted_observations:
                if current_value is None:
                    current_start = timestamp
                    current_value = value
                    continue

                if value == current_value:
                    continue

                if current_start is None:
                    raise ValueError(
                        "Piecewise covariate segment start is unexpectedly None."
                    )

                if timestamp == current_start:
                    current_value = value
                    continue

                completed_rows.append(
                    {
                        "patient_id": patient_id,
                        "covariate": covariate,
                        "start_datetime": current_start,
                        "end_datetime": timestamp,
                        "value": current_value,
                    }
                )
                current_start = timestamp
                current_value = value

            if current_start is None:
                continue
            completed_rows.append(
                {
                    "patient_id": patient_id,
                    "covariate": covariate,
                    "start_datetime": current_start,
                    "end_datetime": None,
                    "value": current_value,
                }
            )
        observation_map.clear()

    def _merge_static_covariate_rows(
        self,
        existing_row: dict[str, object],
        new_row: dict[str, object],
    ) -> dict[str, object]:
        merged_row = dict(existing_row)
        for key, value in new_row.items():
            if key == "patient_id":
                continue
            if key not in merged_row or merged_row[key] is None:
                merged_row[key] = value
        return merged_row

    def _configure_pyarrow_threads(self) -> int:
        import pyarrow as pa

        available_cpus = os.cpu_count() or 1
        if not self.parallel:
            thread_count = 1
        else:
            thread_count = max(1, min(self.max_workers, available_cpus))
        pa.set_cpu_count(thread_count)
        logger.info(
            "Configured pyarrow threads=%d (parallel=%s, max_workers=%d, cpu_count=%d)",
            thread_count,
            self.parallel,
            self.max_workers,
            available_cpus,
        )
        return thread_count
