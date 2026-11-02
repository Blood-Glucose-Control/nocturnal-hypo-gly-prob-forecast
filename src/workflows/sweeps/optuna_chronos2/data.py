# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""MetaboNet data loading, temporal split, and window building for the PoC.

Responsibilities:

* Discover and load per-patient time series from the MetaboNet processed cache.
* Apply :mod:`features` engineering to each patient onto a regular 5-minute grid.
* Perform a per-patient temporal split, holding out the last ``val_days`` days
  as an internal validation region for hyperparameter selection.
* Build an AutoGluon ``TimeSeriesDataFrame`` training panel (gap-handled
  segments) for a chosen covariate subset.
* Build challenge-shaped validation windows: strided anchors in the held-out
  region, each with up to ``max_context`` steps of history and a 24-step
  (+120 min) forecast target, plus deterministic known-future covariates.

Loading is covariate-agnostic and done once; panel/window builders take the
covariate selection so a single loaded dataset serves every Optuna trial.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .features import (
    CARBS_OBSERVED_COL,
    DEMOGRAPHIC_KNOWN_COLS,
    FEATURE_SCHEMA_VERSION,
    INSULIN_OBSERVED_COL,
    MEAL_EVENT_COL,
    MINS_TO_MEAL_COL,
    PAST_COVARIATE_MASKS,
    SEX_KNOWN_COL,
    TARGET_COL,
    available_past_covariates,
    encode_demographics,
    encode_sex,
    engineer_patient_frame,
)

logger = logging.getLogger(__name__)

STATIC_COVARIATES_FILE = "static_covariates.csv"
PIECEWISE_STATIC_COVARIATES_FILE = "piecewise_static_covariates.parquet"
PATIENT_PARQUET_DIR = "patient_timeseries_parquet"
PATIENT_PARTITION_PREFIX = "patient_id="

# Static covariate candidates the search may select (from static_covariates.csv).
STATIC_COVARIATE_CANDIDATES: tuple[str, ...] = (
    "age",
    "gender",
    "source_file",
    "treatment_group",
    "ethnicity",
    "age_of_diagnosis",
    "is_pregnant",
)

# Piecewise static covariates (from piecewise_static_covariates.parquet). These
# are per-patient device/insulin descriptors; we use the first (dominant) value
# per patient as a static feature.
PIECEWISE_STATIC_CANDIDATES: tuple[str, ...] = (
    "cgm_device",
    "insulin_delivery_device",
    "insulin_delivery_modality",
    "insulin_delivery_algorithm",
    "insulin_type_basal",
    "insulin_type_bolus",
    "height",
    "weight",
)

# Per-patient data-availability flags derived from the timeseries (NOT from the
# static CSV): 1 if the patient's study recorded that channel at all, else 0.
# These give a clean structural-absence signal that survives the model's
# per-window instance norm (unlike an all-zero mask channel, which normalizes to
# a constant). They are coupled to their covariates, not independently selected.
AVAILABILITY_STATIC_CANDIDATES: tuple[str, ...] = ("has_insulin", "has_carbs")

# Deterministic time-based known covariates (computed from the timestamp). Any
# other known covariate is a data column whose future values are read from the
# series (e.g. insulin, when routed to known covariates in the so3 study).
_TIME_KNOWN_COLS: tuple[str, ...] = ("hour_sin", "hour_cos")

# Numeric static candidates (everything else is treated as categorical string).
_NUMERIC_STATIC = {
    "age",
    "age_of_diagnosis",
    "height",
    "weight",
    "has_insulin",
    "has_carbs",
}


@dataclass
class PatientData:
    """Per-patient enriched frame plus its temporal split point."""

    patient_id: str
    frame: pd.DataFrame  # regular-grid, feature-engineered (see features.py)
    split_ts: pd.Timestamp  # last timestamp belonging to the train region
    static_row: Dict[str, object] = field(default_factory=dict)


@dataclass
class ValWindows:
    """Assembled validation windows for a covariate selection."""

    context: object  # TimeSeriesDataFrame (context history per anchor item)
    known_future: Optional[object]  # TimeSeriesDataFrame or None
    true_matrix: np.ndarray  # (n_windows, forecast_length) ground-truth mg/dL
    item_ids: List[str]


class MetabonetData:
    """Load MetaboNet, split temporally, and build panels/windows on demand."""

    def __init__(
        self,
        processed_path: str | Path,
        *,
        forecast_length: int = 24,
        max_context: int = 288,
        min_context: int = 48,
        val_days: int = 2,
        interval_mins: int = 5,
        imputation_threshold_mins: int = 30,
        eval_stride_steps: int = 3,
        max_val_windows_per_patient: int = 8,
        max_val_windows_total: Optional[int] = 2000,
        patient_limit: Optional[int] = None,
        stratify_by: Optional[str] = "source_file",
        source_filter: Optional[Sequence[str]] = None,
        feature_cache_dir: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.processed_path = Path(processed_path)
        self.forecast_length = forecast_length
        self.max_context = max_context
        self.min_context = min_context
        self.val_days = val_days
        self.interval_mins = interval_mins
        self.imputation_threshold_mins = imputation_threshold_mins
        self.eval_stride_steps = eval_stride_steps
        self.max_val_windows_per_patient = max_val_windows_per_patient
        self.max_val_windows_total = max_val_windows_total
        self.patient_limit = patient_limit
        # When set (e.g. "source_file"), the patient subsample is drawn
        # proportionally per group so the subset mirrors the population mix of
        # that static covariate instead of a plain random draw.
        self.stratify_by = stratify_by
        # When set (e.g. ["iobp2"]), restrict to partitions from these source
        # datasets (matched against the source tag in each partition dir name),
        # so a run trains/evaluates on only the target cohort.
        self.source_filter = (
            {str(s).lower() for s in source_filter} if source_filter else None
        )
        # Directory for the persisted per-patient engineered-feature cache
        # (IOB/COB kinetics etc.), so features aren't recomputed per worker/run.
        self._feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.seed = seed
        self.patients: List[PatientData] = []
        self._static_lookup: Dict[str, Dict[str, object]] = {}
        # Covariate-independent caches, built once and reused across trials so a
        # single loaded worker never re-segments/re-concats per trial (the CPU
        # cost that starved the GPUs and whose transient copies stacked to OOM).
        self._train_long_all: Optional[pd.DataFrame] = None
        self._train_item_to_patient: Dict[str, str] = {}
        self._val_base: Optional[Dict[str, object]] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _source_tag(partition_name: str) -> str:
        """Source-dataset tag from a partition dir name.

        ``patient_id=IOBP2-426_iobp2-bae47be84d`` -> ``iobp2`` (the tag sits
        between the final ``_`` and the trailing ``-<hash>``).
        """
        return partition_name.rsplit("_", 1)[-1].split("-", 1)[0].lower()

    def _partition_dirs(self) -> List[Path]:
        parts_root = self.processed_path / PATIENT_PARQUET_DIR
        if not parts_root.exists():
            raise FileNotFoundError(f"Missing patient partition root: {parts_root}")
        dirs = sorted(
            p
            for p in parts_root.iterdir()
            if p.is_dir() and p.name.startswith(PATIENT_PARTITION_PREFIX)
        )
        if self.source_filter:
            dirs = [p for p in dirs if self._source_tag(p.name) in self.source_filter]
            if not dirs:
                raise ValueError(
                    f"source_filter {sorted(self.source_filter)} matched no partitions."
                )
        if self.patient_limit is None or self.patient_limit >= len(dirs):
            return dirs

        rng = random.Random(self.seed)
        if self.stratify_by and self._static_lookup:
            selected = self._stratified_sample(dirs, rng)
        else:
            shuffled = list(dirs)
            rng.shuffle(shuffled)
            selected = shuffled[: self.patient_limit]
        return sorted(selected, key=lambda p: p.name)

    def _stratified_sample(self, dirs: List[Path], rng: random.Random) -> List[Path]:
        """Proportionally sample partitions per ``stratify_by`` group.

        Each group's share of the subset mirrors its share of the population, so
        under-represented source datasets are not dropped entirely (largest-
        remainder allocation guarantees the total equals patient_limit).
        """
        assert self.patient_limit is not None
        stratify_col = str(self.stratify_by)
        groups: Dict[str, List[Path]] = {}
        for path in dirs:
            pid = path.name[len(PATIENT_PARTITION_PREFIX) :]
            key = str(self._static_lookup.get(pid, {}).get(stratify_col, "unknown"))
            groups.setdefault(key, []).append(path)

        total = len(dirs)
        # Largest-remainder allocation of patient_limit across groups.
        raw = {
            key: self.patient_limit * len(members) / total
            for key, members in groups.items()
        }
        alloc = {key: int(value) for key, value in raw.items()}
        remainder = self.patient_limit - sum(alloc.values())
        for key, _ in sorted(
            raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True
        )[:remainder]:
            alloc[key] += 1

        selected: List[Path] = []
        for key, members in groups.items():
            shuffled = list(members)
            rng.shuffle(shuffled)
            selected.extend(shuffled[: min(alloc[key], len(shuffled))])
        return selected

    def _load_static_lookup(self) -> None:
        static_path = self.processed_path / STATIC_COVARIATES_FILE
        if not static_path.exists():
            logger.warning("No static covariates file at %s", static_path)
            self._static_lookup = {}
            return
        static_df = pd.read_csv(static_path, low_memory=False)
        static_df["patient_id"] = static_df["patient_id"].astype(str)
        lookup: Dict[str, Dict[str, object]] = {
            row["patient_id"]: {
                col: row[col]
                for col in STATIC_COVARIATE_CANDIDATES
                if col in static_df.columns
            }
            for _, row in static_df.iterrows()
        }
        self._merge_piecewise_static(lookup)
        self._static_lookup = lookup

    def _merge_piecewise_static(self, lookup: Dict[str, Dict[str, object]]) -> None:
        """Fold per-patient piecewise device/insulin descriptors into lookup.

        Piecewise covariates are time-segmented; we use each patient's first
        (dominant) value per covariate as a static feature.
        """
        pw_path = self.processed_path / PIECEWISE_STATIC_COVARIATES_FILE
        if not pw_path.exists():
            return
        pw = pd.read_parquet(pw_path)
        pw["patient_id"] = pw["patient_id"].astype(str)
        pw = pw[pw["covariate"].isin(PIECEWISE_STATIC_CANDIDATES)]
        pw = pw.sort_values(["patient_id", "covariate", "start_datetime"])
        first = pw.groupby(["patient_id", "covariate"], as_index=False).first()
        for pid, cov, value in zip(
            first["patient_id"], first["covariate"], first["value"]
        ):
            lookup.setdefault(pid, {})[cov] = value

    def all_static_candidates(self) -> List[str]:
        """Static + piecewise static covariate names present in the lookup."""
        return [*STATIC_COVARIATE_CANDIDATES, *PIECEWISE_STATIC_CANDIDATES]

    def _patient_insulin_profile(
        self, patient_id: str
    ) -> tuple[Optional[str], Optional[str], bool]:
        """Return (bolus_brand, basal_brand, basal_long_acting) for a patient.

        Basal is long-acting only for MDI patients (or when the basal insulin
        type names a long-acting analog); on pumps (AID/SAP) basal is rapid
        micro-boluses, so it uses the same rapid kinetics as bolus.
        """
        from .kinetics import is_long_acting

        static_row = self._static_lookup.get(patient_id, {})

        def _str(key: str) -> Optional[str]:
            value = static_row.get(key)
            return value if isinstance(value, str) and value.strip() else None

        bolus_brand = _str("insulin_type_bolus") or _str("insulin_type_basal")
        basal_brand = _str("insulin_type_basal") or _str("insulin_type_bolus")
        modality = (_str("insulin_delivery_modality") or "").strip().lower()
        long_acting = modality == "mdi" or is_long_acting(basal_brand)
        return bolus_brand, basal_brand, long_acting

    def _engineer_or_cache(
        self, patient_id: str, parquet_parts: List[Path]
    ) -> pd.DataFrame:
        """Return the engineered frame, using a per-patient parquet cache.

        The engineered frame (target + all covariates incl. IOB/COB kinetics) is
        expensive to recompute for every worker/run, so it is persisted once and
        reused. The cache key includes the schema version + brand + interval so
        schema/brand/interval changes invalidate it.
        """
        if self._feature_cache_dir is None:
            raw = pd.concat(
                (pd.read_parquet(p) for p in parquet_parts), ignore_index=True
            )
            bolus_brand, basal_brand, long_acting = self._patient_insulin_profile(
                patient_id
            )
            return engineer_patient_frame(
                raw,
                interval_mins=self.interval_mins,
                bolus_brand=bolus_brand,
                basal_brand=basal_brand,
                basal_long_acting=long_acting,
            )

        bolus_brand, basal_brand, long_acting = self._patient_insulin_profile(
            patient_id
        )
        key_brand = (bolus_brand or "default") + ("__la" if long_acting else "")
        safe_brand = "".join(c for c in key_brand.lower() if c.isalnum() or c == "_")[
            :28
        ]
        cache_path = (
            self._feature_cache_dir / f"{patient_id}__v{FEATURE_SCHEMA_VERSION}"
            f"__i{self.interval_mins}__{safe_brand}.parquet"
        )
        if cache_path.exists():
            try:
                frame = pd.read_parquet(cache_path)
                frame.index.name = "timestamp"
                return frame
            except Exception:
                # Partial/corrupt cache (e.g. a concurrent writer); recompute.
                logger.warning("Ignoring unreadable feature cache %s", cache_path)

        raw = pd.concat((pd.read_parquet(p) for p in parquet_parts), ignore_index=True)
        frame = engineer_patient_frame(
            raw,
            interval_mins=self.interval_mins,
            bolus_brand=bolus_brand,
            basal_brand=basal_brand,
            basal_long_acting=long_acting,
        )
        try:
            self._feature_cache_dir.mkdir(parents=True, exist_ok=True)
            # Atomic write: workers may race on the same key, so write a unique
            # temp file then os.replace() (atomic on POSIX) so readers never see
            # a partial parquet.
            tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
            frame.to_parquet(tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception as exc:  # pragma: no cover - cache is best-effort
            logger.warning("Feature cache write failed for %s: %s", patient_id, exc)
        return frame

    def load(self) -> "MetabonetData":
        """Load, engineer, and temporally split every selected patient."""
        self._load_static_lookup()
        min_rows = self.min_context + self.forecast_length
        val_delta = pd.Timedelta(days=self.val_days)

        for part_dir in self._partition_dirs():
            patient_id = part_dir.name[len(PATIENT_PARTITION_PREFIX) :]
            parquet_parts = sorted(part_dir.glob("*.parquet"))
            if not parquet_parts:
                continue
            frame = self._engineer_or_cache(patient_id, parquet_parts)
            if len(frame) < min_rows or frame[TARGET_COL].notna().sum() < min_rows:
                continue

            last_ts = frame.index.max()
            split_ts = last_ts - val_delta
            # Require some train history and a usable validation region.
            if split_ts <= frame.index.min():
                continue
            val_region = frame.loc[frame.index > split_ts]
            if val_region[TARGET_COL].notna().sum() < self.forecast_length:
                continue

            # Per-patient data-availability flags from the observed masks (1 if
            # the study ever recorded that channel). Stored on the static lookup
            # so they can be attached as static covariates (coupled to their
            # covariates in build_*_panel).
            static_row = dict(self._static_lookup.get(patient_id, {}))
            if INSULIN_OBSERVED_COL in frame.columns:
                static_row["has_insulin"] = float(frame[INSULIN_OBSERVED_COL].max() > 0)
            if CARBS_OBSERVED_COL in frame.columns:
                static_row["has_carbs"] = float(frame[CARBS_OBSERVED_COL].max() > 0)
            self._static_lookup[patient_id] = static_row
            # Broadcast per-patient sex as a constant column so it can be routed as
            # a known covariate (Chronos-2 ignores static_features but uses
            # covariates). Harmless if never selected into known_cols.
            frame[SEX_KNOWN_COL] = encode_sex(static_row.get("gender"))
            # Broadcast the other per-patient demographics the same way (age,
            # height, weight, BMI, age-of-diagnosis, Hispanic/Latino). Sourced from
            # the metabonet static lookup (csv + piecewise); missing -> neutral.
            for col, val in encode_demographics(static_row).items():
                frame[col] = val

            self.patients.append(
                PatientData(
                    patient_id=patient_id,
                    frame=frame,
                    split_ts=split_ts,
                    static_row=static_row,
                )
            )

        logger.info("Loaded %d usable patients", len(self.patients))
        if not self.patients:
            raise ValueError(
                "No usable patients after loading/splitting. Check processed_path "
                f"({self.processed_path}) and split settings."
            )
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def resolve_past_covariates(self, requested: Sequence[str]) -> List[str]:
        """Intersect requested past covariates with what the data provides, then
        couple observed-mask companions.

        Masks are never independently selectable: for every selected covariate
        that has one, its ``*_observed`` mask (if present in the frames) is
        appended, so the model never receives an imputed value without knowing it
        was imputed.
        """
        cols = set()
        for p in self.patients:
            cols.update(p.frame.columns)
        resolved = [c for c in available_past_covariates(list(cols)) if c in requested]
        result = list(resolved)
        seen = set(result)
        for col in resolved:
            mask = PAST_COVARIATE_MASKS.get(col)
            if mask and mask in cols and mask not in seen:
                result.append(mask)
                seen.add(mask)
        return result

    def couple_availability_flags(
        self, past_cols: Sequence[str], static_cols: Sequence[str]
    ) -> List[str]:
        """Append has_insulin/has_carbs static flags coupled to selected
        covariates, so a structural-availability signal is always present with
        the covariate it describes (parallel to the per-timestep masks).
        """
        result = list(static_cols)
        seen = set(result)
        uses_insulin = any(
            PAST_COVARIATE_MASKS.get(c) == INSULIN_OBSERVED_COL for c in past_cols
        )
        uses_carbs = any(
            PAST_COVARIATE_MASKS.get(c) == CARBS_OBSERVED_COL for c in past_cols
        )
        for flag, used in (("has_insulin", uses_insulin), ("has_carbs", uses_carbs)):
            if used and flag not in seen:
                result.append(flag)
                seen.add(flag)
        return result

    def _static_features_frame(
        self, item_to_patient: Dict[str, str], static_cols: Sequence[str]
    ) -> Optional[pd.DataFrame]:
        if not static_cols:
            return None
        rows = []
        for item_id, patient_id in item_to_patient.items():
            static_row = self._static_lookup.get(patient_id, {})
            record: Dict[str, object] = {"item_id": item_id}
            for col in static_cols:
                value = static_row.get(col, np.nan)
                record[col] = value
            rows.append(record)
        frame = pd.DataFrame(rows).set_index("item_id")
        # Numeric candidates are coerced to floats; everything else becomes a
        # categorical string (AutoGluon requires categoricals as strings).
        for col in static_cols:
            if col in _NUMERIC_STATIC:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            else:
                frame[col] = frame[col].astype("string").fillna("unknown")
        return frame

    def _interpolate_context_target(self, series: pd.Series) -> pd.Series:
        limit = max(1, self.imputation_threshold_mins // self.interval_mins)
        return series.interpolate(method="linear", limit=limit, limit_area="inside")

    # ------------------------------------------------------------------
    # Training panel
    # ------------------------------------------------------------------
    def _all_past_cols(self) -> List[str]:
        """Past covariate candidates + observed-mask companions in the frames."""
        if not self.patients:
            return []
        frame_cols = list(self.patients[0].frame.columns)
        cols = available_past_covariates(frame_cols)
        # Include observed-mask companions (not in the candidate list) so the
        # panels can carry them; resolve_past_covariates decides per trial which
        # masks are actually attached (coupled to their selected parent).
        masks = [
            m for m in dict.fromkeys(PAST_COVARIATE_MASKS.values()) if m in frame_cols
        ]
        # Retain the known-future meal covariates (meal_event, mins_to_meal) so they
        # can be routed to the known set; they are not past-selectable (mirroring how
        # insulin is retained-then-routed-known via use_known__* toggles).
        meal_known = [c for c in (MEAL_EVENT_COL, MINS_TO_MEAL_COL) if c in frame_cols]
        # Retain the broadcast sex column so it can be routed to the known set.
        sex_known = [SEX_KNOWN_COL] if SEX_KNOWN_COL in frame_cols else []
        # Retain the broadcast demographic columns likewise.
        demo_known = [c for c in DEMOGRAPHIC_KNOWN_COLS if c in frame_cols]
        return [*cols, *masks, *meal_known, *sex_known, *demo_known]

    @staticmethod
    def _add_known_cols(frame: pd.DataFrame, known_cols: Sequence[str]) -> None:
        """Add deterministic known covariates (hour_sin/cos) from the timestamp."""
        if not known_cols:
            return
        timestamps = frame.index.get_level_values("timestamp")
        minutes_of_day = timestamps.hour * 60 + timestamps.minute
        if "hour_sin" in known_cols:
            frame["hour_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
        if "hour_cos" in known_cols:
            frame["hour_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)

    def _ensure_train_long(self) -> None:
        """Build the gap-handled training long-frame once (all past covariates)."""
        if self._train_long_all is not None:
            return
        from ....data.preprocessing.gap_handling import segment_all_patients

        all_past = self._all_past_cols()
        keep_cols = [TARGET_COL, *all_past]
        patient_dict: Dict[str, pd.DataFrame] = {}
        for patient in self.patients:
            train_frame = patient.frame.loc[patient.frame.index <= patient.split_ts]
            train_frame = train_frame[keep_cols]
            if len(train_frame) >= self.min_context + self.forecast_length:
                patient_dict[patient.patient_id] = train_frame

        segments = segment_all_patients(
            patient_dict,
            imputation_threshold_mins=self.imputation_threshold_mins,
            min_segment_length=self.min_context + self.forecast_length,
            bg_col=TARGET_COL,
            expected_interval_mins=self.interval_mins,
        )
        if not segments:
            raise ValueError("Gap handling produced no training segments.")

        frames = []
        item_to_patient: Dict[str, str] = {}
        for seg_id, seg_df in segments.items():
            item_to_patient[seg_id] = seg_id.rsplit("_seg_", 1)[0]
            block = seg_df[keep_cols].copy()
            block["item_id"] = seg_id
            block["timestamp"] = block.index
            frames.append(block)

        self._train_long_all = pd.concat(frames, ignore_index=True).set_index(
            ["item_id", "timestamp"]
        )
        self._train_item_to_patient = item_to_patient

    def build_train_panel(
        self,
        past_cols: Sequence[str],
        *,
        known_cols: Sequence[str] = (),
        static_cols: Sequence[str] = (),
    ):
        """Return an AutoGluon training panel for the requested covariates.

        The gap-handled long-frame (target + all past covariates) is built once
        and cached; each call selects only the requested columns and appends the
        deterministic known covariates — no per-trial re-segmentation or full
        re-concatenation.
        """
        from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
            TimeSeriesDataFrame,
        )

        self._ensure_train_long()
        assert self._train_long_all is not None
        time_known = [c for c in known_cols if c in _TIME_KNOWN_COLS]
        data_known = [c for c in known_cols if c not in _TIME_KNOWN_COLS]
        # Include data-column known covariates (e.g. insulin in so3) from the
        # cached long-frame; deterministic time covariates are derived below.
        cols = [TARGET_COL, *dict.fromkeys([*past_cols, *data_known])]
        long_df = self._train_long_all[cols].copy()
        self._add_known_cols(long_df, time_known)
        static_features = self._static_features_frame(
            self._train_item_to_patient, static_cols
        )
        return TimeSeriesDataFrame(long_df, static_features=static_features)

    # ------------------------------------------------------------------
    # Validation windows
    # ------------------------------------------------------------------
    def _anchor_timestamps(self, patient: PatientData) -> List[pd.Timestamp]:
        frame = patient.frame
        step = pd.Timedelta(minutes=self.interval_mins)
        horizon = self.forecast_length * step
        # Anchors are the last context timestamp; the forecast covers the next
        # forecast_length steps, all of which must lie in the held-out region.
        val_index = frame.index[frame.index > patient.split_ts]
        anchors: List[pd.Timestamp] = []
        for anchor in val_index[:: self.eval_stride_steps]:
            future_end = anchor + horizon
            if future_end > frame.index.max():
                break
            anchors.append(anchor)
            if len(anchors) >= self.max_val_windows_per_patient:
                break
        return anchors

    def _ensure_val_base(self) -> None:
        """Build the covariate-agnostic validation base once (all past covariates)."""
        if self._val_base is not None:
            return
        step = pd.Timedelta(minutes=self.interval_mins)
        all_past = self._all_past_cols()
        context_frames: List[pd.DataFrame] = []
        true_rows: List[np.ndarray] = []
        item_ids: List[str] = []
        item_to_patient: Dict[str, str] = {}
        future_index_by_item: Dict[str, pd.DatetimeIndex] = {}
        future_cov_by_item: Dict[str, pd.DataFrame] = {}

        total = 0
        for patient in self.patients:
            if (
                self.max_val_windows_total is not None
                and total >= self.max_val_windows_total
            ):
                break
            frame = patient.frame
            for k, anchor in enumerate(self._anchor_timestamps(patient)):
                if (
                    self.max_val_windows_total is not None
                    and total >= self.max_val_windows_total
                ):
                    break
                context = frame.loc[frame.index <= anchor].tail(self.max_context)
                if len(context) < self.min_context:
                    continue
                target = self._interpolate_context_target(context[TARGET_COL])
                if target.isna().any():
                    continue  # unrecoverable gap in context

                future_index = pd.date_range(
                    anchor + step, periods=self.forecast_length, freq=step
                )
                future_truth = frame[TARGET_COL].reindex(future_index).to_numpy()
                if not np.isfinite(future_truth).any():
                    continue

                item_id = f"{patient.patient_id}__a{k}"
                item_to_patient[item_id] = patient.patient_id
                future_index_by_item[item_id] = future_index
                # Future values of every past covariate (used when a covariate is
                # routed to a KNOWN future covariate, e.g. insulin in so3).
                future_cov_by_item[item_id] = (
                    frame[all_past].reindex(future_index).ffill().fillna(0.0)
                )

                block = pd.DataFrame(index=context.index)
                block[TARGET_COL] = target.to_numpy()
                for col in all_past:
                    block[col] = context[col].ffill().fillna(0.0).to_numpy()
                block["item_id"] = item_id
                block["timestamp"] = block.index
                context_frames.append(block)

                true_rows.append(future_truth)
                item_ids.append(item_id)
                total += 1

        if not item_ids:
            raise ValueError(
                "No validation windows could be built. Relax min_context / "
                "imputation settings or increase patient_limit."
            )

        context_cols = ["item_id", "timestamp", TARGET_COL, *all_past]
        context_long = pd.concat(context_frames, ignore_index=True)[
            context_cols
        ].set_index(["item_id", "timestamp"])
        self._val_base = {
            "context_long": context_long,
            "true_matrix": np.vstack(true_rows),
            "item_ids": item_ids,
            "item_to_patient": item_to_patient,
            "future_index_by_item": future_index_by_item,
            "future_cov_by_item": future_cov_by_item,
        }

    def build_val_windows(
        self,
        past_cols: Sequence[str],
        *,
        known_cols: Sequence[str] = (),
        static_cols: Sequence[str] = (),
    ) -> ValWindows:
        """Return challenge-shaped validation windows for a covariate selection.

        The covariate-agnostic base (anchors, interpolated context targets, all
        past covariates, ground-truth) is built once and cached; each call
        selects columns and appends known covariates cheaply.
        """
        from autogluon.timeseries import (  # pyright: ignore[reportMissingImports]
            TimeSeriesDataFrame,
        )

        self._ensure_val_base()
        assert self._val_base is not None
        base_context = self._val_base["context_long"]
        item_ids: List[str] = self._val_base["item_ids"]  # type: ignore[assignment]
        item_to_patient: Dict[str, str] = self._val_base["item_to_patient"]  # type: ignore[assignment]
        future_index_by_item = self._val_base["future_index_by_item"]
        future_cov_by_item = self._val_base["future_cov_by_item"]

        time_known = [c for c in known_cols if c in _TIME_KNOWN_COLS]
        data_known = [c for c in known_cols if c not in _TIME_KNOWN_COLS]
        # A known covariate must be present over the context too, so include any
        # data-column known covariates (e.g. insulin in so3) in the context frame.
        context_cols = [TARGET_COL, *dict.fromkeys([*past_cols, *data_known])]
        context_long = base_context[context_cols].copy()  # type: ignore[index]
        self._add_known_cols(context_long, time_known)
        static_features = self._static_features_frame(item_to_patient, static_cols)
        context_tsdf = TimeSeriesDataFrame(
            context_long, static_features=static_features
        )

        known_tsdf = None
        if known_cols:
            known_frames: List[pd.DataFrame] = []
            for item_id in item_ids:
                future_index = future_index_by_item[item_id]  # type: ignore[index]
                kf = pd.DataFrame(index=future_index)
                if time_known:
                    minutes_of_day = future_index.hour * 60 + future_index.minute
                    if "hour_sin" in time_known:
                        kf["hour_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
                    if "hour_cos" in time_known:
                        kf["hour_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)
                if data_known:
                    fut = future_cov_by_item[item_id]  # type: ignore[index]
                    for col in data_known:
                        kf[col] = fut[col].to_numpy()
                kf["item_id"] = item_id
                kf["timestamp"] = future_index
                known_frames.append(kf)
            known_long = pd.concat(known_frames, ignore_index=True).set_index(
                ["item_id", "timestamp"]
            )
            known_tsdf = TimeSeriesDataFrame(known_long)

        return ValWindows(
            context=context_tsdf,
            known_future=known_tsdf,
            true_matrix=self._val_base["true_matrix"],  # type: ignore[arg-type]
            item_ids=item_ids,
        )
