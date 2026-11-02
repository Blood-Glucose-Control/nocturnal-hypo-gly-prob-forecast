# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Feature engineering for the MetaboNet Chronos-2 Optuna PoC.

The MetaboNet processed cache stores raw per-patient time series (CGM in mg/dL
plus delivery/intake channels: ``basal``, ``bolus``, ``insulin``, ``carbs``).
This module turns a single patient's raw frame into a regular 5-minute grid and
derives a small set of cheap, leakage-free covariate candidates that the Optuna
search can switch on or off:

* Raw dose/intake past covariates: ``basal_insulin``, ``bolus_insulin``,
  ``total_insulin`` (basal + bolus), and ``carbs``.
* Physiological on-board past covariates: IOB + insulin activity and COB + carb
  activity from cached unit-dose kinetics kernels (see ``kinetics.py``).
* Known-future covariates: ``hour_sin`` / ``hour_cos`` (deterministic from the
  timestamp, safe to provide over the forecast horizon).

The CGM target is left with NaN gaps intact; gap handling / interpolation is the
responsibility of the training-segmentation and validation-window builders in
``data.py`` so both stages share one policy.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .kinetics import add_carb_features, add_insulin_features

TARGET_COL = "target"  # CGM (mg/dL) after renaming
RAW_SOURCE_TARGET = "CGM"

# Bump when the engineered column schema changes so persisted per-patient
# feature caches from an older schema are invalidated (see data.py cache key).
FEATURE_SCHEMA_VERSION = 9

# Raw delivery/intake channels in the MetaboNet cache. For a patient/study that
# records a channel, absence at a step genuinely means zero (no dose/meal), so
# these are zero-filled; a companion ``*_observed`` mask (see below) marks which
# steps were actually recorded so the model can tell a true zero from an
# imputed one. Wearable channels (heartrate/steps/calories) are intentionally
# dropped: they are structurally absent in ~10/11 source studies, so carrying
# them just inflates the covariate space and slows training for little signal.
_RAW_DELIVERY_COLS = ("basal", "bolus", "insulin", "carbs")

# Past-only covariate candidates the search may select. All are known at
# context time but NOT provided for the forecast horizon.
PAST_COVARIATE_CANDIDATES: tuple[str, ...] = (
    "total_insulin",
    "basal_insulin",
    "bolus_insulin",
    "carbs",
    # Physiological kinetics (searchable model choice): insulin-on-board /
    # activity from the exponential (brand-specific), Hovorka, and Wilinska
    # dual-pathway models, plus carbs-on-board / activity. Computed via cached
    # unit-dose kernels.
    "iob_exp",
    "insulin_activity_exp",
    "iob_hov",
    "insulin_activity_hov",
    "iob_wil",
    "insulin_activity_wil",
    "cob_fast",
    "carb_activity_fast",
    "cob_med",
    "carb_activity_med",
    "cob_slow",
    "carb_activity_slow",
    # Minutes since the most recent meal announcement (carb-free meal proxy),
    # selectable as a past covariate. Defined for every step (no observed mask).
    "mins_since_meal",
)

# Per-timestep observed-mask companions. A mask is 1 where the underlying raw
# channel was actually recorded and 0 where it was absent/imputed. Masks are
# NOT independently selectable: whenever the search picks a covariate, the panel
# builder auto-attaches its mask (see data.py), so the model never sees an
# imputed value without knowing it was imputed. Insulin-derived features share
# the insulin mask; carb-derived features share the carb mask.
INSULIN_OBSERVED_COL = "insulin_observed"
CARBS_OBSERVED_COL = "carbs_observed"
PAST_COVARIATE_MASKS: dict[str, str] = {
    "total_insulin": INSULIN_OBSERVED_COL,
    "basal_insulin": INSULIN_OBSERVED_COL,
    "bolus_insulin": INSULIN_OBSERVED_COL,
    "iob_exp": INSULIN_OBSERVED_COL,
    "insulin_activity_exp": INSULIN_OBSERVED_COL,
    "iob_hov": INSULIN_OBSERVED_COL,
    "insulin_activity_hov": INSULIN_OBSERVED_COL,
    "iob_wil": INSULIN_OBSERVED_COL,
    "insulin_activity_wil": INSULIN_OBSERVED_COL,
    "carbs": CARBS_OBSERVED_COL,
    "cob_fast": CARBS_OBSERVED_COL,
    "carb_activity_fast": CARBS_OBSERVED_COL,
    "cob_med": CARBS_OBSERVED_COL,
    "carb_activity_med": CARBS_OBSERVED_COL,
    "cob_slow": CARBS_OBSERVED_COL,
    "carb_activity_slow": CARBS_OBSERVED_COL,
}

# Deterministic covariates that CAN be provided for the forecast horizon.
KNOWN_COVARIATE_CANDIDATES: tuple[str, ...] = ("hour_sin", "hour_cos")

# Meal-announcement flag (known-future). The iLet (IOBP2/annual holdout) logs a
# meal-announcement event whose label spans binary ("Meal") in IOBP2-train and a
# 3-way size ("Usual"/"Less than usual"/"More than usual") in the holdout. We
# collapse all meal-indicating labels to a binary point flag (1 at the announced
# step, 0 elsewhere) so it is trainable on IOBP2-train and applicable to the
# holdout. It is a known-future covariate on the same basis as insulin (available
# on the holdout forecast surface) and, unlike bolus, disambiguates a meal-bolus
# (glucose rises) from a correction-bolus (glucose falls).
# Meal covariates. The iLet (IOBP2/annual holdout) logs a meal-announcement event
# whose label is binary ("Meal") in IOBP2-train and a 3-way size in the holdout;
# all meal-indicating labels collapse to a single announcement. We derive three
# time-varying (NOT static) covariates from the announcement grid:
#   * ``meal_event``      -- point flag (1 at an announcement), known-future.
#   * ``mins_since_meal`` -- minutes since the most recent meal (0 at a meal,
#     rising, capped), a PAST covariate (carb-free "meal-on-board" proxy; IOBP2
#     has no carbs so COB=0).
#   * ``mins_to_meal``    -- minutes until the next meal (falling to 0, capped),
#     a KNOWN-future covariate (future meals are on the holdout forecast surface,
#     same basis as insulin). Read at the anchor these give the minutes before /
#     after the forecast start that a meal occurs.
MEAL_LABEL_COL = "meal_label"
MEAL_EVENT_COL = "meal_event"
MINS_SINCE_MEAL_COL = "mins_since_meal"
MINS_TO_MEAL_COL = "mins_to_meal"
MEAL_POSITIVE_LABELS: frozenset[str] = frozenset(
    {"Meal", "Usual", "Less than usual", "More than usual"}
)
# Broadcast (constant-per-patient) known covariate for patient sex/gender. Fed as
# a known-future covariate because Chronos-2 (AutoGluon) consumes covariates but
# NOT static_features; a constant covariate is how per-item metadata reaches it.
SEX_KNOWN_COL = "sex"


def encode_sex(gender: object) -> float:
    """Map a gender/sex label to a numeric known-covariate value.

    Female -> 0.0, Male -> 1.0, unknown/other/missing -> 0.5 (neutral).
    """
    g = str(gender).strip().lower()
    if g in ("female", "f"):
        return 0.0
    if g in ("male", "m"):
        return 1.0
    return 0.5


# Additional broadcast (constant-per-patient) demographic known covariates. Like
# sex, these reach Chronos-2 as constant known-future covariates (static_features
# are ignored). Units are consistent between IOBP2 train (metabonet static +
# piecewise) and the annual holdout parquet: age/age_of_diagnosis in years,
# height in feet, weight in pounds. Continuous channels use a fixed center/scale
# (clipped to +/-3 sigma) so the SAME transform applies to both cohorts; missing
# values map to 0.0 (the cohort center). Ethnicity reduces to Hispanic/Latino
# status (the only label common to both the metabonet and Lynch encodings).
AGE_KNOWN_COL = "age_known"
HEIGHT_KNOWN_COL = "height_known"
WEIGHT_KNOWN_COL = "weight_known"
BMI_KNOWN_COL = "bmi_known"
AGE_DX_KNOWN_COL = "age_dx_known"
HISPANIC_KNOWN_COL = "hispanic_known"

# All broadcast demographic known covariates (excluding sex, which predates this).
DEMOGRAPHIC_KNOWN_COLS: tuple[str, ...] = (
    AGE_KNOWN_COL,
    HEIGHT_KNOWN_COL,
    WEIGHT_KNOWN_COL,
    BMI_KNOWN_COL,
    AGE_DX_KNOWN_COL,
    HISPANIC_KNOWN_COL,
)


def _to_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def _standardize(value: object, center: float, scale: float) -> float:
    """Fixed center/scale standardization; missing -> 0.0 (cohort center)."""
    f = _to_float(value)
    if f is None:
        return 0.0
    z = (f - center) / scale
    return float(max(-3.0, min(3.0, z)))


def encode_age(age: object) -> float:
    """Age (years) -> standardized known covariate (center 40, scale 20)."""
    return _standardize(age, 40.0, 20.0)


def encode_height_ft(height_ft: object) -> float:
    """Height (feet) -> standardized known covariate (center 5.4, scale 0.6)."""
    return _standardize(height_ft, 5.4, 0.6)


def encode_weight_lbs(weight_lbs: object) -> float:
    """Weight (pounds) -> standardized known covariate (center 155, scale 45)."""
    return _standardize(weight_lbs, 155.0, 45.0)


def encode_bmi(height_ft: object, weight_lbs: object) -> float:
    """BMI from height (feet) + weight (pounds) -> standardized (center 27, 6).

    Converts to metric (BMI = kg / m^2); missing height/weight -> 0.0 (center).
    """
    h = _to_float(height_ft)
    w = _to_float(weight_lbs)
    if h is None or w is None or h <= 0:
        return 0.0
    height_m = h * 0.3048
    weight_kg = w * 0.45359237
    bmi = weight_kg / (height_m * height_m)
    return _standardize(bmi, 27.0, 6.0)


def encode_age_of_diagnosis(age_dx: object) -> float:
    """Age of diagnosis (years) -> standardized known covariate (center 22, 14)."""
    return _standardize(age_dx, 22.0, 14.0)


def encode_is_hispanic(ethnicity: object) -> float:
    """Ethnicity label -> Hispanic/Latino indicator (1.0 Hispanic, 0.0 not, 0.5
    unknown/missing). The only ethnicity signal common to both the metabonet
    (race+ethnicity strings) and Lynch ("Hispanic or Latino") encodings.
    """
    if ethnicity is None or (isinstance(ethnicity, float) and ethnicity != ethnicity):
        return 0.5
    label = str(ethnicity).strip().lower()
    if not label or label in ("nan", "unknown/not reported", "unknown"):
        return 0.5
    if "hispanic" in label and "not hispanic" not in label:
        return 1.0
    return 0.0


def encode_demographics(static_row: dict) -> dict:
    """Build the broadcast demographic known-covariate values from a static row.

    Accepts metabonet static-lookup keys (``age``, ``height``, ``weight``,
    ``age_of_diagnosis``, ``ethnicity``); missing keys degrade to the neutral
    cohort center. Returns a mapping of DEMOGRAPHIC_KNOWN_COLS -> float.
    """
    return {
        AGE_KNOWN_COL: encode_age(static_row.get("age")),
        HEIGHT_KNOWN_COL: encode_height_ft(static_row.get("height")),
        WEIGHT_KNOWN_COL: encode_weight_lbs(static_row.get("weight")),
        BMI_KNOWN_COL: encode_bmi(static_row.get("height"), static_row.get("weight")),
        AGE_DX_KNOWN_COL: encode_age_of_diagnosis(static_row.get("age_of_diagnosis")),
        HISPANIC_KNOWN_COL: encode_is_hispanic(static_row.get("ethnicity")),
    }


# Cap the meal-timing ramps so "no nearby meal" maps to a bounded value.
MEAL_TIMING_CAP_MIN = 240  # 4h


def _total_insulin(
    basal: pd.Series, bolus: pd.Series, explicit: pd.Series | None
) -> pd.Series:
    """Total per-step insulin units (basal + bolus), preferring an explicit
    total ``insulin`` channel where the dataset provides one and it is nonzero.
    """
    combined = basal.fillna(0.0) + bolus.fillna(0.0)
    if explicit is None or not explicit.notna().any():
        return combined
    explicit_filled = explicit.fillna(0.0)
    # Use the explicit total where present (>0); otherwise basal + bolus.
    return explicit_filled.where(explicit_filled > 0, combined)


def engineer_patient_frame(
    raw: pd.DataFrame,
    *,
    interval_mins: int = 5,
    time_col: str = "datetime",
    bolus_brand: str | None = None,
    basal_brand: str | None = None,
    basal_long_acting: bool = False,
) -> pd.DataFrame:
    """Return a regular-grid, feature-engineered frame for one patient.

    Args:
        raw: Raw MetaboNet patient parquet (must contain ``time_col`` and
            ``RAW_SOURCE_TARGET``).
        interval_mins: Sampling interval of the regular grid (minutes).
        time_col: Timestamp column name in ``raw``.
        bolus_brand: Rapid-acting bolus insulin type (free text ok) for brand
            kinetics; None uses the rapid default.
        basal_brand: Basal insulin type; used for long-acting kinetics when
            ``basal_long_acting`` is True.
        basal_long_acting: True for MDI patients whose basal is a long-acting
            analog; False for pumps (basal = rapid micro-boluses).

    Returns:
        DataFrame indexed by a regular DatetimeIndex with columns:
        ``target`` plus every entry of ``PAST_COVARIATE_CANDIDATES`` and
        ``KNOWN_COVARIATE_CANDIDATES``, plus the meal covariates ``meal_event``,
        ``mins_since_meal`` (past), and ``mins_to_meal`` (known-future).
    """
    if RAW_SOURCE_TARGET not in raw.columns:
        raise ValueError(
            f"Expected target column '{RAW_SOURCE_TARGET}' not found. "
            f"Available: {list(raw.columns)}"
        )

    df = raw.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).drop_duplicates(subset=time_col, keep="first")
    df = df.set_index(time_col)

    # Reindex onto a strict regular grid spanning the observed range so
    # step-count horizons map to fixed wall-clock offsets.
    freq = f"{interval_mins}min"
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    out = pd.DataFrame(index=df.index)
    out[TARGET_COL] = pd.to_numeric(df[RAW_SOURCE_TARGET], errors="coerce")

    # Raw dose/intake channels. Read the source channels (NaN where a step was
    # not recorded), build per-timestep observed masks BEFORE zero-filling, then
    # expose renamed, unambiguous covariates. Zero-fill is correct for a study
    # that records the channel (absence = no dose/meal); the mask disambiguates
    # a genuine zero from a structurally/intermittently missing reading.
    raw_channels = {}
    observed = {}
    for col in _RAW_DELIVERY_COLS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            observed[col] = series.notna().astype("float32")
            raw_channels[col] = series.fillna(0.0)
        else:
            observed[col] = pd.Series(0.0, index=df.index, dtype="float32")
            raw_channels[col] = pd.Series(0.0, index=df.index)
    basal_insulin = raw_channels["basal"]
    bolus_insulin = raw_channels["bolus"]
    explicit_total = raw_channels["insulin"] if "insulin" in df.columns else None
    total_insulin = _total_insulin(basal_insulin, bolus_insulin, explicit_total)
    carbs = raw_channels["carbs"]

    out["total_insulin"] = total_insulin
    out["basal_insulin"] = basal_insulin
    out["bolus_insulin"] = bolus_insulin
    out["carbs"] = carbs

    # Observed masks: a step counts as insulin-observed if any insulin channel
    # (basal/bolus/explicit) was recorded; carbs use their own channel.
    out[INSULIN_OBSERVED_COL] = (
        (observed["basal"] + observed["bolus"] + observed["insulin"]) > 0
    ).astype("float32")
    out[CARBS_OBSERVED_COL] = observed["carbs"]

    # Physiological kinetics: IOB/insulin-activity (exp + hovorka + wilinska) and
    # COB/carb-activity from cached unit-dose kernels. Bolus + pump basal use
    # rapid brand kinetics; MDI basal uses long-acting kinetics. These decayed
    # on-board signals replace naive rolling dose sums.
    add_insulin_features(
        out,
        basal_insulin,
        bolus_insulin,
        interval_mins=interval_mins,
        bolus_brand=bolus_brand,
        basal_brand=basal_brand,
        basal_long_acting=basal_long_acting,
    )
    add_carb_features(out, carbs, interval_mins=interval_mins)

    # Deterministic time-of-day features (safe as known-future covariates).
    minutes_of_day = out.index.hour * 60 + out.index.minute
    out["hour_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)

    # Meal covariates from the announcement grid (time-varying, not static).
    if MEAL_LABEL_COL in df.columns:
        event = df[MEAL_LABEL_COL].isin(MEAL_POSITIVE_LABELS).to_numpy(dtype="float32")
    else:
        event = np.zeros(len(out), dtype="float32")
    out[MEAL_EVENT_COL] = event

    # mins_since_meal: steps since the last meal at/for each position (rising ramp,
    # 0 at a meal), * interval, capped. mins_to_meal: steps to the next meal
    # (falling ramp, 0 at a meal), * interval, capped. No-meal -> cap.
    n = len(event)
    cap_steps = MEAL_TIMING_CAP_MIN // interval_mins
    idx = np.arange(n)
    is_event = event > 0
    prev_meal = np.where(is_event, idx, -1)
    np.maximum.accumulate(prev_meal, out=prev_meal)
    since_steps = np.where(prev_meal >= 0, idx - prev_meal, cap_steps)
    rev = idx[::-1]
    next_meal_rev = np.where(is_event[::-1], rev, n)
    np.minimum.accumulate(next_meal_rev, out=next_meal_rev)
    next_meal = next_meal_rev[::-1]
    to_steps = np.where(next_meal < n, next_meal - idx, cap_steps)
    out[MINS_SINCE_MEAL_COL] = (
        np.minimum(since_steps, cap_steps).astype("float32") * interval_mins
    )
    out[MINS_TO_MEAL_COL] = (
        np.minimum(to_steps, cap_steps).astype("float32") * interval_mins
    )

    return out


def available_past_covariates(columns: List[str] | tuple[str, ...]) -> List[str]:
    """Filter ``PAST_COVARIATE_CANDIDATES`` to those present in ``columns``."""
    col_set = set(columns)
    return [c for c in PAST_COVARIATE_CANDIDATES if c in col_set]
