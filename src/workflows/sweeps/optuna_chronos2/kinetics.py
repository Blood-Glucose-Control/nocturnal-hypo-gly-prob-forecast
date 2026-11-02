# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Insulin/carb kinetics feature layer with per-dose curve caching.

Computes insulin-on-board (IOB) + insulin activity and carbs-on-board (COB) +
carb activity as covariate time series. Compartment/kinetic responses are
expensive to recompute per dose, so we exploit the fact that these models are
**linear in dose**: compute the UNIT-dose response kernel once (cached by model
+ parameters + interval), then obtain the full series by convolving the dose
timeline with the kernel. This turns O(doses x ODE) into one cached kernel plus
an FFT-free convolution.

Models
------
* ``exponential`` — the fast peak-time/DIA model used by real-time AID systems
  (Loop/OpenAPS), with **brand-specific** rapid-acting parameters (Fiasp,
  Lyumjev, Humalog/Novolog, Apidra, Afrezza, ...).
* ``hovorka`` — reuses the repository's Hovorka 2004 3-compartment model as a
  cached unit-dose kernel.
* ``wilinska`` — the Wilinska 2005 dual-pathway subcutaneous absorption
  structure (fast + slow channels), implemented as a linear biexponential
  unit-dose kernel using the paper's median rates. The paper's saturable
  Michaelis-Menten local degradation is intentionally omitted: it is nonlinear
  in dose (breaking the linear kernel/convolution framework) and would conflate
  locally-degraded insulin with insulin that becomes active. Ref: Wilinska ME
  et al., IEEE TBME 52(1):3-12, 2005 (Model 11); DOI 10.1109/TBME.2004.839639.
"""

from __future__ import annotations

import functools
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Brand-specific rapid-acting insulin parameters: (peak_minutes, DIA_minutes).
# Peaks follow standard exponential-model values used in AID algorithms.
INSULIN_BRAND_PARAMS: Dict[str, Tuple[float, float]] = {
    "humalog": (75, 300),
    "novolog": (75, 300),
    "apidra": (72, 300),
    "fiasp": (55, 300),
    "lyumjev": (45, 300),
    "afrezza": (15, 120),
    # Long-acting (basal) analogs — used only for MDI basal insulin. Peakless
    # agents (glargine/degludec) are approximated with a late, broad peak; the
    # exponential model requires peak < DIA/2. Values are clinical approximations
    # (onset/peak/duration) for glargine ~24h, degludec ~42h, detemir/NPH shorter.
    "glargine": (600, 1440),  # Lantus/Basaglar/Toujeo: ~flat, ~24h
    "degludec": (1000, 2520),  # Tresiba: ~flat, ~42h
    "detemir": (400, 1200),  # Levemir: mild peak ~6-8h, ~20h
    "nph": (300, 960),  # NPH: peak ~5h, ~16h
    "long_acting": (600, 1440),  # generic long-acting -> glargine-like
}
DEFAULT_RAPID: Tuple[float, float] = (75, 300)

# Canonical brand keys that are long-acting basal insulins (not rapid).
LONG_ACTING_BRANDS: frozenset[str] = frozenset(
    {"glargine", "degludec", "detemir", "nph", "long_acting"}
)

# Wilinska 2005 Model 11 dual-pathway SC absorption (median values, Table I).
# ka for the slow/fast channels and the fraction routed through the slow
# channel. The saturable local degradation (Vmax/kM) is omitted on purpose
# (nonlinear in dose; see module docstring).
WILINSKA_KA_SLOW: float = 0.011  # slow-channel absorption rate (min^-1)
WILINSKA_KA_FAST: float = 0.021  # fast-channel absorption rate (min^-1)
WILINSKA_SLOW_FRACTION: float = 0.67  # fraction of a dose via the slow channel

# Substring aliases -> canonical brand key, for matching the messy free-text
# insulin_type values in the MetaboNet piecewise covariates (e.g.
# "Fiasp (Rapid Aspart)", "Humalog (Lispro)", "novalog", "NovoLog").
# Order matters: more specific brands are checked before generic aspart/lispro.
_BRAND_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("fiasp", "fiasp"),
    ("faster aspart", "fiasp"),
    ("lyumjev", "lyumjev"),
    ("lispro-aabc", "lyumjev"),
    ("afrezza", "afrezza"),
    ("apidra", "apidra"),
    ("glulisine", "apidra"),
    ("glusine", "apidra"),
    ("humalog", "humalog"),
    ("lispro", "humalog"),
    ("novolog", "novolog"),
    ("novalog", "novolog"),
    ("novorapid", "novolog"),
    ("aspart", "novolog"),
    # Long-acting basal analogs.
    ("glargine", "glargine"),
    ("lantus", "glargine"),
    ("basaglar", "glargine"),
    ("toujeo", "glargine"),
    ("degludec", "degludec"),
    ("tresiba", "degludec"),
    ("detemir", "detemir"),
    ("levemir", "detemir"),
    ("nph", "nph"),
    ("humulin n", "nph"),
    ("novolin n", "nph"),
    ("long-acting", "long_acting"),
    ("long acting", "long_acting"),
)


def normalize_brand(raw: str | None) -> str | None:
    """Map a free-text insulin type to a canonical brand key, or None."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    for alias, canonical in _BRAND_ALIASES:
        if alias in text:
            return canonical
    return None


def is_long_acting(insulin_type: str | None) -> bool:
    """True if a free-text insulin type names a long-acting (basal) analog."""
    return normalize_brand(insulin_type) in LONG_ACTING_BRANDS


def brand_params(brand: str | None) -> Tuple[float, float]:
    """Look up (peak, DIA) for an insulin brand/type, falling back to default.

    Accepts messy free-text (e.g. "Fiasp (Rapid Aspart)") via substring
    normalization; unknown/None values return the generic rapid-acting default.
    """
    canonical = normalize_brand(brand)
    if canonical is None:
        return DEFAULT_RAPID
    return INSULIN_BRAND_PARAMS.get(canonical, DEFAULT_RAPID)


def _activity_from_iob(iob: np.ndarray) -> np.ndarray:
    """Per-interval insulin/carb *activity* = the amount that left IOB/COB.

    Defined as the discrete decrement of the on-board kernel, so ``activity[k]``
    is exactly ``iob[k-1] - iob[k]`` (units that became active during interval
    ``k``). ``activity[0]`` is 0 (nothing has been absorbed at the dose instant)
    and the series sums to ``iob[0] - iob[-1] ~= 1`` over the DIA. This keeps IOB
    and activity on one consistent, interpretable scale for every model and
    guarantees the mass balance ``iob[t] = iob[t-1] - activity[t] + dose[t]``.
    """
    activity = -np.diff(iob, prepend=iob[0])
    return np.clip(activity, 0.0, None)


@functools.lru_cache(maxsize=512)
def exponential_kernels(
    peak_min: float, dia_min: float, interval_mins: int
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Unit-dose (iob_fraction, activity) kernels for the exponential model.

    Cached; returns tuples so the lru_cache result stays immutable. ``iob`` is
    the fraction of a unit dose remaining; ``activity`` is the fraction absorbed
    in each interval (the per-step decrement of ``iob``), which sums to ~1 over
    the DIA.
    """
    if not (0 < peak_min < dia_min / 2):
        # Model is only defined for peak < DIA/2; clamp to a safe peak.
        peak_min = min(peak_min, dia_min / 2 - interval_mins)
    n = int(round(dia_min / interval_mins)) + 1
    t = np.arange(n, dtype=float) * interval_mins
    tp, td = float(peak_min), float(dia_min)
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    s = 1 / (1 - a + (1 + a) * np.exp(-td / tau))

    iob = 1 - s * (1 - a) * (
        (t**2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1
    )
    iob = np.clip(iob, 0.0, 1.0)
    iob[0] = 1.0
    activity = _activity_from_iob(iob)
    return tuple(iob.tolist()), tuple(activity.tolist())


@functools.lru_cache(maxsize=64)
def hovorka_kernels(
    interval_mins: int, dia_min: int
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Unit-dose (iob_fraction, activity) kernels from the Hovorka 2004 model.

    ``activity`` is derived as the per-step decrement of the Hovorka IOB curve so
    it is on the same "units that left IOB this interval" scale as the
    exponential model (rather than the model's raw plasma-availability signal,
    which integrates to a different constant).
    """
    from ....data.physiological.insulin_model.insulin_model import (
        calculate_insulin_availability_and_iob_single_delivery,
    )

    _ins_availability, iob, _q1, _q2, _ip = (
        calculate_insulin_availability_and_iob_single_delivery(
            1.0, interval_mins, dia_min
        )
    )
    iob = np.clip(np.asarray(iob, dtype=float), 0.0, 1.0)
    activity = _activity_from_iob(iob)
    return tuple(iob.tolist()), tuple(activity.tolist())


@functools.lru_cache(maxsize=64)
def wilinska_kernels(
    interval_mins: int, dia_min: int = 300
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Unit-dose (iob_fraction, activity) kernels for the Wilinska 2005 model.

    Uses the dual-pathway (fast + slow) absorption structure as a linear
    biexponential: a unit dose splits ``WILINSKA_SLOW_FRACTION`` into the slow
    channel (rate ``WILINSKA_KA_SLOW``) and the rest into the fast channel
    (rate ``WILINSKA_KA_FAST``). IOB is the depot fraction still to be absorbed;
    ``activity`` is its per-step decrement (units becoming active this interval).
    The saturable local degradation from the paper is omitted (see module
    docstring) so the kernel stays linear in dose.
    """
    n = int(round(dia_min / interval_mins)) + 1
    t = np.arange(n, dtype=float) * interval_mins
    f = WILINSKA_SLOW_FRACTION
    iob = f * np.exp(-WILINSKA_KA_SLOW * t) + (1 - f) * np.exp(-WILINSKA_KA_FAST * t)
    iob = np.clip(iob, 0.0, 1.0)
    iob[0] = 1.0
    activity = _activity_from_iob(iob)
    return tuple(iob.tolist()), tuple(activity.tolist())


def insulin_kernels(
    model: str, interval_mins: int, *, brand: str | None = None, dia_min: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (iob_kernel, activity_kernel) for the requested kinetics model."""
    model = model.lower()
    if model == "exponential":
        peak, dia = brand_params(brand)
        iob, act = exponential_kernels(peak, dia, interval_mins)
        return np.asarray(iob), np.asarray(act)
    if model == "hovorka":
        iob, act = hovorka_kernels(interval_mins, dia_min)
        return np.asarray(iob), np.asarray(act)
    if model == "wilinska":
        iob, act = wilinska_kernels(interval_mins, dia_min)
        return np.asarray(iob), np.asarray(act)
    raise ValueError(f"Unknown insulin kinetics model {model!r}.")


def convolve_dose_series(
    dose_series, iob_kernel: np.ndarray, activity_kernel: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve a dose timeline with unit-dose kernels -> (iob, activity)."""
    dose = np.nan_to_num(np.asarray(dose_series, dtype=float))
    n = dose.shape[0]
    iob = np.convolve(dose, iob_kernel)[:n]
    activity = np.convolve(dose, activity_kernel)[:n]
    return iob, activity


def add_insulin_features(
    frame: pd.DataFrame,
    basal: pd.Series,
    bolus: pd.Series,
    *,
    interval_mins: int,
    bolus_brand: str | None = None,
    basal_brand: str | None = None,
    basal_long_acting: bool = False,
) -> pd.DataFrame:
    """Add total IOB + insulin-activity columns for ALL kinetics models.

    Insulin on board is modeled per delivery component and summed, so basal and
    bolus can carry different kinetics:

    * ``bolus`` is always rapid-acting (``bolus_brand``), modeled with each of
      the exponential / Hovorka / Wilinska kernels.
    * ``basal`` is rapid-acting on pumps (AID/SAP micro-boluses use the same
      rapid insulin as bolus) and long-acting on MDI. When ``basal_long_acting``
      is True the basal component uses a long-acting exponential kernel
      (``basal_brand`` -> glargine/detemir/degludec/NPH peak+DIA); otherwise it
      uses the same rapid kernel as bolus, which makes the total identical to
      convolving (basal + bolus) with one rapid kernel.

    Emits ``iob_{exp,hov,wil}`` / ``insulin_activity_{exp,hov,wil}`` so the
    kinetics model stays a searchable covariate.
    """
    la_iob = la_act = None
    if basal_long_acting:
        # MDI basal is long-acting by definition; if the recorded basal type is
        # missing or not a long-acting name, default to a glargine-like profile
        # rather than silently falling back to rapid kinetics.
        la_brand = basal_brand if is_long_acting(basal_brand) else "long_acting"
        la_k = insulin_kernels("exponential", interval_mins, brand=la_brand)
        la_iob, la_act = convolve_dose_series(basal, la_k[0], la_k[1])

    for model, suffix in (
        ("exponential", "exp"),
        ("hovorka", "hov"),
        ("wilinska", "wil"),
    ):
        iob_k, act_k = insulin_kernels(model, interval_mins, brand=bolus_brand)
        bolus_iob, bolus_act = convolve_dose_series(bolus, iob_k, act_k)
        if basal_long_acting:
            basal_iob, basal_act = la_iob, la_act
        else:
            basal_iob, basal_act = convolve_dose_series(basal, iob_k, act_k)
        frame[f"iob_{suffix}"] = bolus_iob + basal_iob
        frame[f"insulin_activity_{suffix}"] = bolus_act + basal_act
    return frame


# Carb absorption variants (peak, duration mins) for the exponential model,
# exposed as selectable covariates so the search can pick the best profile
# without per-trial recompute (keeps the pre-built feature cache valid).
CARB_ABSORPTION_VARIANTS: Dict[str, Tuple[float, float]] = {
    "fast": (30, 120),  # simple/high-GI carbs
    "med": (45, 180),  # mixed meal (previous default)
    "slow": (75, 300),  # complex/high-fat/protein meal
}


def add_carb_features(
    frame: pd.DataFrame,
    carbs: pd.Series,
    *,
    interval_mins: int,
) -> pd.DataFrame:
    """Add COB + carb-activity columns for each carb-absorption variant.

    Emits ``cob_{fast,med,slow}`` / ``carb_activity_{fast,med,slow}`` from the
    exponential absorption model at fast/medium/slow peak+duration presets, so
    the search can select the best carb kinetics (a categorical covariate
    choice) while the feature cache stays valid.
    """
    for name, (peak_min, dur_min) in CARB_ABSORPTION_VARIANTS.items():
        iob_k, act_k = exponential_kernels(peak_min, dur_min, interval_mins)
        cob, activity = convolve_dose_series(
            carbs, np.asarray(iob_k), np.asarray(act_k)
        )
        frame[f"cob_{name}"] = cob
        frame[f"carb_activity_{name}"] = activity
    return frame
