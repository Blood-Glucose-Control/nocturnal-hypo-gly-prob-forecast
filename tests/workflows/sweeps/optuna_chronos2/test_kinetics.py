# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Unit tests for the insulin/carb kinetics feature layer."""

import numpy as np
import pytest

from src.workflows.sweeps.optuna_chronos2.kinetics import (
    DEFAULT_RAPID,
    brand_params,
    convolve_dose_series,
    exponential_kernels,
    hovorka_kernels,
    insulin_kernels,
    wilinska_kernels,
)


def test_brand_params_lookup_and_default():
    assert brand_params("Fiasp") == (55, 300)
    assert brand_params("lyumjev") == (45, 300)
    assert brand_params("afrezza") == (15, 120)
    assert brand_params("unknown-brand") == DEFAULT_RAPID
    assert brand_params(None) == DEFAULT_RAPID


def test_brand_params_fuzzy_free_text():
    # Matches the messy MetaboNet insulin_type values via substring aliases.
    assert brand_params("Fiasp (Rapid Aspart)") == (55, 300)
    assert brand_params("Humalog (Lispro)") == (75, 300)
    assert brand_params("Apidra (Glusine)") == (72, 300)
    assert brand_params("novalog") == (75, 300)
    assert brand_params("NovoLog") == (75, 300)
    # Unrecognized -> generic rapid default.
    assert brand_params("Novolin R") == DEFAULT_RAPID


def test_exponential_kernel_iob_decays_and_activity_integrates():
    iob, activity = exponential_kernels(75, 300, 5)
    iob = np.asarray(iob)
    activity = np.asarray(activity)
    assert iob[0] == pytest.approx(1.0)
    assert iob[-1] == pytest.approx(0.0, abs=1e-2)
    assert np.all(np.diff(iob) <= 1e-9)  # non-increasing
    # activity is the per-interval fraction absorbed (decrement of IOB); it
    # integrates to ~1 unit over the DIA without any extra scaling.
    assert activity.sum() == pytest.approx(1.0, abs=0.05)


def test_activity_equals_iob_decrement():
    # activity[k] must equal iob[k-1]-iob[k] so IOB and activity are on one
    # consistent scale (what left IOB this interval), for all models.
    for iob, activity in (
        exponential_kernels(75, 300, 5),
        hovorka_kernels(5, 300),
        wilinska_kernels(5, 300),
    ):
        iob = np.asarray(iob)
        activity = np.asarray(activity)
        expected = np.clip(-np.diff(iob, prepend=iob[0]), 0.0, None)
        np.testing.assert_allclose(activity, expected, atol=1e-9)


def test_wilinska_dual_pathway_kernel():
    iob, activity = insulin_kernels("wilinska", 5, dia_min=300)
    iob = np.asarray(iob)
    activity = np.asarray(activity)
    assert iob[0] == pytest.approx(1.0)
    assert np.all(np.diff(iob) <= 1e-9)  # non-increasing depot fraction
    # Dual-pathway (fast + slow) absorbs directly from the depot, so activity is
    # front-loaded (peaks at the first interval) unlike the delayed Erlang-2
    # Hovorka / exponential peaks.
    assert int(np.argmax(activity)) == 1
    # Heavier tail than Hovorka: still non-trivial IOB near the DIA edge.
    assert iob[-1] > 0.0


def test_exponential_peak_shifts_with_brand():
    def peak_minute(brand):
        _iob, act = insulin_kernels("exponential", 5, brand=brand)
        return int(np.argmax(act)) * 5

    assert peak_minute("afrezza") < peak_minute("lyumjev") < peak_minute("humalog")


def test_convolution_is_linear_in_dose():
    iob_k, act_k = insulin_kernels("exponential", 5, brand="humalog")
    dose1 = np.zeros(50)
    dose1[0] = 1.0
    dose2 = dose1 * 3.0
    iob1, _ = convolve_dose_series(dose1, iob_k, act_k)
    iob2, _ = convolve_dose_series(dose2, iob_k, act_k)
    assert iob1[0] == pytest.approx(1.0)
    np.testing.assert_allclose(iob2, iob1 * 3.0, rtol=1e-6)


def test_hovorka_kernel_available():
    iob, _act = insulin_kernels("hovorka", 5, dia_min=300)
    assert iob[0] == pytest.approx(1.0, abs=1e-6)


def test_is_long_acting_detection():
    from src.workflows.sweeps.optuna_chronos2.kinetics import is_long_acting

    assert is_long_acting("insulin glargine")
    assert is_long_acting("Lantus")
    assert is_long_acting("Tresiba (degludec)")
    assert is_long_acting("Levemir")
    assert is_long_acting("Long-Acting")
    assert not is_long_acting("Humalog (Lispro)")
    assert not is_long_acting("Fiasp")
    assert not is_long_acting(None)


def test_pump_basal_matches_total_rapid_convolution():
    # For a pump (basal_long_acting=False), modeling basal+bolus separately with
    # the same rapid kernel must equal convolving the summed dose once.
    import pandas as pd

    from src.workflows.sweeps.optuna_chronos2.kinetics import add_insulin_features

    idx = pd.RangeIndex(60)
    basal = pd.Series(0.05, index=idx)
    bolus = pd.Series(0.0, index=idx)
    bolus.iloc[10] = 4.0
    total = basal + bolus

    split = add_insulin_features(
        pd.DataFrame(index=idx),
        basal,
        bolus,
        interval_mins=5,
        bolus_brand="humalog",
        basal_brand="humalog",
        basal_long_acting=False,
    )
    k_iob, k_act = insulin_kernels("exponential", 5, brand="humalog")
    total_iob = np.convolve(total.to_numpy(), k_iob)[: len(idx)]
    np.testing.assert_allclose(split["iob_exp"].to_numpy(), total_iob, atol=1e-9)


def test_mdi_long_acting_basal_iob_persists_longer():
    # A long-acting basal bolus should retain far more IOB at 4h than rapid.
    import pandas as pd

    from src.workflows.sweeps.optuna_chronos2.kinetics import add_insulin_features

    idx = pd.RangeIndex(80)
    basal = pd.Series(0.0, index=idx)
    basal.iloc[0] = 10.0  # a single long-acting basal injection
    bolus = pd.Series(0.0, index=idx)

    rapid = add_insulin_features(
        pd.DataFrame(index=idx),
        basal,
        bolus,
        interval_mins=5,
        bolus_brand="humalog",
        basal_brand="glargine",
        basal_long_acting=False,
    )
    la = add_insulin_features(
        pd.DataFrame(index=idx),
        basal,
        bolus,
        interval_mins=5,
        bolus_brand="humalog",
        basal_brand="glargine",
        basal_long_acting=True,
    )
    # step 48 == 240 min. Long-acting keeps much more on board than rapid.
    assert la["iob_exp"].iloc[48] > rapid["iob_exp"].iloc[48] + 3.0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        insulin_kernels("does-not-exist", 5)
