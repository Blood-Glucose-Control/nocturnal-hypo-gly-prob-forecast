#!/usr/bin/env python3
"""GPU smoke test: Chronos2 predict() and predict_batch() with/without quantile_levels.

Tests both zero-shot and fine-tuned inference paths.
Run via SLURM: sbatch --partition=HI --gres=gpu:1 --time=00:30:00 \
    --wrap="source .venvs/chronos2/bin/activate && python scripts/tests/gpu_smoke_test_chronos2_prob.py"
"""

import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.models.chronos2.config import Chronos2Config
from src.models.chronos2.model import Chronos2Forecaster

CHECKPOINT = (
    "trained_models/artifacts/_tsfm_testing/"
    "2026-03-11_00:09_chronos2_holdout_workflow/model.pt"
)
CONTEXT_LEN = 512
FORECAST_LEN = 72  # overridden by checkpoint's forecast_length for fine-tuned tests
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ZS_AVAILABLE = True  # set to False if Chronos2Pipeline can't load

passed = 0
failed = 0
errors = []


def check(name, condition, msg=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        errors.append(f"{name}: {msg}")
        print(f"  FAIL: {name} — {msg}")


def make_fake_data(n_rows=CONTEXT_LEN, n_episodes=1):
    """Create synthetic BG data for testing."""
    dfs = []
    base = pd.Timestamp("2025-01-01")
    for i in range(n_episodes):
        t = pd.date_range(base, periods=n_rows, freq="5min")
        df = pd.DataFrame({
            "datetime": t,
            "bg_mM": 5.0 + np.sin(np.linspace(0, 4 * np.pi, n_rows)) + np.random.randn(n_rows) * 0.2,
        })
        df["episode_id"] = f"ep_{i}"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ===========================================================================
# Zero-shot tests
# ===========================================================================
print("\n=== Zero-shot Chronos2 ===")
zs_config = Chronos2Config(
    model_type="chronos2",
    model_path="autogluon/chronos-bolt-base",
    context_length=CONTEXT_LEN,
    forecast_length=FORECAST_LEN,
)
zs_model = Chronos2Forecaster(zs_config)

# Check if Chronos2Pipeline is available in this environment
try:
    zs_model._ensure_zs_pipeline()
except Exception as e:
    ZS_AVAILABLE = False
    print(f"  SKIP: Chronos2Pipeline not available ({e})")

single_data = make_fake_data(n_rows=CONTEXT_LEN, n_episodes=1)
single_ep = single_data[single_data["episode_id"] == "ep_0"].copy()
batch_data = make_fake_data(n_rows=CONTEXT_LEN, n_episodes=3)

if ZS_AVAILABLE:
    # Test 1: Zero-shot single predict (point)
    print("\n[1] predict() — point forecast")
    try:
        t0 = time.time()
        result = zs_model.predict(single_ep)
        dt = time.time() - t0
        check("returns ndarray", isinstance(result, np.ndarray))
        check("shape is (forecast_len,)", result.shape == (FORECAST_LEN,), f"got {result.shape}")
        check("no NaNs", not np.isnan(result).any())
        check("reasonable BG range", result.min() > 0 and result.max() < 30, f"range [{result.min():.2f}, {result.max():.2f}]")
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"ZS predict point: {e}")
        traceback.print_exc()

    # Test 2: Zero-shot single predict (quantiles)
    print("\n[2] predict(quantile_levels=...) — quantile forecast")
    try:
        t0 = time.time()
        result = zs_model.predict(single_ep, quantile_levels=QUANTILE_LEVELS)
        dt = time.time() - t0
        check("returns ndarray", isinstance(result, np.ndarray))
        check("shape is (n_q, forecast_len)", result.shape == (len(QUANTILE_LEVELS), FORECAST_LEN), f"got {result.shape}")
        check("no NaNs", not np.isnan(result).any())
        check("quantiles are ordered", np.all(np.diff(result, axis=0) >= -0.01), "quantile ordering violated")
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"ZS predict quantiles: {e}")
        traceback.print_exc()

    # Test 3: Zero-shot batch predict (point)
    print("\n[3] predict_batch() — point forecast, 3 episodes")
    try:
        t0 = time.time()
        results = zs_model.predict_batch(batch_data, episode_col="episode_id")
        dt = time.time() - t0
        check("returns dict", isinstance(results, dict))
        check("3 episodes", len(results) == 3, f"got {len(results)}")
        for ep_id, arr in results.items():
            check(f"  {ep_id} shape", arr.shape == (FORECAST_LEN,), f"got {arr.shape}")
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"ZS predict_batch point: {e}")
        traceback.print_exc()

    # Test 4: Zero-shot batch predict (quantiles)
    print("\n[4] predict_batch(quantile_levels=...) — quantile forecast, 3 episodes")
    try:
        t0 = time.time()
        results = zs_model.predict_batch(batch_data, episode_col="episode_id", quantile_levels=QUANTILE_LEVELS)
        dt = time.time() - t0
        check("returns dict", isinstance(results, dict))
        check("3 episodes", len(results) == 3, f"got {len(results)}")
        for ep_id, arr in results.items():
            check(f"  {ep_id} shape", arr.shape == (len(QUANTILE_LEVELS), FORECAST_LEN), f"got {arr.shape}")
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"ZS predict_batch quantiles: {e}")
        traceback.print_exc()
else:
    print("  (skipping zero-shot tests 1-4)")


# ===========================================================================
# Fine-tuned tests
# ===========================================================================
print("\n\n=== Fine-tuned Chronos2 ===")
try:
    ft_model = Chronos2Forecaster.load(CHECKPOINT)
    ft_loaded = True
    print(f"  Loaded checkpoint from {CHECKPOINT}")
    print(f"  config.quantile_levels = {ft_model.config.quantile_levels}")
    print(f"  config.eval_metric = {ft_model.config.eval_metric}")
except Exception as e:
    ft_loaded = False
    failed += 1
    errors.append(f"Failed to load checkpoint: {e}")
    traceback.print_exc()

if ft_loaded:
    ft_flen = ft_model.config.forecast_length
    print(f"  Using checkpoint forecast_length={ft_flen}")
    n_q = len(QUANTILE_LEVELS)

    # Test 5: Fine-tuned single predict (point)
    print("\n[5] predict() — point forecast")
    try:
        t0 = time.time()
        result = ft_model.predict(single_ep)
        dt = time.time() - t0
        check("returns ndarray", isinstance(result, np.ndarray))
        check(f"shape is ({ft_flen},)", result.shape == (ft_flen,), f"got {result.shape}")
        check("no NaNs", not np.isnan(result).any())
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"FT predict point: {e}")
        traceback.print_exc()

    # Test 6: Fine-tuned single predict (quantiles)
    print("\n[6] predict(quantile_levels=...) — quantile forecast")
    try:
        t0 = time.time()
        result = ft_model.predict(single_ep, quantile_levels=QUANTILE_LEVELS)
        dt = time.time() - t0
        check("returns ndarray", isinstance(result, np.ndarray))
        check(f"shape is ({n_q}, {ft_flen})", result.shape == (n_q, ft_flen), f"got {result.shape}")
        check("no NaNs", not np.isnan(result).any())
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"FT predict quantiles: {e}")
        traceback.print_exc()

    # Test 7: Fine-tuned batch predict (point)
    print("\n[7] predict_batch() — point forecast, 3 episodes")
    try:
        t0 = time.time()
        results = ft_model.predict_batch(batch_data, episode_col="episode_id")
        dt = time.time() - t0
        check("returns dict", isinstance(results, dict))
        check("3 episodes", len(results) == 3, f"got {len(results)}")
        for ep_id, arr in results.items():
            check(f"  {ep_id} shape", arr.shape == (ft_flen,), f"got {arr.shape}")
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"FT predict_batch point: {e}")
        traceback.print_exc()

    # Test 8: Fine-tuned batch predict (quantiles)
    print("\n[8] predict_batch(quantile_levels=...) — quantile forecast, 3 episodes")
    try:
        t0 = time.time()
        results = ft_model.predict_batch(batch_data, episode_col="episode_id", quantile_levels=QUANTILE_LEVELS)
        dt = time.time() - t0
        check("returns dict", isinstance(results, dict))
        check("3 episodes", len(results) == 3, f"got {len(results)}")
        for ep_id, arr in results.items():
            check(f"  {ep_id} shape", arr.shape == (n_q, ft_flen), f"got {arr.shape}")
        print(f"  Time: {dt:.2f}s")
    except Exception as e:
        failed += 1
        errors.append(f"FT predict_batch quantiles: {e}")
        traceback.print_exc()


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
if errors:
    print("\nFAILURES:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
