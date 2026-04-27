#!/usr/bin/env python3
"""Smoke test: verify every TimesFM loss_fn variant runs forward+backward correctly.

For each supported loss_fn value, creates a small synthetic batch, runs a
forward pass through TimesFMForTrainer, asserts the loss is a finite scalar,
then calls loss.backward() and asserts no NaN gradients appear.

This catches:
  - Import / path errors in DILATE utilities
  - Wrong tensor shapes / device mismatches in each loss branch
  - Numba JIT errors in soft_dtw on first execution
  - NaN-producing code paths (e.g. division by zero in normalisation)

Run from project root (timesfm venv):
  source .venvs/timesfm/bin/activate
  python scripts/scratch/timesfm_loss_fn_smoketest.py

The script exits non-zero if any check fails.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.timesfm.model import TimesFMForTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_HF_DIR = (
    PROJECT_ROOT
    / "trained_models/artifacts/timesfm"
    / "2026-04-22_10:11_RID20260422_101119_gpu1_625_holdout_workflow"
    / "model.pt"
    / "hf_model"
)

# Small batch to keep the test fast.  context_length just needs to be a valid
# positive integer; horizon must be <= 128 (TimesFM's internal output length).
BATCH_SIZE = 4
CONTEXT_LEN = 128
HORIZON = 32

# Use bfloat16 to match actual training dtype — catches numeric issues that
# float32 would silently mask.
TORCH_DTYPE = torch.bfloat16

# All six supported loss_fn values.
# fmt: off
LOSS_FN_CASES = [
    "mse",
    "pinball",
    "joint",
    "dilate",              # DILATE on mean trajectory; triggers numba JIT on first call
    "dilate_pinball_median",  # pinball + DILATE on median (0.5) trajectory
    "dilate_pinball",      # pinball + DILATE on all 9 quantile trajectories (~9× slower)
]
# fmt: on

# DILATE hyperparameters — mirrors the defaults used in configs 03 / 04.
DILATE_ALPHA = 0.5
DILATE_GAMMA = 0.01
DILATE_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0
_errors: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"    PASS  {label}")
    else:
        _failed += 1
        msg = f"{label}: {detail}" if detail else label
        _errors.append(msg)
        print(f"    FAIL  {label}" + (f" — {detail}" if detail else ""))


def make_batch(device: str):
    """Return a small synthetic BG-like batch.

    past_values : list of B CPU float32 tensors, each shape (CONTEXT_LEN,)
    past_values_padding : zeros tensor (B, CONTEXT_LEN) long  — all observed
    freq        : zeros tensor (B,) long                      — freq_type=0 (5-min)
    future_values : float32 tensor (B, HORIZON) on `device`
    """
    rng = np.random.default_rng(42)
    past_values = []
    for _ in range(BATCH_SIZE):
        ctx = 6.0 + np.cumsum(rng.normal(0, 0.04, CONTEXT_LEN)).astype(np.float32)
        ctx = np.clip(ctx, 2.5, 22.0)
        past_values.append(torch.tensor(ctx, dtype=torch.float32))  # stays on CPU

    past_values_padding = torch.zeros(BATCH_SIZE, CONTEXT_LEN, dtype=torch.long)
    freq = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)
    future_values = torch.tensor(
        rng.uniform(4.0, 9.0, (BATCH_SIZE, HORIZON)).astype(np.float32)
    ).to(device)

    # HF Trainer's _prepare_input walks lists and moves each tensor to the
    # device.  When calling the wrapper directly (as in this test), we need
    # to replicate that move manually.
    past_values = [pv.to(device) for pv in past_values]

    return past_values, past_values_padding, freq, future_values


# ---------------------------------------------------------------------------
# Load the HF model once — shared across all loss_fn tests
# ---------------------------------------------------------------------------

print("=" * 64)
print("  TimesFM loss_fn forward+backward smoke test")
print("=" * 64)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device  : {device}")
print(f"  Model   : {MODEL_HF_DIR}")
print()

try:
    from transformers import TimesFmModelForPrediction

    hf_model = TimesFmModelForPrediction.from_pretrained(
        str(MODEL_HF_DIR),
        torch_dtype=TORCH_DTYPE,
        attn_implementation="sdpa",
    )
    hf_model.to(device)
    n_params = sum(p.numel() for p in hf_model.parameters()) / 1e6
    print(f"  Loaded  : {n_params:.0f}M params  dtype={TORCH_DTYPE}")
except Exception as exc:
    print(f"FATAL: could not load model — {exc}")
    sys.exit(1)

print()

# ---------------------------------------------------------------------------
# Test each loss_fn
# ---------------------------------------------------------------------------

for loss_fn in LOSS_FN_CASES:
    print(f"  --- loss_fn={loss_fn!r} ---")
    t0 = time.perf_counter()

    if "dilate" in loss_fn:
        print("    (DILATE: first call may be slow — numba JIT compiling)")

    # Fresh wrapper; model weights are shared.
    wrapper = TimesFMForTrainer(
        hf_model,
        loss_fn=loss_fn,
        dilate_alpha=DILATE_ALPHA,
        dilate_gamma=DILATE_GAMMA,
        dilate_weight=DILATE_WEIGHT,
    )
    wrapper.train()

    past_values, past_values_padding, freq, future_values = make_batch(device)

    # --- forward ---
    # HF Trainer uses torch.autocast(dtype=bfloat16) when bf16=True in
    # TrainingArguments.  Replicate that here so the float32 DataLoader
    # tensors are upcast transparently, matching actual training conditions.
    out = None
    loss = None
    amp_ctx = (
        torch.autocast("cuda", dtype=TORCH_DTYPE)
        if device == "cuda"
        else torch.autocast("cpu", dtype=torch.bfloat16)
    )
    try:
        with amp_ctx:
            out = wrapper(
                past_values=past_values,
                past_values_padding=past_values_padding,
                freq=freq,
                future_values=future_values,
            )
        loss = out["loss"]
    except Exception as exc:
        check(f"{loss_fn}: forward pass", False, repr(exc))
        print()
        continue

    check(f"{loss_fn}: forward pass runs", True)

    loss_val = float(loss.detach().cpu())
    check(
        f"{loss_fn}: loss is finite scalar",
        math.isfinite(loss_val),
        f"got {loss_val!r}",
    )

    if not math.isfinite(loss_val):
        # No point running backward if loss is already bad.
        print()
        continue

    # --- backward ---
    for p in hf_model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    backward_ok = False
    try:
        loss.backward()
        backward_ok = True
    except Exception as exc:
        check(f"{loss_fn}: backward pass", False, repr(exc))
        print()
        continue

    check(f"{loss_fn}: backward pass runs", backward_ok)

    # --- gradient sanity ---
    # Only inspect params that were actually reached by the backward graph.
    grad_params = [p for p in hf_model.parameters() if p.grad is not None]
    has_nan = any(torch.isnan(p.grad).any() for p in grad_params)
    check(
        f"{loss_fn}: no NaN gradients ({len(grad_params)} param tensors checked)",
        not has_nan,
        "NaN detected in at least one .grad tensor",
    )

    elapsed = time.perf_counter() - t0
    print(f"    loss={loss_val:.5f}  elapsed={elapsed:.2f}s")
    print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 64)
print(
    f"  Results: {_passed} passed, {_failed} failed  "
    f"(out of {_passed + _failed} checks across {len(LOSS_FN_CASES)} loss functions)"
)
if _errors:
    print()
    print("  Failures:")
    for e in _errors:
        print(f"    - {e}")
print("=" * 64)

sys.exit(0 if _failed == 0 else 1)
