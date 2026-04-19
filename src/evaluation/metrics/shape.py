# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Shape-aware forecast evaluation metrics (DILATE).

DILATE decomposes time-series forecast quality into two complementary terms:

  - **Shape** (Soft-DTW): How well the predicted trajectory matches the shape
    of the actual trajectory, invariant to small time shifts.
  - **Temporal** (TDI): How well the timing of predicted events aligns with
    the actual event timing (Time Distortion Index).

Combined:  DILATE = alpha * shape + (1 - alpha) * temporal

The gamma parameter controls the softness of the DTW alignment:
  - gamma → 0: hard DTW (exact minimum-cost alignment)
  - gamma large: blurry / over-smoothed alignment

We compute DILATE at three gamma values {0.001, 0.01, 0.1} per episode to
capture sensitivity across scales. Alpha is fixed at 0.5.

The core Soft-DTW and path computation is vendored from the original DILATE
implementation (vincent-leguen/DILATE) and the batched variant
(marcdemers/batch-DILATE), both MIT-licensed.  Only the forward evaluation
path is used (no autograd / backward pass for training).

Reference:
    Le Guen & Thissart, "Shape and Time Distortion Loss for Training Deep
    Time Series Forecasting Models", NeurIPS 2019.

Usage::

    from src.evaluation.metrics.shape import compute_dilate_metrics

    # pred, actual: 1-D numpy arrays of shape (forecast_length,)
    metrics = compute_dilate_metrics(pred, actual)
    # metrics == {
    #     "dilate_g0001": ..., "shape_g0001": ..., "temporal_g0001": ...,
    #     "dilate_g001":  ..., "shape_g001":  ..., "temporal_g001":  ...,
    #     "dilate_g01":   ..., "shape_g01":   ..., "temporal_g01":   ...,
    # }

    # Batch mode — preds/actuals are 2-D arrays of shape (B, forecast_length):
    from src.evaluation.metrics.shape import compute_dilate_metrics_batch
    batch_metrics = compute_dilate_metrics_batch(preds_2d, actuals_2d)
    # Each value is a 1-D array of shape (B,).
"""

from typing import Dict

import numpy as np
from numba import njit

# Gamma values and their column suffixes
_GAMMAS = {
    0.001: "g0001",
    0.01: "g001",
    0.1: "g01",
}

DILATE_COLUMNS = [
    f"{metric}_{suffix}"
    for suffix in _GAMMAS.values()
    for metric in ("dilate", "shape", "temporal")
]
"""All 9 DILATE column names added to per-episode results."""

_ALPHA = 0.5

# ---------------------------------------------------------------------------
# Numba-accelerated Soft-DTW / DILATE kernels
# Adapted from vincent-leguen/DILATE and marcdemers/batch-DILATE (MIT).
# ---------------------------------------------------------------------------


@njit(cache=True)
def _softdtw_forward(D, gamma):
    """Soft-DTW dynamic-programming forward pass.

    Parameters
    ----------
    D : ndarray (N, M)
        Pairwise squared-Euclidean distance matrix.
    gamma : float
        Smoothing parameter (> 0).

    Returns
    -------
    sdtw : float
        Soft-DTW value.
    R : ndarray (N+2, M+2)
        Cost matrix used by the backward / path computation.
    """
    N, M = D.shape
    R = np.full((N + 2, M + 2), 1e8)
    R[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Three-way soft-minimum (log-sum-exp stabilised)
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = -gamma * (np.log(rsum) + rmax)
            R[i, j] = D[i - 1, j - 1] + softmin
    return R[N, M], R


@njit(cache=True)
def _softdtw_path(D, R, gamma):
    """Compute the soft alignment path matrix from Soft-DTW.

    This is the gradient of the Soft-DTW value with respect to the
    distance matrix *D*.  Each entry ``E[i, j]`` gives the (soft)
    probability that time-step *i* in the target aligns with
    time-step *j* in the prediction.

    Parameters
    ----------
    D : ndarray (N, M)
        Same distance matrix passed to :func:`_softdtw_forward`.
    R : ndarray (N+2, M+2)
        Cost matrix returned by :func:`_softdtw_forward`.
        **Modified in-place** (boundary rows/cols are overwritten).
    gamma : float
        Smoothing parameter (must match the forward call).

    Returns
    -------
    E : ndarray (N, M)
        Soft alignment path weights.
    """
    N, M = D.shape

    # Pad D with zeros for boundary indexing
    D_pad = np.zeros((N + 2, M + 2))
    D_pad[1 : N + 1, 1 : M + 1] = D

    E = np.zeros((N + 2, M + 2))
    E[N + 1, M + 1] = 1.0

    # Set boundary conditions on R for the backward sweep
    R[:, M + 1] = -1e8
    R[N + 1, :] = -1e8
    R[N + 1, M + 1] = R[N, M]

    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a = np.exp((R[i + 1, j] - R[i, j] - D_pad[i + 1, j]) / gamma)
            b = np.exp((R[i, j + 1] - R[i, j] - D_pad[i, j + 1]) / gamma)
            c = np.exp((R[i + 1, j + 1] - R[i, j] - D_pad[i + 1, j + 1]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c

    return E[1 : N + 1, 1 : M + 1]


@njit(cache=True)
def _dilate_single(pred, actual, alpha, gamma):
    """Core DILATE computation for one (pred, actual) pair.

    Parameters
    ----------
    pred, actual : 1-D float64 arrays of equal length *N*.
    alpha : float  —  weight in [0, 1] for shape vs temporal.
    gamma : float  —  Soft-DTW smoothing.

    Returns
    -------
    dilate, shape, temporal : float
    """
    N = len(pred)

    # 1. Pairwise squared-Euclidean distance matrix
    D = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            diff = actual[i] - pred[j]
            D[i, j] = diff * diff

    # 2. Soft-DTW (shape loss)
    shape_val, R = _softdtw_forward(D, gamma)

    # 3. Soft alignment path
    E = _softdtw_path(D, R, gamma)

    # 4. Time Distortion Index (temporal loss)
    temporal_val = 0.0
    for i in range(N):
        for j in range(N):
            t_diff = float(i + 1) - float(j + 1)
            temporal_val += E[i, j] * t_diff * t_diff
    temporal_val /= float(N * N)

    dilate_val = alpha * shape_val + (1.0 - alpha) * temporal_val
    return dilate_val, shape_val, temporal_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_dilate_metrics(
    pred: np.ndarray,
    actual: np.ndarray,
    alpha: float = _ALPHA,
) -> Dict[str, float]:
    """Compute DILATE (shape + temporal) at three gamma values.

    Args:
        pred: Point forecast array, shape ``(forecast_length,)``.
        actual: Ground truth array, shape ``(forecast_length,)``.
        alpha: Weight for shape vs temporal. 0.5 = equal weighting.

    Returns:
        Dict with 9 keys: ``{dilate,shape,temporal}_g{0001,001,01}``.
        Returns NaN values if inputs are too short.
    """
    nan_result = {
        f"{m}_{s}": float("nan")
        for s in _GAMMAS.values()
        for m in ("dilate", "shape", "temporal")
    }

    if len(pred) < 2 or len(actual) < 2:
        return nan_result

    pred = np.asarray(pred, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()

    result: Dict[str, float] = {}
    for gamma, suffix in _GAMMAS.items():
        dilate, shape, temporal = _dilate_single(pred, actual, alpha, gamma)
        result[f"dilate_{suffix}"] = float(dilate)
        result[f"shape_{suffix}"] = float(shape)
        result[f"temporal_{suffix}"] = float(temporal)

    return result


def compute_dilate_metrics_batch(
    preds: np.ndarray,
    actuals: np.ndarray,
    alpha: float = _ALPHA,
) -> Dict[str, np.ndarray]:
    """Compute DILATE for a batch of episodes in one call.

    Args:
        preds: Predicted values, shape ``(B, forecast_length)``.
        actuals: Ground truth values, shape ``(B, forecast_length)``.
        alpha: Weight for shape vs temporal. 0.5 = equal weighting.

    Returns:
        Dict with 9 keys (same as :func:`compute_dilate_metrics`), where
        each value is a 1-D float64 array of shape ``(B,)``.
    """
    preds = np.asarray(preds, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    B = preds.shape[0]

    result: Dict[str, np.ndarray] = {}
    for gamma, suffix in _GAMMAS.items():
        d_arr = np.empty(B, dtype=np.float64)
        s_arr = np.empty(B, dtype=np.float64)
        t_arr = np.empty(B, dtype=np.float64)
        for b in range(B):
            d, s, t = _dilate_single(preds[b], actuals[b], alpha, gamma)
            d_arr[b] = d
            s_arr[b] = s
            t_arr[b] = t
        result[f"dilate_{suffix}"] = d_arr
        result[f"shape_{suffix}"] = s_arr
        result[f"temporal_{suffix}"] = t_arr

    return result
