# Vendored from https://github.com/vincent-leguen/DILATE (MIT licence)
# Source: loss/dilate_loss.py
# Modifications:
#   - Added a thin normalisation wrapper (dilate_loss_normalized) that accepts
#     (B, horizon) float tensors in original data space and handles the
#     unsqueeze to (B, horizon, 1) that the upstream code expects.
import torch
from . import soft_dtw
from . import path_soft_dtw


def dilate_loss(outputs, targets, alpha, gamma, device):
    """Compute DILATE loss (upstream signature preserved).

    Args:
        outputs: (batch_size, N_output, 1) predicted sequences.
        targets: (batch_size, N_output, 1) ground-truth sequences.
        alpha:   Weight between shape (soft-DTW) and temporal (path-DTW) terms.
                 alpha=1 → pure shape loss; alpha=0 → pure temporal loss.
        gamma:   Soft-DTW smoothing parameter (e.g. 0.01).
        device:  torch device string or object.

    Returns:
        loss, loss_shape, loss_temporal
    """
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0

    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(
            targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1)
        )
        D[k : k + 1, :, :] = Dk

    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    Omega = soft_dtw.pairwise_distances(
        torch.arange(1, N_output + 1, dtype=torch.float32).view(N_output, 1)
    ).to(device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)

    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal


def dilate_loss_normalized(preds, targets, locs, scales, alpha, gamma, device):
    """Normalised DILATE loss for use in TimesFMForTrainer.

    Accepts (B, horizon) float tensors in original data space, applies
    per-window normalisation, then calls dilate_loss.

    Args:
        preds:   (B, horizon) mean predictions.
        targets: (B, horizon) ground-truth values.
        locs:    (B, 1) per-window mean (from context).
        scales:  (B, 1) per-window std (from context, clamped >= 0.1).
        alpha:   Shape vs. temporal weight (see dilate_loss).
        gamma:   Soft-DTW smoothing (see dilate_loss).
        device:  torch device.

    Returns:
        loss scalar (shape + temporal combined).
    """
    pred_norm = (preds - locs) / scales  # (B, horizon)
    target_norm = (targets - locs) / scales  # (B, horizon)

    # DILATE expects (B, horizon, 1)
    pred_3d = pred_norm.unsqueeze(-1)
    target_3d = target_norm.unsqueeze(-1)

    loss, _, _ = dilate_loss(pred_3d, target_3d, alpha, gamma, device)
    return loss
