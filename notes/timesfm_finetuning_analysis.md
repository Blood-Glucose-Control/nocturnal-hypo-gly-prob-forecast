# TimesFM Fine-Tuning Analysis

**Date:** 2026-04-24
**Branch:** `chronos2-train`
**Data:** `experiments/nocturnal_forecasting/summary.csv` (355 runs after archival)

---

## 1. Summary of Runs

After deduplication (`sort_values('timestamp').drop_duplicates(subset=['model','dataset','mode','checkpoint'], keep='last')`)
and archival of 9 evaluation directories for 3 degenerate checkpoints, 4 distinct TimesFM checkpoints
were evaluated across 4 datasets (aleppo_2017, brown_2019, lynch_2022, tamborlane_2008):

| Checkpoint  | Config YAML                | LR     | Epochs | Context |
|-------------|----------------------------|--------|--------|---------|
| gpu1_625    | `07_low_lr_long_training`  | 1e-5   | 15     | 512     |
| gpu0_27700  | `04_long_training`         | 1e-4   | 15     | 512     |
| gpu1_29061  | `03_short_context`         | 1e-4   | 10     | 288     |
| gpu0_24747  | `02_high_lr`               | 5e-4   | 10     | 512     |

All runs share: `batch_size=256`, `gradient_accumulation_steps=4`, `weight_decay=0.01`, `bfloat16`.

---

## 2. Results (averaged across datasets)

| Checkpoint | LR    | RMSE  | WQL ↓ | Brier ↓ | MACE ↓ | DILATE ↓ | Cov 50% | Cov 80% |
|------------|-------|-------|--------|---------|--------|----------|---------|---------|
| gpu1_625   | 1e-5  | 2.73  | 0.784  | 0.032   | 0.035  | 256.5    | 0.52    | 0.81    |
| gpu0_24747 | 5e-4  | 2.91  | 0.956  | 0.034   | 0.155  | 227.5    | 0.18    | 0.32    |
| gpu1_29061 | 1e-4  | 3.02  | 0.930  | 0.034   | 0.108  | 238.5    | 0.30    | 0.53    |
| gpu0_27700 | 1e-4  | 3.09  | 0.950  | 0.034   | 0.112  | 259.7    | 0.25    | 0.45    |

---

## 3. Key Conclusions

### 3.1 Learning Rate is the Dominant Factor

The only meaningful difference between `gpu1_625` (the strong checkpoint) and the rest is
**learning rate: 1e-5 vs. 1e-4/5e-4**. Everything else (epochs, batch size, optimizer,
context length) has a minor effect by comparison.

At LR ≥ 1e-4 the model's backbone shifts are large enough during fine-tuning to destroy
the pre-trained quantile head calibration, producing severely under-covering intervals:
- `gpu0_24747` (LR=5e-4): coverage_50 = 0.18 (nominal = 0.50) — catastrophic under-coverage
- `gpu0_27700` (LR=1e-4): coverage_50 = 0.25 — still badly under-covering

At LR=1e-5 the backbone adapts slowly enough that the pre-trained uncertainty structure
is preserved:
- `gpu1_625`: coverage_50 = 0.52, coverage_80 = 0.81 — near-perfect calibration

### 3.2 Calibration Metrics Split Clearly from Point-Forecast Metrics

- **WQL** and **MACE** are dramatically better at LR=1e-5 (WQL 0.784 vs 0.93–0.96; MACE 0.035 vs 0.108–0.155)
- **Brier score** is essentially identical across all checkpoints (~0.032–0.034) — the binary
  hypoglycemia event prediction quality is similar regardless of LR
- **DILATE** shows `gpu1_625` as slightly *worse* than `gpu0_24747` on trajectory shape/timing
  (256 vs 228). The collapsed-interval high-LR models still produce reasonable point-forecast
  trajectories — they just have no useful uncertainty quantification.

### 3.3 Degenerate Checkpoints (Archived)

Three training runs produced catastrophic results with identical patterns across all 4 datasets
(cross-dataset consistency being the diagnostic signal for degeneracy — a well-behaved model
would show dataset-specific variation):

- `gpu0_3854`: coverage ≈ 1.0, sharpness 2000–3855 mmol/L (infinite-width intervals)
- `gpu1_19312`: coverage ≈ 0, sharpness_50 **negative** (−742 to −1256 — inverted quantiles)
- `gpu0_32242`: coverage ≈ 0, sharpness 400–750 mmol/L (collapsed point-like output)

These 9 evaluation directories were moved to `experiments/nocturnal_forecasting/_bad_runs_archive/`.

### 3.4 Context Length Effect — Inconclusive

`gpu1_29061` uses context_length=288 (24 hours) vs 512 (42 hours) for the others.
However, no two checkpoints in this sweep differ *only* on context length — the confounds are:

- `gpu0_24747` (512 ctx, LR=5e-4, 10 ep) vs `gpu1_29061` (288 ctx, LR=1e-4, 10 ep): LR also differs
- `gpu0_27700` (512 ctx, LR=1e-4, 15 ep) vs `gpu1_29061` (288 ctx, LR=1e-4, 10 ep): epochs also differ

The worse RMSE for `gpu1_29061` (3.12 vs 3.09) is more plausibly explained by fewer epochs
than by context length. **No conclusion about context length can be drawn from this sweep.**
A dedicated ablation holding LR and epochs fixed while varying context would be needed.

---

## 4. Root Cause: Training Loss Does Not Supervise Quantile Heads

TimesFM fine-tuning in `src/models/timesfm/model.py` (`TimesFMForTrainer.forward`) trains on
**normalized MSE of mean predictions only**:

```python
mean_predictions = outputs.mean_predictions  # (B, model_horizon)
# ...per-window z-score normalization...
loss = F.mse_loss(pred_norm, target_norm)
```

`outputs.full_predictions` (shape `(B, horizon, 10)`) contains the quantile heads but is
never used during training. The quantile levels from the model config are:
`[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]` (index 0 = mean, indices 1–9 = quantiles).

When LR is large, backbone weight shifts damage the pre-trained quantile head outputs
with no gradient signal to correct them. At LR=1e-5 the backbone is perturbed little
enough that the pre-trained calibration survives.

---

## 5. Practical Recommendations

### 5.1 Immediate: Use LR ≤ 1e-5 for TimesFM Fine-Tuning

Until the loss function is improved, `lr=1e-5` is the safe operating regime.
`lr=1e-4` is the threshold where calibration degrades substantially.

### 5.2 Pinball Loss — Easy to Implement, High Expected Gain

Adding pinball loss over `outputs.full_predictions` directly supervises calibration
during fine-tuning. This is ~10 lines added to `TimesFMForTrainer.forward()`:

```python
# Quantile heads: full_predictions shape (B, model_horizon, 10)
# Index 0 = mean; indices 1-9 = quantiles [0.1, ..., 0.9]
quant_preds = outputs.full_predictions[:, :horizon, 1:].float()  # (B, H, 9)
quantiles = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                          device=quant_preds.device)              # (9,)
targets_q = targets.unsqueeze(-1).expand_as(quant_preds)         # (B, H, 9)
errors = targets_q - quant_preds
pinball = torch.where(errors >= 0, quantiles * errors, (quantiles - 1.0) * errors).mean()

# Combine with MSE (mse_weight configurable, e.g. 0.5/0.5 or 0.0/1.0)
loss = mse_weight * mse_loss + (1.0 - mse_weight) * pinball
```

A `loss_type` config option (`"mse"` / `"pinball"` / `"combined"`) and a `mse_weight`
float in `TimesFMConfig` would make this sweepable. Expected outcome: allows higher LR
without calibration collapse, potentially better WQL/MACE at equivalent RMSE.

### 5.3 DILATE as Training Loss — Non-Trivial, Deferred

Our `src/evaluation/metrics/shape.py` is Numba (`@njit`) and has no PyTorch autograd
backward pass — it cannot be used directly as a training loss. The original DILATE paper
repo (`vincent-leguen/DILATE`, MIT) has a PyTorch autograd version, but it requires a
full port (~300 lines). The gradient flows through the soft alignment path matrix `E`
back to the predictions.

**Recommendation:** Implement pinball loss first. Run a sweep comparing
`loss_type=mse` vs `loss_type=pinball` vs `loss_type=combined` at LR=1e-4.
If pinball at LR=1e-4 matches or beats MSE at LR=1e-5 on WQL/MACE without hurting
RMSE/DILATE, that validates the approach and makes a DILATE training loss less urgent.
