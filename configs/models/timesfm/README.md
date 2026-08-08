# TimesFM Sweep Configs

## Loss Functions (`loss_fn`)

Configured via the `loss_fn` field in `TimesFMConfig`. Default is `"pinball"`.

| `loss_fn` | Supervision targets | Notes |
|---|---|---|
| `"mse"` | Mean-prediction head only (index 0 of `full_predictions`) | Original behaviour; quantile heads never trained |
| `"pinball"` | All 9 quantile heads (0.1–0.9) | WQL-equivalent; fully supervises calibration |
| `"joint"` | Mean head (MSE) + all 9 quantile heads (pinball) | Combines both; mean head has dual supervision |
| `"dilate"` | Mean head via DILATE loss | Temporal shape + delay penalty; no quantile supervision |
| `"dilate_pinball_median"` | All 9 quantile heads (pinball) + median trajectory (DILATE) | Temporal regulariser on the 0.5 quantile only; cheapest DILATE variant |
| `"dilate_pinball"` | All 9 quantile heads (pinball + DILATE on each) | DILATE applied to every quantile trajectory (mean-pooled); ~9× slower per step than `dilate_pinball_median` |

### DILATE hyperparameters

| Field | Default | Meaning |
|---|---|---|
| `dilate_alpha` | `0.5` | Balance between shape loss (soft-DTW) and timing loss (path DTW). 1 = pure shape, 0 = pure timing |
| `dilate_gamma` | `0.01` | Soft-DTW smoothing. Smaller → closer to hard DTW |
| `dilate_weight` | `0.5` | Scalar weight on the DILATE term added to pinball: `loss = pinball + dilate_weight * dilate` |

---

## Mid-Training Eval Callback

All configs support per-epoch in-training evaluation via `MidTrainingEvalCallback` (enabled with
`eval_during_training: true`, which is the default).

### How the temporal eval split works

The raw per-patient segments are split **before** windowing to prevent context-window leakage:

1. For each patient segment of length *n*, an eval slice covers the **last `eval_temporal_frac × n`
   samples** (minimum: one full forecast horizon).
2. The slice is prepended with `context_length` samples from just before the cutpoint so the first
   eval window has full context. All **forecast targets** come from the held-out tail.
3. Eval windows use `stride = horizon_length` (non-overlapping) to avoid double-counting.
4. If `eval_subsample` is set, windows are downsampled uniformly to that count.

### Output

At the end of every epoch the callback appends a row to **`epoch_metrics.csv`** in the run's
output directory:

```
epoch, train_loss, wql, coverage_50, coverage_80, coverage_95, mace, rmse
```

This replaces the previous approach of starting 6 separate training runs from scratch at different
epoch budgets (configs `00_3ep` → `05_35ep`). A single 35-epoch run with the callback produces the
full duration-ablation curve at a fraction of the compute.

### Config fields

| Field | Default | Meaning |
|---|---|---|
| `eval_during_training` | `true` | Enable/disable the callback entirely |
| `eval_temporal_frac` | `0.10` | Fraction of each series reserved as temporal eval targets |
| `eval_subsample` | `null` | Max eval windows; `null` = use all |

---

## Sweep Config Index

| File | LR | Epochs | Loss | GPU lane | Purpose |
|---|---|---|---|---|---|
| `00_long_run.yaml` | 1e-5 | 35 | pinball | GPU 0 (solo) | Duration ablation via per-epoch callback; replaces 6 epoch-ablation configs |
| `01_lr_2e5.yaml` | 2e-5 | 15 | pinball | GPU 1 | High-LR probe A (2× safe LR) |
| `02_lr_5e5.yaml` | 5e-5 | 10 | pinball | GPU 1 | High-LR probe B (5× safe LR) |
| `03_dilate_pinball_median_3ep.yaml` | 1e-5 | 3 | dilate_pinball_median | GPU 1 | Smoke-test: DILATE on median trajectory |
| `04_dilate_pinball_all_3ep.yaml` | 1e-5 | 3 | dilate_pinball | GPU 1 | Smoke-test: DILATE on all 9 quantile trajectories |

**GPU lane balance:** GPU 0 runs `00_long_run` (35 ep ≈ long pole). GPU 1 runs configs 01–04
sequentially (15 + 10 + 3 + 3 = 31 ep ≈ balanced).

Once configs 03 and 04 are verified correct, bump `num_epochs` to 15 and rerun for the full
DILATE vs pinball comparison at matched duration.
