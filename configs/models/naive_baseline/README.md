# Naive Baseline Configs

Two zero-parameter baselines. AutoGluon fits Naive/Average in seconds.

| Config | Model | Datasets |
|--------|-------|---------|
| `00_naive.yaml` | Naive (last-value carry-forward) | all 4 |
| `01_average.yaml` | Average (historical mean) | all 4 |

> **Note:** PIT histograms will show miscalibration. This is intentional —
> synthetic quantiles from residuals contrast with natively probabilistic
> models (NPTS, DeepAR, PatchTST, TFT, Chronos-2) in the paper.

## Run

```bash
# Step 1 — Train (fits AutoGluon predictor; ~seconds per config)
bash scripts/experiments/naive_baseline_sweep_train.sh 2>&1 | tee logs/naive_train.log

# Step 2 — Evaluate
bash scripts/experiments/naive_baseline_sweep_eval.sh 2>&1 | tee logs/naive_eval.log

# Control eval parallelism
JOBS_PER_CPU=4 bash scripts/experiments/naive_baseline_sweep_eval.sh | tee logs/naive_eval.log
```
