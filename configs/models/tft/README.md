# TFT Configs

Temporal Fusion Transformer baseline via AutoGluon (quantile regression).
GPU: ~8 GB peak → 2 workers per 96 GB Blackwell GPU is safe.

| Config | Covariates | Context | Notes |
|--------|-----------|---------|-------|
| `00_bg_only.yaml` | none | 512 | all 4 datasets |
| `01_bg_iob.yaml` | iob | 512 | aleppo, brown, lynch |
| `02_bg_iob_high_lr.yaml` | iob | 512 | lr 1e-3 → 3e-3 |
| `03_bg_iob_short_ctx.yaml` | iob | 256 | ctx ablation |

## Run

```bash
# 1. Train
GPUS="0 1" JOBS_PER_GPU=2 bash scripts/experiments/tft_sweep_train.sh

# 2. Evaluate
GPUS="0 1" JOBS_PER_GPU=2 bash scripts/experiments/tft_sweep_eval.sh

# Log to file
GPUS="0 1" JOBS_PER_GPU=2 bash scripts/experiments/tft_sweep_train.sh 2>&1 | tee logs/tft_sweep_train.log
GPUS="0 1" JOBS_PER_GPU=2 bash scripts/experiments/tft_sweep_eval.sh  2>&1 | tee logs/tft_sweep_eval.log
```
