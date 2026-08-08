# PatchTST Configs

Patch-based Transformer baseline via AutoGluon.
PatchTST supports static + known covariates only — IOB cannot be passed as a
past covariate, so this sweep is **BG-only** and runs across all four datasets.

GPU: ~6 GB peak → 6 workers per 96 GB Blackwell GPU is safe.

| Config | ctx | patch / stride | d_model | nhead | layers | lr | wd | Notes |
|--------|-----|----------------|---------|-------|--------|------|------|-------|
| `00_baseline.yaml`     | 512 | 16 / 8  | 128 | 16 | 3 | 1e-4 | 1e-8 | default |
| `01_short_patch.yaml`  | 512 |  8 / 4  | 128 | 16 | 3 | 1e-4 | 1e-8 | finer tokens |
| `02_long_patch.yaml`   | 512 | 32 / 16 | 128 | 16 | 3 | 1e-4 | 1e-8 | coarser tokens |
| `03_short_ctx.yaml`    | 256 | 16 / 8  | 128 | 16 | 3 | 1e-4 | 1e-8 | history ablation |
| `04_long_ctx.yaml`     | 768 | 16 / 8  | 128 | 16 | 3 | 1e-4 | 1e-8 | history ablation |
| `05_narrow_d.yaml`     | 512 | 16 / 8  |  64 |  8 | 3 | 1e-4 | 1e-8 | width ablation |
| `06_wide_d.yaml`       | 512 | 16 / 8  | 256 | 16 | 3 | 1e-4 | 1e-8 | width ablation |
| `07_deep.yaml`         | 512 | 16 / 8  | 128 | 16 | 6 | 1e-4 | 1e-8 | depth ablation |
| `08_shallow.yaml`      | 512 | 16 / 8  | 128 | 16 | 2 | 1e-4 | 1e-8 | depth ablation |
| `09_high_lr.yaml`      | 512 | 16 / 8  | 128 | 16 | 3 | 3e-4 | 1e-8 | optimiser |
| `10_low_lr.yaml`       | 512 | 16 / 8  | 128 | 16 | 3 | 3e-5 | 1e-8 | optimiser |
| `11_weight_decay.yaml` | 512 | 16 / 8  | 128 | 16 | 3 | 1e-4 | 1e-4 | regularisation |

All configs use `forecast_length=96`, `batch_size=256`,
`num_batches_per_epoch=100`, `max_epochs=100`, `early_stopping_patience=20`,
datasets = `aleppo_2017 brown_2019 lynch_2022 tamborlane_2008`.

## Run

```bash
bash scripts/experiments/patchtst_sweep_train.sh
GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/patchtst_sweep_train.sh 2>&1 | tee logs/patchtst_sweep_train.log
GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/patchtst_sweep_eval.sh  2>&1 | tee logs/patchtst_sweep_eval.log
```
