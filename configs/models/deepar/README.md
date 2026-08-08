# DeepAR Configs

RNN-based probabilistic baseline (StudentT output) via AutoGluon.
AutoGluon's PyTorch DeepAR is **LSTM-only** and supports static + known
covariates only — IOB cannot be passed as a past covariate, so this sweep is
**BG-only** and runs across all four datasets.

GPU: ~4 GB peak → 6 workers per 96 GB Blackwell GPU is safe.

| Config | ctx | hidden | layers | dropout | lr | Notes |
|--------|-----|--------|--------|---------|------|-------|
| `00_baseline.yaml`     | 512 |  64 | 2 | 0.1 | 1e-3 | default |
| `01_short_ctx.yaml`    | 256 |  64 | 2 | 0.1 | 1e-3 | history ablation |
| `02_long_ctx.yaml`     | 768 |  64 | 2 | 0.1 | 1e-3 | history ablation |
| `03_wide.yaml`         | 512 | 128 | 2 | 0.1 | 1e-3 | width ablation |
| `04_narrow.yaml`       | 512 |  32 | 2 | 0.1 | 1e-3 | width ablation |
| `05_deep.yaml`         | 512 |  64 | 3 | 0.1 | 1e-3 | depth ablation |
| `06_shallow.yaml`      | 512 |  64 | 1 | 0.1 | 1e-3 | depth ablation |
| `07_high_dropout.yaml` | 512 |  64 | 2 | 0.3 | 1e-3 | regularisation |
| `08_low_dropout.yaml`  | 512 |  64 | 2 | 0.0 | 1e-3 | regularisation |
| `09_high_lr.yaml`      | 512 |  64 | 2 | 0.1 | 3e-3 | optimiser |
| `10_low_lr.yaml`       | 512 |  64 | 2 | 0.1 | 3e-4 | optimiser |
| `11_big.yaml`          | 768 | 128 | 3 | 0.1 | 1e-3 | combined large |

All configs use `forecast_length=96`, `batch_size=128`,
`num_batches_per_epoch=50`, `max_epochs=100`, `early_stopping_patience=20`,
`gradient_clip_val=10.0`, datasets = `aleppo_2017 brown_2019 lynch_2022 tamborlane_2008`.

## Run

```bash
bash scripts/experiments/deepar_sweep_train.sh
GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/deepar_sweep_train.sh 2>&1 | tee logs/deepar_sweep_train.log
GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/deepar_sweep_eval.sh  2>&1 | tee logs/deepar_sweep_eval.log
```
