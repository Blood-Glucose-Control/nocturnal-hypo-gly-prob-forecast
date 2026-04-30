# TFT Configs

Temporal Fusion Transformer baseline via AutoGluon (quantile regression).
TFT is the **only deep model in this stack that consumes past covariates**, so
the sweep is split: BG-only (00–05) on all four datasets, BG+IOB (06–11)
on the IOB-bearing datasets only (`aleppo_2017 brown_2019 lynch_2022`), and
BG+IOB+COB (12–17) on `aleppo_2017` only (the sole dataset with both insulin
and carb covariates).

GPU: ~8 GB peak → 6 workers per 96 GB Blackwell GPU is safe.

| Config | covariates | ctx | hidden_dim | var_dim | heads | dropout | lr | Notes |
|--------|------------|-----|------------|---------|-------|---------|------|-------|
| `00_bg_baseline.yaml`     | none  | 512 | 32 | 32 | 4 | 0.1 | 1e-3 | BG default |
| `01_bg_wide.yaml`         | none  | 512 | 64 | 32 | 4 | 0.1 | 1e-3 | width ablation |
| `02_bg_long_ctx.yaml`     | none  | 768 | 32 | 32 | 4 | 0.1 | 1e-3 | history ablation |
| `03_bg_high_dropout.yaml` | none  | 512 | 32 | 32 | 4 | 0.3 | 1e-3 | regularisation |
| `04_bg_more_heads.yaml`   | none  | 512 | 32 | 32 | 8 | 0.1 | 1e-3 | attention ablation |
| `05_bg_high_lr.yaml`      | none  | 512 | 32 | 32 | 4 | 0.1 | 3e-3 | optimiser |
| `06_iob_baseline.yaml`     | iob  | 512 | 32 | 32 | 4 | 0.1 | 1e-3 | IOB default |
| `07_iob_wide.yaml`         | iob  | 512 | 64 | 32 | 4 | 0.1 | 1e-3 | width ablation |
| `08_iob_long_ctx.yaml`     | iob  | 768 | 32 | 32 | 4 | 0.1 | 1e-3 | history ablation |
| `09_iob_high_dropout.yaml` | iob  | 512 | 32 | 32 | 4 | 0.3 | 1e-3 | regularisation |
| `10_iob_more_heads.yaml`   | iob  | 512 | 32 | 32 | 8 | 0.1 | 1e-3 | attention ablation |
| `11_iob_high_lr.yaml`      | iob  | 512 | 32 | 32 | 4 | 0.1 | 3e-3 | optimiser |
| `12_iob_cob_baseline.yaml`    | iob, cob  | 512 | 32 | 32 | 4 | 0.1 | 1e-3 | IOB+COB default |
| `13_iob_cob_wide.yaml`        | iob, cob  | 512 | 64 | 32 | 4 | 0.1 | 1e-3 | width ablation |
| `14_iob_cob_high_dropout.yaml`| iob, cob  | 512 | 32 | 32 | 4 | 0.3 | 1e-3 | regularisation |
| `15_iob_cob_more_heads.yaml`  | iob, cob  | 512 | 32 | 32 | 8 | 0.1 | 1e-3 | attention ablation |
| `16_iob_cob_high_lr.yaml`     | iob, cob  | 512 | 32 | 32 | 4 | 0.1 | 3e-3 | optimiser |

All configs use `forecast_length=96`, `batch_size=256`,
`num_batches_per_epoch=50`, `max_epochs=100`, `early_stopping_patience=20`,
`gradient_clip_val=1.0`.

## Run

```bash
bash scripts/experiments/tft_sweep_train.sh
GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/tft_sweep_train.sh 2>&1 | tee logs/tft_sweep_train.log
GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/tft_sweep_eval.sh  2>&1 | tee logs/tft_sweep_eval.log
```
