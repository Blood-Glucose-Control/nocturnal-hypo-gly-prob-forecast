# Statistical Baseline Configs

Three statistical baselines via AutoGluon. A training step is required
(AutoGluon fits to compute residuals for quantile synthesis).

| Config | Model | Datasets | Notes |
|--------|-------|---------|-------|
| `00_autoarima_bg_only.yaml` | AutoARIMA | all 4 | seasonal=False |
| `01_autoarima_bg_iob.yaml` | AutoARIMA + IOB | aleppo, brown, lynch | seasonal=False |
| `02_theta_bg_only.yaml` | Theta | all 4 | season_length=1 |
| `03_npts_bg_only.yaml` | NPTS | all 4 | natively probabilistic |

All configs apply a **2-hour `time_limit`** cap. Series not fit in time
fall back to Naive — no work is lost.

## Run

```bash
# 1. Train (CPU — runs configs sequentially, writes manifest)
bash scripts/experiments/statistical_sweep_train.sh

# 2. Evaluate (CPU — parallel workers, reads manifest)
bash scripts/experiments/statistical_sweep_eval.sh

# Control CPU parallelism for eval
JOBS_PER_CPU=2 bash scripts/experiments/statistical_sweep_eval.sh

# Log both to file
bash scripts/experiments/statistical_sweep_train.sh 2>&1 | tee logs/statistical_sweep_train.log
bash scripts/experiments/statistical_sweep_eval.sh  2>&1 | tee logs/statistical_sweep_eval.log
```
