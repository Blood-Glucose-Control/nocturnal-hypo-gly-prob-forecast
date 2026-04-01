# Chronos-2 Hyperparameter Sweep — Training Commands

All commands assume you are in the repo root.
`CONFIG_DIR` is set to `holdout_10pct` throughout; change if needed.

---

## Environment Setup (run once per session)

**1. Activate conda base environment:**
```bash
conda activate env-in-conda
```

**2. Create or activate the Chronos-2 model venv** (first run creates `.venvs/chronos2` and installs deps; subsequent runs just activate it):
```bash
source scripts/setup_model_env.sh chronos2
```

> The training workflow script (`run_holdout_generic_workflow.sh`) will auto-activate `.venvs/chronos2` once it exists. The `source scripts/setup_model_env.sh chronos2` step above is only needed to create it the first time, or if you want to run Python directly outside the workflow script.

---

## Smoke Tests (run these first)

**CGM+Insulin path** — exercises `02_bg_iob.yaml` end-to-end:
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/02_bg_iob.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017" SKIP_TRAINING="false" fine_tune_steps=100 ./scripts/examples/run_holdout_generic_workflow.sh
```

**CGM+Insulin+Carbs path** — exercises `05_bg_iob_cob.yaml` end-to-end (carb feature engineering path):
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/05_bg_iob_cob.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017" SKIP_TRAINING="false" fine_tune_steps=100 ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## CGM + Insulin Runs

Datasets: `aleppo_2017 brown_2019 lynch_2022`

### 02 — IOB only  |  lr=1e-5  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/02_bg_iob.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 03 — IOB + insulin_availability  |  lr=5e-5  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/03_bg_iob_ia_high_lr.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 04 — IOB + insulin_availability  |  lr=1e-5  |  ctx=288
```bash
CUDA_VISIBLE_DEVICES=1 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/04_bg_iob_short_ctx.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## CGM + Insulin + Carbs Runs

Datasets: `aleppo_2017 lynch_2022`
(brown_2019 excluded — no meal data in DCLP3 export)

### 05 — IOB + COB  |  lr=1e-5  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/05_bg_iob_cob.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 06 — IOB + COB + insulin_availability + carb_availability  |  lr=1e-5  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/06_bg_full_features.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 07 — IOB + COB  |  lr=5e-5  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/07_bg_iob_cob_high_lr.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 08 — IOB + COB  |  lr=1e-5  |  ctx=288
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/08_bg_iob_cob_short_ctx.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## Reference: Existing Baselines

### 00 — BG only  |  lr=1e-5  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/00_bg_only.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 01 — IOB + insulin_availability  |  lr=1e-5  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="chronos2" MODEL_CONFIG="configs/models/chronos2/01_bg_iob_insulin_availability.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## Nocturnal Evaluation — Latest Trained Model

The latest run (`2026-03-12`) trained `02_bg_iob.yaml` (`[iob]`, lr=1e-5, ctx=512) on `aleppo_2017`.
Checkpoint: `trained_models/artifacts/chronos2/2026-03-12_04:26_RID20260312_042610_1124210_holdout_workflow/resumed_training/model.pt`

Run nocturnal evaluation against the `aleppo_2017` holdout:
```bash
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset aleppo_2017 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/02_bg_iob.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_04:26_RID20260312_042610_1124210_holdout_workflow/resumed_training/model.pt
```

To evaluate on the other two insulin-capable datasets against the same checkpoint:
```bash
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset brown_2019 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/02_bg_iob.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_04:26_RID20260312_042610_1124210_holdout_workflow/resumed_training/model.pt
```

```bash
python scripts/experiments/nocturnal_hypo_eval.py \
    --model chronos2 \
    --dataset lynch_2022 \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/chronos2/02_bg_iob.yaml \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/chronos2/2026-03-12_04:26_RID20260312_042610_1124210_holdout_workflow/resumed_training/model.pt
```

> Results are written to `experiments/nocturnal_forecasting/512ctx_96fh/chronos2/`.

---

## Comparison Matrix

| Config | Covariates                          | LR   | Context | Datasets      | Key comparison               |
| ------ | ----------------------------------- | ---- | ------- | ------------- | ---------------------------- |
| 00     | none                                | 1e-5 | 512     | ale, bro, lyn | BG-only baseline             |
| 01     | iob, insulin_avail                  | 1e-5 | 512     | ale, bro, lyn | existing baseline            |
| **02** | iob                                 | 1e-5 | 512     | ale, bro, lyn | isolates iob vs 00 and 01    |
| **03** | iob, insulin_avail                  | 5e-5 | 512     | ale, bro, lyn | higher LR vs 01              |
| **04** | iob, insulin_avail                  | 1e-5 | 288     | ale, bro, lyn | shorter ctx vs 01            |
| **05** | iob, cob                            | 1e-5 | 512     | ale, lyn      | baseline carb addition vs 02 |
| **06** | iob, cob, insulin_avail, carb_avail | 1e-5 | 512     | ale, lyn      | kitchen-sink vs 05           |
| **07** | iob, cob                            | 5e-5 | 512     | ale, lyn      | higher LR vs 05              |
| **08** | iob, cob                            | 1e-5 | 288     | ale, lyn      | shorter ctx vs 05            |
