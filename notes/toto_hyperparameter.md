# Toto Hyperparameter Sweep — Training Commands

All commands assume you are in the repo root.
`CONFIG_DIR` is set to `holdout_10pct` throughout; change if needed.
Step 7 (resume training) is skipped via `SKIP_STEPS="7"` on every run.

---

## Environment Setup (run once per session)

**1. Activate conda base environment:**
```bash
conda activate env-in-conda
```

**2. Create or activate the Toto model venv** (first run creates `.venvs/toto` and installs deps; subsequent runs just activate it):
```bash
source scripts/setup_model_env.sh toto
```

> The training workflow script (`run_holdout_generic_workflow.sh`) will auto-activate `.venvs/toto` once it exists. The `source scripts/setup_model_env.sh toto` step above is only needed to create it the first time, or if you want to run Python directly outside the workflow script.

---

## Smoke Tests (run these first)

These use 100-step configs for fast end-to-end validation (<5 min each).

**BG-only path** — verifies the basic Toto fine-tuning pipeline:
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/bg_only_smoke_test.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

**CGM+Insulin path** — exercises `bg_iob_smoke_test.yaml` end-to-end:
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/bg_iob_smoke_test.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

**CGM+Insulin+Carbs path** — exercises `bg_iob_cob_smoke_test.yaml` end-to-end (carb feature engineering path):
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/bg_iob_cob_smoke_test.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## BG-Only Baseline

Datasets: `aleppo_2017 brown_2019 lynch_2022 tamborlane_2008`

### 00 — BG only  |  lr=1e-4  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/00_bg_only.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022 tamborlane_2008" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## CGM + Insulin Runs

Datasets: `aleppo_2017 brown_2019 lynch_2022`

### 01 — IOB only  |  lr=1e-4  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/01_bg_iob.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 02 — IOB + insulin_availability  |  lr=1e-4  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/02_bg_iob_ia.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 03 — IOB + insulin_availability  |  lr=5e-4  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/03_bg_iob_ia_high_lr.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 04 — IOB + insulin_availability  |  lr=1e-4  |  ctx=288
```bash
CUDA_VISIBLE_DEVICES=1 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/04_bg_iob_short_ctx.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 brown_2019 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## CGM + Insulin + Carbs Runs

Datasets: `aleppo_2017 lynch_2022`
(brown_2019 excluded — no meal data in DCLP3 export)

### 05 — IOB + COB  |  lr=1e-4  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/05_bg_iob_cob.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 06 — IOB + COB + insulin_availability + carb_availability  |  lr=1e-4  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/06_bg_full_features.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 07 — IOB + COB  |  lr=5e-4  |  ctx=512
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/07_bg_iob_cob_high_lr.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

### 08 — IOB + COB  |  lr=1e-4  |  ctx=288
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_TYPE="toto" MODEL_CONFIG="configs/models/toto/08_bg_iob_cob_short_ctx.yaml" CONFIG_DIR="configs/data/holdout_10pct" DATASETS="aleppo_2017 lynch_2022" SKIP_TRAINING="false" SKIP_STEPS="7" ./scripts/examples/run_holdout_generic_workflow.sh
```

---

## Comparison Matrix

| Config | Covariates                          | LR   | Context | Datasets           | Key comparison                |
| ------ | ----------------------------------- | ---- | ------- | ------------------ | ----------------------------- |
| 00     | none                                | 1e-4 | 512     | ale, bro, lyn, tam | BG-only baseline              |
| **01** | iob                                 | 1e-4 | 512     | ale, bro, lyn      | isolates IOB vs 00            |
| **02** | iob, insulin_avail                  | 1e-4 | 512     | ale, bro, lyn      | +availability vs 01           |
| **03** | iob, insulin_avail                  | 5e-4 | 512     | ale, bro, lyn      | higher LR vs 02               |
| **04** | iob, insulin_avail                  | 1e-4 | 288     | ale, bro, lyn      | shorter ctx vs 02             |
| **05** | iob, cob                            | 1e-4 | 512     | ale, lyn           | baseline carb addition vs 01  |
| **06** | iob, cob, insulin_avail, carb_avail | 1e-4 | 512     | ale, lyn           | kitchen-sink vs 05            |
| **07** | iob, cob                            | 5e-4 | 512     | ale, lyn           | higher LR vs 05               |
| **08** | iob, cob                            | 1e-4 | 288     | ale, lyn           | shorter ctx vs 05             |

---

## Nocturnal Evaluation

After training completes, run nocturnal evaluation (midnight-anchored 8-hour overnight forecasting) on each holdout dataset. Replace `<CHECKPOINT>` with the actual `model.pt` path from the training output.

> **Tip**: Each training run prints the checkpoint path at the end. It lives at
> `trained_models/artifacts/toto/<timestamp>_RID<run_id>_holdout_workflow/model.pt`

### 00 — BG only (zero-shot baseline: omit `--checkpoint`)

Checkpoint: `trained_models/artifacts/toto/2026-03-18_06:21_RID20260318_062105_1962383_holdout_workflow/model.pt`

```bash
# Zero-shot (no checkpoint)
for ds in aleppo_2017 brown_2019 lynch_2022 tamborlane_2008; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/00_bg_only.yaml \
    --context-length 512 --forecast-length 96 \
    --cuda-device 0
done

# Fine-tuned
for ds in aleppo_2017 brown_2019 lynch_2022 tamborlane_2008; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/00_bg_only.yaml \
    --context-length 512 --forecast-length 96 \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/toto/2026-03-18_06:21_RID20260318_062105_1962383_holdout_workflow/model.pt
done
```

### 01 — IOB only

Checkpoint: `trained_models/artifacts/toto/2026-03-18_20:06_RID20260318_200624_2050916_holdout_workflow/model.pt`

```bash
for ds in aleppo_2017 brown_2019 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/01_bg_iob.yaml \
    --context-length 512 --forecast-length 96 \
    --covariate-cols iob \
    --cuda-device 0 \
    --checkpoint trained_models/artifacts/toto/2026-03-18_20:06_RID20260318_200624_2050916_holdout_workflow/model.pt
done
```

### 02 — IOB + insulin_availability
```bash
for ds in aleppo_2017 brown_2019 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/02_bg_iob_ia.yaml \
    --context-length 512 --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint <CHECKPOINT>
done
```

### 03 — IOB + insulin_availability  |  lr=5e-4
```bash
for ds in aleppo_2017 brown_2019 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/03_bg_iob_ia_high_lr.yaml \
    --context-length 512 --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint <CHECKPOINT>
done
```

### 04 — IOB + insulin_availability  |  ctx=288
```bash
for ds in aleppo_2017 brown_2019 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/04_bg_iob_short_ctx.yaml \
    --context-length 288 --forecast-length 96 \
    --covariate-cols iob insulin_availability \
    --cuda-device 0 \
    --checkpoint <CHECKPOINT>
done
```

### 05 — IOB + COB
```bash
for ds in aleppo_2017 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/05_bg_iob_cob.yaml \
    --context-length 512 --forecast-length 96 \
    --covariate-cols iob cob \
    --cuda-device 0 \
    --checkpoint <CHECKPOINT>
done
```

### 06 — IOB + COB + insulin_availability + carb_availability
```bash
for ds in aleppo_2017 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/06_bg_full_features.yaml \
    --context-length 512 --forecast-length 96 \
    --covariate-cols iob cob insulin_availability carb_availability \
    --cuda-device 0 \
    --checkpoint <CHECKPOINT>
done
```

### 07 — IOB + COB  |  lr=5e-4
```bash
for ds in aleppo_2017 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/07_bg_iob_cob_high_lr.yaml \
    --context-length 512 --forecast-length 96 \
    --covariate-cols iob cob \
    --cuda-device 0 \
    --checkpoint <CHECKPOINT>
done
```

### 08 — IOB + COB  |  ctx=288
```bash
for ds in aleppo_2017 lynch_2022; do
  python scripts/experiments/nocturnal_hypo_eval.py \
    --model toto \
    --dataset "$ds" \
    --config-dir configs/data/holdout_10pct \
    --model-config configs/models/toto/08_bg_iob_cob_short_ctx.yaml \
    --context-length 288 --forecast-length 96 \
    --covariate-cols iob cob \
    --cuda-device 0 \
    --checkpoint <CHECKPOINT>
done
```

---

## Notes

- **Model**: `Datadog/Toto-Open-Base-1.0` — downloaded automatically on first run.
- **Training**: Toto uses PyTorch Lightning with `max_steps` (gradient steps, not epochs). Default 3000 steps. Smoke tests use 100 steps.
- **LR schedule**: Warmup (200 steps) → stable peak (1000 steps) → cosine decay (1000 steps) down to `min_lr`.
- **Covariates**: Passed as exogenous variables (past-only) via `covariate_cols` in the YAML config. Toto encodes these alongside BG as multivariate input.
- **Step 7 skipped**: Resume training is skipped (`SKIP_STEPS="7"`) on all runs since we want clean single-phase training results.
- **Forecast length**: 96 steps (8 hours at 5-min intervals) to match Chronos-2 sweep.
- **Results**: Written to `trained_models/artifacts/toto/<timestamp>_RID<run_id>_holdout_workflow/`.

## Environment Setup

The `.venvs/toto` venv required the following fixes for Blackwell GPUs:

```bash
# PyTorch 2.10 for sm_120 (Blackwell) support
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128

# setuptools <82 (82+ removed pkg_resources which toto-ts imports)
pip install "setuptools<82"
```

Additionally, the installed toto package's `helpers.py` was patched to skip `pd.infer_freq()` when
`freq` is already present in the HuggingFace dataset (our CGM data has gaps that cause
`pd.infer_freq` to return `None`). The patch is at:

```
.venvs/toto/lib/python3.12/site-packages/toto/data/util/helpers.py  (line ~213)
```

Changed unconditional `hf_dataset.map(lambda: {"freq": pd.infer_freq(...)})` to:
```python
if "freq" not in hf_dataset.column_names:
    hf_dataset = hf_dataset.map(lambda item: {"freq": pd.infer_freq(...)})
```
