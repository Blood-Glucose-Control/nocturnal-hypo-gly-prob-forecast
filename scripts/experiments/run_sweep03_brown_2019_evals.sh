#!/usr/bin/env bash
# Run all sweep-03 brown_2019 evaluations sequentially (5k → 10k → 15k → 20k → 25k → 30k → 35k → 40k → 45k → 50k)
set -euo pipefail

EVAL="python scripts/experiments/nocturnal_hypo_eval.py"
COMMON_ARGS="--model chronos2 --dataset brown_2019 --config-dir configs/data/holdout_10pct \
    --context-length 512 --forecast-length 96 --covariate-cols iob insulin_availability --cuda-device 0"

echo "=== sweep03 brown_2019 evals ==="

echo "[1/10] 5k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_5000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps5000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2311_sweep03_5k_modelpt_brown_2019_finetuned

echo "[2/10] 10k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_10000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps10000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2313_sweep03_10k_modelpt_brown_2019_finetuned

echo "[3/10] 15k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_15000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps15000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-16_1000_sweep03_15k_modelpt_brown_2019_finetuned

echo "[4/10] 20k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_20000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps20000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-16_1001_sweep03_20k_modelpt_brown_2019_finetuned

echo "[5/10] 25k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_25000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps25000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2315_sweep03_25k_modelpt_brown_2019_finetuned

echo "[6/10] 30k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_30000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps30000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-16_1002_sweep03_30k_modelpt_brown_2019_finetuned

echo "[7/10] 35k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_35000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps35000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-17_1001_sweep03_35k_modelpt_brown_2019_finetuned

echo "[8/10] 40k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_40000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps40000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-17_1003_sweep03_40k_modelpt_brown_2019_finetuned

echo "[9/10] 45k"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/_tmp_03_steps_45000.yaml \
    --checkpoint trained_models/artifacts/chronos2/sweep_03_steps45000_holdout_workflow/model.pt \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-17_1005_sweep03_45k_modelpt_brown_2019_finetuned

echo "[10/10] 50k (recheck_03)"
$EVAL $COMMON_ARGS \
    --model-config configs/models/chronos2/03_bg_iob_ia_high_lr.yaml \
    --checkpoint "trained_models/artifacts/chronos2/2026-03-12_23:07_RID20260312_230758_1217900_holdout_workflow/model.pt" \
    --output-dir experiments/nocturnal_forecasting/512ctx_96fh/chronos2/2026-03-15_2303_recheck_03_modelpt_brown_2019_finetuned

echo "=== All brown_2019 evals complete ==="
