#!/usr/bin/env bash
# timegrad_eval.sh
#
# Evaluates the trained TimeGrad (cgm_only) checkpoint against all four
# holdout datasets with full probabilistic and DILATE metrics.
#
# Checkpoint: trained_models/artifacts/timegrad/
#             2026-02-24_01:12_RID20260224_011201_2800320_holdout_workflow/model.pt
# Config:     configs/models/timegrad/cgm_only.yaml  (diff_steps=10, num_samples=20)
#
# All four datasets are evaluated — TimeGrad has no covariate support so the
# bg_only covariate bucket applies universally.
#
# Usage:
#   bash scripts/experiments/timegrad_eval.sh
#   CUDA_DEVICE=1 bash scripts/experiments/timegrad_eval.sh
#   DATASETS="lynch_2022" bash scripts/experiments/timegrad_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/timegrad/bin/python"
GPU="${CUDA_DEVICE:-0}"
CONFIG_DIR="${CONFIG_DIR:-configs/data/holdout_10pct}"
MODEL_CONFIG="configs/models/timegrad/cgm_only.yaml"
CHECKPOINT="trained_models/artifacts/timegrad/2026-02-24_01:12_RID20260224_011201_2800320_holdout_workflow/model.pt"
REQUESTED_DATASETS_STR="${DATASETS:-lynch_2022 aleppo_2017 brown_2019 tamborlane_2008}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: timegrad venv not found at $PYTHON"
    exit 1
fi

if [[ ! -e "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT"
    exit 1
fi

IFS=' ' read -r -a DATASETS_ARR <<< "$REQUESTED_DATASETS_STR"

PASS=0
FAIL=0
FAILED=()

for dataset in "${DATASETS_ARR[@]}"; do
    label="timegrad / cgm_only / ${dataset}"
    echo ""
    echo "============================================================"
    echo "  Eval: ${label}"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "============================================================"

    CMD=(
        "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
        --model timegrad
        --model-config "$MODEL_CONFIG"
        --dataset "$dataset"
        --config-dir "$CONFIG_DIR"
        --checkpoint "$CHECKPOINT"
        --context-length 512
        --forecast-length 96
        --cuda-device "$GPU"
        --probabilistic
    )

    if "${CMD[@]}"; then
        echo "[OK] ${label}"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] ${label}"
        FAIL=$((FAIL + 1))
        FAILED+=("${label}")
    fi
done

echo ""
echo "============================================================"
echo "  TimeGrad eval complete  $(date)"
echo "  Passed: ${PASS} / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - ${f}"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
