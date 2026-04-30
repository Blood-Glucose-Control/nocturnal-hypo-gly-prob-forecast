#!/usr/bin/env bash
# sundial_sweep_eval.sh
#
# Evaluates Sundial in zero-shot mode across all four holdout datasets with
# full probabilistic and DILATE metrics.
#
# Sundial is a foundation model — no fine-tuning or checkpoint required.
# Config: configs/models/sundial/02_low_samples.yaml  (num_samples=20)
#
# All four datasets are evaluated — Sundial has no covariate support.
#
# Usage:
#   bash scripts/experiments/sundial_sweep_eval.sh
#   CUDA_DEVICE=1 bash scripts/experiments/sundial_sweep_eval.sh
#   DATASETS="lynch_2022" bash scripts/experiments/sundial_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/sundial/bin/python"
GPU="${CUDA_DEVICE:-0}"
CONFIG_DIR="${CONFIG_DIR:-configs/data/holdout_10pct}"
MODEL_CONFIG="configs/models/sundial/02_low_samples.yaml"
REQUESTED_DATASETS_STR="${DATASETS:-lynch_2022 aleppo_2017 brown_2019 tamborlane_2008}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: sundial venv not found at $PYTHON"
    exit 1
fi

IFS=' ' read -r -a DATASETS_ARR <<< "$REQUESTED_DATASETS_STR"

PASS=0
FAIL=0
FAILED=()

for dataset in "${DATASETS_ARR[@]}"; do
    label="sundial / 02_low_samples / ${dataset}"
    echo ""
    echo "============================================================"
    echo "  Eval: ${label}"
    echo "============================================================"

    CMD=(
        "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
        --model sundial
        --model-config "$MODEL_CONFIG"
        --dataset "$dataset"
        --config-dir "$CONFIG_DIR"
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
echo "  Sundial sweep eval complete  $(date)"
echo "  Passed: ${PASS} / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - ${f}"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
