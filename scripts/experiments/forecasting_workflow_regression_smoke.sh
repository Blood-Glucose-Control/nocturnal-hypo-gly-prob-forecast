#!/bin/bash
# Deterministic, bounded generic forecasting regression profile.
#
# Use this after major workflow/model-runtime changes to confirm that the
# end-to-end path (config validation -> train -> load checkpoint) still works.
#
# Usage:
#   bash scripts/experiments/forecasting_workflow_regression_smoke.sh
#   MODEL_TYPE=ttm MODEL_CONFIG=configs/models/ttm/default.yaml \
#     bash scripts/experiments/forecasting_workflow_regression_smoke.sh

set -euo pipefail

: "${MODEL_TYPE:=chronos2}"
: "${DATASETS:=brown_2019}"
: "${CONFIG_DIR:=configs/data/holdout_10pct}"
: "${MODEL_CONFIG:=configs/models/chronos2/bg_only_test.yaml}"
: "${SKIP_TRAINING:=false}"
: "${SKIP_STEPS:=7}"
: "${EPOCHS:=1}"
: "${BATCH_SIZE:=}"
: "${RUN_ID:=regression_smoke_$(date +%Y%m%d_%H%M%S)_$$}"

echo "==================================================================="
echo "Generic Forecasting Regression Smoke"
echo "==================================================================="
echo "MODEL_TYPE:    ${MODEL_TYPE}"
echo "DATASETS:      ${DATASETS}"
echo "CONFIG_DIR:    ${CONFIG_DIR}"
echo "MODEL_CONFIG:  ${MODEL_CONFIG}"
echo "SKIP_TRAINING: ${SKIP_TRAINING}"
echo "SKIP_STEPS:    ${SKIP_STEPS}"
echo "EPOCHS:        ${EPOCHS}"
echo "BATCH_SIZE:    ${BATCH_SIZE:-from config/default}"
echo "RUN_ID:        ${RUN_ID}"
echo "==================================================================="
echo ""

export MODEL_TYPE
export DATASETS
export CONFIG_DIR
export MODEL_CONFIG
export SKIP_TRAINING
export SKIP_STEPS
export EPOCHS
export BATCH_SIZE
export RUN_ID

exec bash scripts/experiments/run_forecasting_workflow.sh
