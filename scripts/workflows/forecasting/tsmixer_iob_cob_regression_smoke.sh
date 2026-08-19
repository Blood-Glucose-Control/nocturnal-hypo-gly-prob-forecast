#!/usr/bin/env bash
# Lightweight end-to-end TSMixer integration smoke profile.
#
# Uses the maintained generic forecasting regression smoke wrapper and only
# overrides model/dataset/config defaults for TSMixer + IOB/COB on Aleppo.

set -euo pipefail

: "${MODEL_TYPE:=tsmixer}"
: "${VENV_NAME:=darts}"
: "${DATASETS:=aleppo_2017}"
: "${CONFIG_DIR:=configs/data/holdout_smoke_aleppo}"
: "${MODEL_CONFIG:=configs/models/tsmixer/00_iob_cob_smoke.yaml}"
: "${SKIP_TRAINING:=false}"
: "${SKIP_STEPS:=7}"
: "${EPOCHS:=1}"
: "${RUN_ID:=tsmixer_iob_cob_smoke_$(date +%Y%m%d_%H%M%S)_$$}"

export MODEL_TYPE
export VENV_NAME
export DATASETS
export CONFIG_DIR
export MODEL_CONFIG
export SKIP_TRAINING
export SKIP_STEPS
export EPOCHS
export RUN_ID

exec bash scripts/workflows/forecasting/forecasting_workflow_regression_smoke.sh
