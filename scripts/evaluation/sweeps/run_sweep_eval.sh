#!/usr/bin/env bash
# Generic, experiment-spec-driven sweep evaluation launcher.
#
# Defaults target the Chronos-2 forecasting sweep profile but callers can
# override with MODEL_TYPE, SWEEP_SPEC, and EVAL_PYTHON.
#
# Contract preserved via env vars:
#   GPUS, JOBS_PER_GPU, CONFIG_DIR, DRY_RUN
#
# Examples:
#   MODEL_TYPE=chronos2 SWEEP_SPEC=configs/experiments/nocturnal_forecast/chronos2_forecasting_eval_sweep.yaml \
#     bash scripts/evaluation/sweeps/run_sweep_eval.sh
#   DRY_RUN=1 GPUS="0" JOBS_PER_GPU=1 bash scripts/evaluation/sweeps/run_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="${MODEL_TYPE:-chronos2}"
SWEEP_SPEC="${SWEEP_SPEC:-configs/experiments/nocturnal_forecast/chronos2_forecasting_eval_sweep.yaml}"
EVAL_PYTHON="${EVAL_PYTHON:-.venvs/autogluon/bin/python}"

python scripts/experiments/sweep_eval.py \
  --model-type "$MODEL_TYPE" \
  --sweep-spec "$SWEEP_SPEC" \
  --python-executable "$EVAL_PYTHON" \
  "$@"
