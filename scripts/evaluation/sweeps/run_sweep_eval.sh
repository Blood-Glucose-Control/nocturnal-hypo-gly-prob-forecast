#!/usr/bin/env bash
# Generic, experiment-spec-driven sweep evaluation launcher.
#
# Defaults target the forecasting/nocturnal_forecast task+experiment profile
# with Chronos-2 configs, but callers can override.
#
# Contract preserved via env vars:
#   TASK_FAMILY, EXPERIMENT_TYPE, MODEL_TYPE, SWEEP_SPEC, EVAL_PYTHON,
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
TASK_FAMILY="${TASK_FAMILY:-forecasting}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}"

python scripts/orchestration/sweeps/sweep_eval.py \
  --task-family "$TASK_FAMILY" \
  --experiment-type "$EXPERIMENT_TYPE" \
  --model-type "$MODEL_TYPE" \
  --sweep-spec "$SWEEP_SPEC" \
  --python-executable "$EVAL_PYTHON" \
  "$@"
