#!/usr/bin/env bash
# Generic, experiment-spec-driven sweep training launcher.
#
# Defaults target the forecasting/nocturnal_forecast task+experiment profile
# with Chronos-2 model configs, but callers can override.
#
# Contract preserved via env vars:
#   TASK_FAMILY, EXPERIMENT_TYPE, MODEL_TYPE, SWEEP_SPEC,
#   GPUS, JOBS_PER_GPU, CONFIG_DIR, SKIP_STEPS, DRY_RUN
#
# Examples:
#   MODEL_TYPE=chronos2 SWEEP_SPEC=configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml \
#     bash scripts/training/sweeps/run_sweep_train.sh
#   DRY_RUN=1 GPUS="0" JOBS_PER_GPU=1 bash scripts/training/sweeps/run_sweep_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="${MODEL_TYPE:-chronos2}"
SWEEP_SPEC="${SWEEP_SPEC:-configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml}"
TASK_FAMILY="${TASK_FAMILY:-forecasting}"
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}"

python scripts/experiments/sweep_train.py \
  --task-family "$TASK_FAMILY" \
  --experiment-type "$EXPERIMENT_TYPE" \
  --model-type "$MODEL_TYPE" \
  --sweep-spec "$SWEEP_SPEC" \
  "$@"
