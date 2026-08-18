#!/usr/bin/env bash
# Profile launcher for TimesFM forecast sweep training.
#
# Canonical generic launcher:
#   scripts/training/sweeps/run_sweep_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${SINGLE_GPU:-0}" == "1" ]]; then
    RESOLVED_GPUS="${GPU0:-0}"
elif [[ -n "${GPUS:-}" ]]; then
    RESOLVED_GPUS="$GPUS"
else
    RESOLVED_GPUS="${GPU0:-0} ${GPU1:-1}"
fi

MODEL_TYPE="timesfm" \
TASK_FAMILY="${TASK_FAMILY:-forecasting}" \
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}" \
SWEEP_SPEC="configs/experiments/nocturnal_forecast/timesfm_forecasting_train_sweep.yaml" \
GPUS="$RESOLVED_GPUS" \
SKIP_STEPS="${SKIP_STEPS:-1 2 4 6 7}" \
JOBS_PER_GPU="${JOBS_PER_GPU:-1}" \
bash scripts/training/sweeps/run_sweep_train.sh "$@"
