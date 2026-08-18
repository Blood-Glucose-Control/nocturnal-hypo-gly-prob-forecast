#!/usr/bin/env bash
# Profile launcher for Moirai forecast sweep training.
#
# Canonical generic launcher:
#   scripts/training/sweeps/run_sweep_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="moirai" \
TASK_FAMILY="${TASK_FAMILY:-forecasting}" \
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}" \
SWEEP_SPEC="configs/experiments/nocturnal_forecast/moirai_forecasting_train_sweep.yaml" \
SKIP_STEPS="${SKIP_STEPS:-7}" \
JOBS_PER_GPU="${JOBS_PER_GPU:-1}" \
bash scripts/training/sweeps/run_sweep_train.sh "$@"
