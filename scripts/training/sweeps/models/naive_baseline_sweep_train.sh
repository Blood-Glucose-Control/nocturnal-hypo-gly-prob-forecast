#!/usr/bin/env bash
# Profile launcher for naive baseline forecast sweep training.
#
# Canonical generic launcher:
#   scripts/training/sweeps/run_sweep_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="naive_baseline" \
TASK_FAMILY="${TASK_FAMILY:-forecasting}" \
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}" \
SWEEP_SPEC="configs/experiments/nocturnal_forecast/naive_baseline_forecasting_train_sweep.yaml" \
VENV_NAME="${VENV_NAME:-chronos2}" \
SKIP_STEPS="${SKIP_STEPS:-1 2 4 7}" \
JOBS_PER_GPU="${JOBS_PER_GPU:-1}" \
bash scripts/training/sweeps/run_sweep_train.sh "$@"
