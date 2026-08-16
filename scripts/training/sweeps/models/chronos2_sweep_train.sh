#!/usr/bin/env bash
# Thin compatibility launcher for Chronos-2 sweep training.
#
# Canonical generic launcher:
#   scripts/training/sweeps/run_sweep_train.sh
#
# Optional fast path-check:
#   DRY_RUN=1 bash scripts/training/sweeps/run_sweep_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="chronos2" \
TASK_FAMILY="${TASK_FAMILY:-forecasting}" \
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}" \
SWEEP_SPEC="configs/experiments/nocturnal_forecast/chronos2_forecasting_train_sweep.yaml" \
bash scripts/training/sweeps/run_sweep_train.sh "$@"
