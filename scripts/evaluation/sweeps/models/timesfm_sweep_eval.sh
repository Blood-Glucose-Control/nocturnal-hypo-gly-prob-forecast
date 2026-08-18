#!/usr/bin/env bash
# Profile launcher for TimesFM forecast sweep evaluation.
#
# Canonical generic launcher:
#   scripts/evaluation/sweeps/run_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="timesfm" \
TASK_FAMILY="${TASK_FAMILY:-forecasting}" \
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}" \
SWEEP_SPEC="configs/experiments/nocturnal_forecast/timesfm_forecasting_eval_sweep.yaml" \
EVAL_PYTHON="${EVAL_PYTHON:-.venvs/timesfm/bin/python}" \
JOBS_PER_GPU="${JOBS_PER_GPU:-1}" \
bash scripts/evaluation/sweeps/run_sweep_eval.sh "$@"
