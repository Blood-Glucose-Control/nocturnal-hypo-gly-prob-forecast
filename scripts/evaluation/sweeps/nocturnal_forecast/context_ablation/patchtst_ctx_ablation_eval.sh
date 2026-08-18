#!/usr/bin/env bash
# Profile launcher for PatchTST context-ablation sweep evaluation.
#
# Canonical generic launcher:
#   scripts/evaluation/sweeps/run_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="patchtst" \
TASK_FAMILY="${TASK_FAMILY:-forecasting}" \
EXPERIMENT_TYPE="${EXPERIMENT_TYPE:-nocturnal_forecast}" \
SWEEP_SPEC="configs/experiments/nocturnal_forecast/patchtst_ctx_ablation_forecasting_eval_sweep.yaml" \
EVAL_PYTHON="${EVAL_PYTHON:-.venvs/autogluon/bin/python}" \
JOBS_PER_GPU="${JOBS_PER_GPU:-10}" \
bash scripts/evaluation/sweeps/run_sweep_eval.sh \
  --manifest-path "trained_models/artifacts/patchtst/ctx_ablation_manifest.txt" \
  --done-file "logs/patchtst_ctx_ablation_eval_done.log" \
  "$@"
