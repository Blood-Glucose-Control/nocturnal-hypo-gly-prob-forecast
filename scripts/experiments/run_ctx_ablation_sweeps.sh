#!/usr/bin/env bash
# run_ctx_ablation_sweeps.sh
#
# Orchestrator for the full context-ablation pipeline:
#   deepar train → patchtst train → tft train →
#   deepar eval  → patchtst eval  → tft eval
#
# Each stage must succeed before the next begins.
# Per-stage logs: logs/ctx_ablation_{model}_{train,eval}.log
#
# Usage:
#   bash scripts/experiments/run_ctx_ablation_sweeps.sh
#   nohup bash scripts/experiments/run_ctx_ablation_sweeps.sh \
#     > logs/ctx_ablation_chain.log 2>&1 &
#   echo "PID: $!"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_stage() {
    local stage_name="$1"
    local script="$2"
    local log_file="$3"

    echo ""
    echo "========================================================"
    echo "  STAGE: ${stage_name}"
    echo "  Script: ${script}"
    echo "  Log:    ${log_file}"
    echo "  Start:  $(date)"
    echo "========================================================"

    if bash "$script" > "$log_file" 2>&1; then
        echo "  [OK] ${stage_name} completed at $(date)"
    else
        echo "  [FAIL] ${stage_name} FAILED at $(date)"
        echo "  See ${log_file} for details."
        exit 1
    fi
}

echo "========================================================"
echo "  Context Ablation Sweep — Full Pipeline"
echo "  Start: $(date)"
echo "========================================================"

run_stage "DeepAR train" \
    "scripts/experiments/deepar_ctx_ablation_train.sh" \
    "${LOG_DIR}/ctx_ablation_deepar_train.log"

run_stage "PatchTST train" \
    "scripts/experiments/patchtst_ctx_ablation_train.sh" \
    "${LOG_DIR}/ctx_ablation_patchtst_train.log"

run_stage "TFT train" \
    "scripts/experiments/tft_ctx_ablation_train.sh" \
    "${LOG_DIR}/ctx_ablation_tft_train.log"

run_stage "DeepAR eval" \
    "scripts/experiments/deepar_ctx_ablation_eval.sh" \
    "${LOG_DIR}/ctx_ablation_deepar_eval.log"

run_stage "PatchTST eval" \
    "scripts/experiments/patchtst_ctx_ablation_eval.sh" \
    "${LOG_DIR}/ctx_ablation_patchtst_eval.log"

run_stage "TFT eval" \
    "scripts/experiments/tft_ctx_ablation_eval.sh" \
    "${LOG_DIR}/ctx_ablation_tft_eval.log"

echo ""
echo "========================================================"
echo "  Context Ablation Sweep — ALL STAGES COMPLETE"
echo "  End: $(date)"
echo "========================================================"
echo ""
echo "Results in: experiments/nocturnal_forecasting_ctx_ablation/"
echo "  512ctx_96fh/  — anchor results (copied from sweep)"
echo "  256ctx_96fh/  — ablation level"
echo "  128ctx_96fh/  — ablation level"
echo "  64ctx_96fh/   — ablation level"
