#!/usr/bin/env bash
# run_chronos2_sweep.sh
#
# Full Chronos-2 sweep orchestrator.  Runs in three phases:
#
#   Phase 1 — Smoketest training
#     Trains config 98_checkpoint_smoketest.yaml on brown_2019 (fastest dataset,
#     200 steps + 2 checkpoints). Verifies that periodic checkpoint materialisation
#     works before committing GPU-hours to the full sweep.
#
#   Phase 2 — Smoketest eval
#     Runs run_sweep03_checkpoint_evals.sh on the smoketest artifact, checking that
#     step_100 and step_200 eval results are produced.
#
#   Phase 3 — Full sweep training (parallel GPUs)
#     Distributes configs 00–09 across all available GPUs via
#     chronos2_sweep_train.sh.  Only runs if both smoke phases pass.
#
# Usage:
#   bash scripts/experiments/run_chronos2_sweep.sh           # smoke + full sweep
#   SMOKE_ONLY=1 bash scripts/experiments/run_chronos2_sweep.sh   # smoke only, then stop
#   GPUS="0 1"   bash scripts/experiments/run_chronos2_sweep.sh
#   SMOKE_DATASET="lynch_2022" bash scripts/experiments/run_chronos2_sweep.sh
#   SKIP_SMOKETEST=1 bash scripts/experiments/run_chronos2_sweep.sh
#
# After the full sweep finishes, run checkpoint evals with:
#   bash scripts/experiments/run_sweep03_checkpoint_evals.sh \
#       <artifact_dir_from_sweep_manifest>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMOKE_DATASET="${SMOKE_DATASET:-brown_2019}"           # fastest dataset
SMOKE_CONFIG="configs/models/chronos2/98_checkpoint_smoketest.yaml"
SMOKE_GPU="${SMOKE_GPU:-${GPUS:-0}}"                   # single GPU for smoketest
# Use first GPU listed if GPUS is multi-valued
SMOKE_GPU_ID="${SMOKE_GPU%% *}"

CONFIG_DIR="configs/data/holdout_10pct"
WORKFLOW="scripts/examples/run_holdout_generic_workflow.sh"
EVAL_SCRIPT="scripts/experiments/run_sweep03_checkpoint_evals.sh"
TRAIN_SCRIPT="scripts/experiments/chronos2_sweep_train.sh"

ARTIFACT_BASE="trained_models/artifacts/chronos2"
LOG_DIR="logs"
mkdir -p "$ARTIFACT_BASE" "$LOG_DIR"

SKIP_SMOKETEST="${SKIP_SMOKETEST:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

echo "================================================================"
echo "  Chronos-2 sweep orchestrator  $(date)"
echo "  GPUS: ${GPUS:-auto-detect}"
echo "  Smoke GPU: ${SMOKE_GPU_ID}  |  Smoke dataset: ${SMOKE_DATASET}"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
phase() { echo ""; echo "--- Phase $1: $2 ---"; echo ""; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Phase 1: Smoketest training
# ---------------------------------------------------------------------------
if [[ "$SKIP_SMOKETEST" == "1" ]]; then
    echo "SKIP_SMOKETEST=1 — skipping smoke phases, proceeding to full sweep."
else
    phase 1 "Smoketest training (200 steps, ${SMOKE_DATASET})"

    SMOKE_RUN_ID="smoke_$(date +%Y%m%d_%H%M%S)_$$"
    SMOKE_ARTIFACT="${ARTIFACT_BASE}/${SMOKE_RUN_ID}_smoketest"

    if CUDA_VISIBLE_DEVICES="$SMOKE_GPU_ID" \
       MODEL_TYPE="chronos2" \
       MODEL_CONFIG="$SMOKE_CONFIG" \
       CONFIG_DIR="$CONFIG_DIR" \
       DATASETS="$SMOKE_DATASET" \
       SKIP_TRAINING="false" \
       SKIP_STEPS="1 2 4 7" \
       OUTPUT_BASE_DIR="$SMOKE_ARTIFACT" \
       RUN_ID="$SMOKE_RUN_ID" \
       "$WORKFLOW"; then
        pass "Smoketest training completed"
    else
        fail "Smoketest training failed — stopping. Check logs above."
    fi

    # -------------------------------------------------------------------------
    # Phase 2: Smoketest checkpoint eval
    # -------------------------------------------------------------------------
    phase 2 "Smoketest checkpoint eval"

    # Find the actual artifact sub-dir created by the holdout workflow
    # (it creates a subdir like <run_id>_<dataset>/ or similar)
    SMOKE_EVAL_DIR=$(find "$SMOKE_ARTIFACT" -maxdepth 1 -type d -name "snapshots" -printf '%h\n' 2>/dev/null | head -1 || true)
    if [[ -z "$SMOKE_EVAL_DIR" ]]; then
        # Holdout workflow may wrap into a dataset subdir
        SMOKE_EVAL_DIR=$(find "$SMOKE_ARTIFACT" -maxdepth 2 -type d -name "snapshots" -printf '%h\n' 2>/dev/null | head -1 || true)
    fi

    if [[ -z "$SMOKE_EVAL_DIR" ]]; then
        fail "Could not find snapshots/ under ${SMOKE_ARTIFACT}. Did training save checkpoints?"
    fi

    echo "  Artifact: ${SMOKE_EVAL_DIR}"

    if DATASETS="$SMOKE_DATASET" \
       CUDA_DEVICE="$SMOKE_GPU_ID" \
       MODEL_CONFIG="$SMOKE_CONFIG" \
       "$EVAL_SCRIPT" "$SMOKE_EVAL_DIR"; then
        pass "Smoketest checkpoint eval completed"
    else
        fail "Smoketest checkpoint eval failed — stopping."
    fi

    # Verify expected checkpoint output dirs exist
    SMOKE_RESULTS_BASE="experiments/nocturnal_forecasting/512ctx_96fh/chronos2/250k_checkpoints"
    MISSING=()
    for expected_step in step_100 step_200; do
        expected_dir="${SMOKE_RESULTS_BASE}/${expected_step}_${SMOKE_DATASET}"
        if [[ ! -d "$expected_dir" ]]; then
            MISSING+=("$expected_dir")
        fi
    done

    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "[FAIL] Expected checkpoint eval output dirs not found:"
        for d in "${MISSING[@]}"; do
            echo "         $d"
        done
        fail "Smoketest gate failed — expected step_100 and step_200 results."
    fi

    pass "Smoketest gate: step_100 and step_200 eval results confirmed."

    if [[ "$SMOKE_ONLY" == "1" ]]; then
        echo ""
        echo "SMOKE_ONLY=1 — smoke phases passed. Stopping before full sweep."
        echo "  Smoketest artifact : ${SMOKE_ARTIFACT}"
        echo "  Eval results       : ${SMOKE_RESULTS_BASE}/step_*_${SMOKE_DATASET}/"
        echo ""
        echo "To launch the full sweep:"
        echo "  GPUS=\"0 1\" bash scripts/experiments/run_chronos2_sweep.sh"
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Phase 3: Full sweep training (parallel GPUs)
# ---------------------------------------------------------------------------
phase 3 "Full sweep training (configs 00–09, parallel GPUs)"

SWEEP_LOG="${LOG_DIR}/chronos2_sweep_train_$(date +%Y%m%d_%H%M%S).log"
echo "  Logging to: ${SWEEP_LOG}"
echo "  (tail -f ${SWEEP_LOG} to monitor)"
echo ""

if GPUS="${GPUS:-}" bash "$TRAIN_SCRIPT" 2>&1 | tee "$SWEEP_LOG"; then
    pass "Full sweep training complete."
else
    fail "Full sweep training reported failures — check ${SWEEP_LOG}."
fi

echo ""
echo "================================================================"
echo "  All phases complete  $(date)"
echo ""
echo "  Next step — run checkpoint evals:"
echo "    For each run in trained_models/artifacts/chronos2/sweep_manifest.txt:"
echo "    bash scripts/experiments/run_sweep03_checkpoint_evals.sh <artifact_dir>"
echo ""
echo "  Then visualise:"
echo "    python scripts/visualization/plot_checkpoint_rmse_vs_steps.py"
echo "================================================================"
