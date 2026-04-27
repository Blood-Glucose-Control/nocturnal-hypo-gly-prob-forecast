#!/usr/bin/env bash
# timesfm_sweep_train.sh
#
# Fine-tunes all TimesFM sweep configs in two parallel lanes:
#   - Lane A (GPU 0): configs matching 00_*.yaml (long baseline; runs solo)
#   - Lane B (GPU 1): configs matching 01_*.yaml and above (shorter variants)
#
# GPU lane assignment is name-based, not alternating:
#   00_long_run.yaml   → GPU 0  (35 ep, long pole)
#   01_lr_2e5.yaml     → GPU 1  }
#   02_lr_5e5.yaml     → GPU 1  } ~31 ep total, balanced with lane A
#   03_dilate_*        → GPU 1  }
#   04_dilate_*        → GPU 1  }
#
# Configs are discovered from configs/models/timesfm/[0-9][0-9]_*.yaml, sorted.
#
# Default skip behavior is optimized for sweeps:
#   SKIP_STEPS="1 2 4 6 7"
# This skips holdout config regeneration/validation, zero-shot eval,
# checkpoint reload verification, and resume-training.
# Step 5 (fine-tuning) is still executed.
# Override with e.g. SKIP_STEPS="7" for fuller workflow checks.
#
# TimesFM has no covariate support — all four datasets are always used.
#
# A manifest is appended at:
#   trained_models/artifacts/timesfm/sweep_manifest.txt
# Format:
#   <stem>\t<output_dir>
#
# Usage:
#   bash scripts/experiments/timesfm_sweep_train.sh
#   GPU0=0 GPU1=1 bash scripts/experiments/timesfm_sweep_train.sh
#   DATASETS="aleppo_2017" bash scripts/experiments/timesfm_sweep_train.sh
#
# Single-GPU mode (one GPU occupied / testing):
#   SINGLE_GPU=1 bash scripts/experiments/timesfm_sweep_train.sh
#   SINGLE_GPU=1 GPU0=1 bash scripts/experiments/timesfm_sweep_train.sh
# All configs run sequentially on GPU0; no background processes are spawned.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SINGLE_GPU="${SINGLE_GPU:-0}"
REQUESTED_DATASETS_STR="${DATASETS:-lynch_2022 aleppo_2017 brown_2019 tamborlane_2008}"
CONFIG_DIR="${CONFIG_DIR:-configs/data/holdout_10pct}"
WORKFLOW="scripts/examples/run_holdout_generic_workflow.sh"
# Faster sweep default: keep only core data load + fine-tuning path.
SKIP_STEPS="${SKIP_STEPS:-1 2 4 6 7}"
ARTIFACT_DIR="trained_models/artifacts/timesfm"
MANIFEST="${ARTIFACT_DIR}/sweep_manifest.txt"

mkdir -p "$ARTIFACT_DIR"
touch "$MANIFEST"

IFS=' ' read -r -a REQUESTED_DATASETS <<< "$REQUESTED_DATASETS_STR"
if [[ ${#REQUESTED_DATASETS[@]} -eq 0 ]]; then
    echo "ERROR: DATASETS is empty"
    exit 1
fi

mapfile -t ALL_CONFIGS < <(ls configs/models/timesfm/[0-9][0-9]_*.yaml 2>/dev/null | sort)
if [[ ${#ALL_CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: no TimesFM sweep configs found at configs/models/timesfm/[0-9][0-9]_*.yaml"
    exit 1
fi

CONFIGS_GPU0=()
CONFIGS_GPU1=()
for cfg in "${ALL_CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    if [[ "$stem" == 00_* ]]; then
        CONFIGS_GPU0+=("$cfg")
    else
        CONFIGS_GPU1+=("$cfg")
    fi
done

if [[ ${#CONFIGS_GPU0[@]} -eq 0 ]]; then
    echo "WARNING: no 00_*.yaml config found — GPU ${GPU0} lane will be empty"
fi

run_train_lane() {
    local lane_name="$1"
    local gpu="$2"
    local lane_manifest="$3"
    shift 3
    local configs=("$@")

    : > "$lane_manifest"

    local pass=0
    local fail=0
    local failed=()

    echo ""
    echo "============================================================"
    echo "  Starting lane: ${lane_name} (GPU ${gpu})"
    echo "  Config count: ${#configs[@]}"
    echo "============================================================"

    for config in "${configs[@]}"; do
        local stem
        stem="$(basename "$config" .yaml)"

        local run_id
        run_id="$(date +%Y%m%d_%H%M%S)_${lane_name}_${RANDOM}"
        local out_dir
        out_dir="${ARTIFACT_DIR}/$(date +%Y-%m-%d_%H:%M)_RID${run_id}_holdout_workflow"

        echo ""
        echo "------------------------------------------------------------"
        echo "  Lane:     ${lane_name}"
        echo "  Training: timesfm / ${stem}"
        echo "  Config:   ${config}"
        echo "  Datasets: ${REQUESTED_DATASETS_STR}"
        echo "  Output:   ${out_dir}"
        echo "------------------------------------------------------------"

        if CUDA_VISIBLE_DEVICES="$gpu" \
           MODEL_TYPE="timesfm" \
           MODEL_CONFIG="$config" \
           CONFIG_DIR="$CONFIG_DIR" \
           DATASETS="$REQUESTED_DATASETS_STR" \
           SKIP_TRAINING="false" \
           SKIP_STEPS="$SKIP_STEPS" \
           OUTPUT_BASE_DIR="$out_dir" \
           RUN_ID="$run_id" \
           EPOCHS="${EPOCHS:-}" \
           BATCH_SIZE="${BATCH_SIZE:-}" \
           "$WORKFLOW"; then
            echo "[OK] ${lane_name}: timesfm / ${stem}"
            echo "${stem}"$'\t'"${out_dir}" >> "$lane_manifest"
            pass=$((pass + 1))
        else
            echo "[FAIL] ${lane_name}: timesfm / ${stem}"
            fail=$((fail + 1))
            failed+=("timesfm / ${stem}")
        fi
    done

    echo ""
    echo "============================================================"
    echo "  Lane complete: ${lane_name}"
    echo "  Passed: ${pass} / $((pass + fail))"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "  Failed in ${lane_name}:"
        for f in "${failed[@]}"; do
            echo "    - ${f}"
        done
    fi
    echo "============================================================"

    [[ $fail -eq 0 ]]
}

LANE0_MANIFEST="${ARTIFACT_DIR}/.sweep_manifest_gpu${GPU0}_$$.txt"
LANE1_MANIFEST="${ARTIFACT_DIR}/.sweep_manifest_gpu${GPU1}_$$.txt"

if [[ "$SINGLE_GPU" == "1" ]]; then
    # Collapse both lanes into one sequential run on GPU0.  No background
    # processes — avoids two training jobs competing for VRAM on the same card.
    # Sort by num_epochs (ascending) so short smoke-tests surface results first.
    mapfile -t SINGLE_GPU_CONFIGS < <(
        for cfg in "${ALL_CONFIGS[@]}"; do
            epochs=$(grep '^num_epochs:' "$cfg" | awk '{print $2}')
            echo "${epochs} ${cfg}"
        done | sort -n | awk '{print $2}'
    )
    echo "SINGLE_GPU=1: running all ${#SINGLE_GPU_CONFIGS[@]} configs sequentially on GPU ${GPU0} (shortest first)"
    for cfg in "${SINGLE_GPU_CONFIGS[@]}"; do
        epochs=$(grep '^num_epochs:' "$cfg" | awk '{print $2}')
        echo "  $(basename "$cfg") — ${epochs} epochs"
    done
    run_train_lane "gpu${GPU0}" "$GPU0" "$LANE0_MANIFEST" "${SINGLE_GPU_CONFIGS[@]}"
    FAIL_LANES=$?
    if [[ -s "$LANE0_MANIFEST" ]]; then
        cat "$LANE0_MANIFEST" >> "$MANIFEST"
    fi
    rm -f "$LANE0_MANIFEST"
else
    run_train_lane "gpu${GPU0}" "$GPU0" "$LANE0_MANIFEST" "${CONFIGS_GPU0[@]}" &
    PID0=$!
    run_train_lane "gpu${GPU1}" "$GPU1" "$LANE1_MANIFEST" "${CONFIGS_GPU1[@]}" &
    PID1=$!

    FAIL_LANES=0
    if ! wait "$PID0"; then
        FAIL_LANES=$((FAIL_LANES + 1))
    fi
    if ! wait "$PID1"; then
        FAIL_LANES=$((FAIL_LANES + 1))
    fi

    for lane_manifest in "$LANE0_MANIFEST" "$LANE1_MANIFEST"; do
        if [[ -s "$lane_manifest" ]]; then
            cat "$lane_manifest" >> "$MANIFEST"
        fi
        rm -f "$lane_manifest"
    done
fi

echo ""
echo "============================================================"
echo "  TimesFM sweep training complete  $(date)"
echo "  Lanes failed: ${FAIL_LANES}"
echo "  Manifest: ${MANIFEST}"
echo "============================================================"

[[ $FAIL_LANES -eq 0 ]]
