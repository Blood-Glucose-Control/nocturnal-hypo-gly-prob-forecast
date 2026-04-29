#!/usr/bin/env bash
# tft_sweep_train.sh
#
# Trains all TFT sweep configs in parallel, JOBS_PER_GPU workers per GPU.
# Configs are distributed round-robin across all GPU slots.
#
# Sweep: 12 configs total — 6 BG-only (00–05, all four datasets) and
# 6 BG+IOB (06–11, IOB datasets only). TFT is the only deep model in the
# stack that can consume past covariates.
#
# Manifest: trained_models/artifacts/tft/sweep_manifest.txt
#
# GPU memory: ~8 GB peak at defaults → 6 workers per 96 GB Blackwell GPU is safe.
#
# Usage:
#   bash scripts/experiments/tft_sweep_train.sh
#   GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/tft_sweep_train.sh
#   GPUS="0 1" JOBS_PER_GPU=6 bash scripts/experiments/tft_sweep_train.sh 2>&1 | tee tft_sweep_train.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
if [[ -n "${GPUS:-}" ]]; then
    read -ra GPUS_ARRAY <<< "$GPUS"
else
    mapfile -t GPUS_ARRAY < <(
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo "0"
    )
fi

N_GPUS=${#GPUS_ARRAY[@]}
echo "=== TFT sweep training  $(date) ==="
echo "  GPUs: ${GPUS_ARRAY[*]}  (${N_GPUS} total)"
echo "  Venv: ${PYTHON}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found at $PYTHON"
    exit 1
fi

# Smoke-test imports
"$PYTHON" - <<'EOF'
from autogluon.timeseries.models import TemporalFusionTransformerModel
print("AutoGluon TFT import OK")
EOF
echo ""

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
DATASETS_ALL="aleppo_2017 brown_2019 lynch_2022 tamborlane_2008"
DATASETS_WITH_IOB="aleppo_2017 brown_2019 lynch_2022"

CONFIGS=(
    "configs/models/tft/00_bg_baseline.yaml|ALL"
    "configs/models/tft/01_bg_wide.yaml|ALL"
    "configs/models/tft/02_bg_long_ctx.yaml|ALL"
    "configs/models/tft/03_bg_high_dropout.yaml|ALL"
    "configs/models/tft/04_bg_more_heads.yaml|ALL"
    "configs/models/tft/05_bg_high_lr.yaml|ALL"
    "configs/models/tft/06_iob_baseline.yaml|IOB"
    "configs/models/tft/07_iob_wide.yaml|IOB"
    "configs/models/tft/08_iob_long_ctx.yaml|IOB"
    "configs/models/tft/09_iob_high_dropout.yaml|IOB"
    "configs/models/tft/10_iob_more_heads.yaml|IOB"
    "configs/models/tft/11_iob_high_lr.yaml|IOB"
)

CONFIG_DIR="configs/data/holdout_10pct"
WORKFLOW="scripts/examples/run_holdout_generic_workflow.sh"
MANIFEST_DIR="trained_models/artifacts/tft"
SKIP_STEPS="${SKIP_STEPS:-1 2 4 7}"
MANIFEST="${MANIFEST_DIR}/sweep_manifest.txt"
LOG_DIR="logs"

mkdir -p "$MANIFEST_DIR" "$LOG_DIR"
touch "$MANIFEST"

# ---------------------------------------------------------------------------
# Distribute configs round-robin across slots
# ---------------------------------------------------------------------------
JOBS_PER_GPU="${JOBS_PER_GPU:-6}"
N_SLOTS=$(( N_GPUS * JOBS_PER_GPU ))
echo "  Jobs per GPU: ${JOBS_PER_GPU}  (${N_SLOTS} total slots)"
echo ""

declare -A SLOT_CONFIGS
declare -A SLOT_GPU
for (( slot=0; slot<N_SLOTS; slot++ )); do
    SLOT_CONFIGS[$slot]=""
    SLOT_GPU[$slot]="${GPUS_ARRAY[$(( slot % N_GPUS ))]}"
done

for i in "${!CONFIGS[@]}"; do
    slot=$(( i % N_SLOTS ))
    SLOT_CONFIGS[$slot]+="${CONFIGS[$i]}"$'\n'
done

echo "Config distribution:"
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    n=$(echo "${SLOT_CONFIGS[$slot]}" | grep -c '|' || true)
    echo "  GPU ${gpu} worker ${slot}: ${n} configs"
done
echo ""

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------
run_gpu_worker() {
    local gpu="$1"
    local slot="$2"
    local configs_block="$3"
    local label="GPU${gpu}w${slot}"
    local pass=0 fail=0
    local failed=()

    echo "[${label}] Starting at $(date)"

    while IFS= read -r entry; do
        [[ -z "$entry" ]] && continue
        IFS='|' read -r config datasets_key <<< "$entry"
        local stem
        stem="$(basename "$config" .yaml)"

        # Skip if already in manifest
        local existing_dir
        existing_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST" 2>/dev/null || true)"
        if [[ -n "$existing_dir" && -d "${existing_dir}" ]]; then
            echo "[${label}] [SKIP] ${stem} — already in manifest"
            pass=$(( pass + 1 ))
            continue
        fi

        local datasets
        if [[ "$datasets_key" == "IOB" ]]; then
            datasets="$DATASETS_WITH_IOB"
        else
            datasets="$DATASETS_ALL"
        fi

        local run_id
        run_id="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
        local out_dir="${MANIFEST_DIR}/$(date +%Y-%m-%d_%H%M)_RID${run_id}_${stem}"

        echo ""
        echo "[${label}] ============================================"
        echo "[${label}]  Training: tft / ${stem}"
        echo "[${label}]  Config:   ${config}"
        echo "[${label}]  Datasets: ${datasets}"
        echo "[${label}]  Output:   ${out_dir}"
        echo "[${label}] ============================================"

        if CUDA_VISIBLE_DEVICES="$gpu" \
           MODEL_TYPE="tft" \
           VENV_NAME="chronos2" \
           MODEL_CONFIG="$config" \
           CONFIG_DIR="$CONFIG_DIR" \
           DATASETS="$datasets" \
           SKIP_TRAINING="false" \
           SKIP_STEPS="$SKIP_STEPS" \
           OUTPUT_BASE_DIR="$out_dir" \
           RUN_ID="$run_id" \
           "$WORKFLOW"; then
            echo "[${label}] [OK] ${stem}"
            echo "${stem}"$'\t'"${out_dir}" >> "$MANIFEST"
            pass=$(( pass + 1 ))
        else
            echo "[${label}] [FAIL] ${stem}"
            fail=$(( fail + 1 ))
            failed+=("${stem}")
        fi
    done <<< "$configs_block"

    echo ""
    echo "[${label}] Done at $(date)  —  passed: ${pass} / $(( pass + fail ))"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "[${label}] Failed: ${failed[*]}"
        return 1
    fi
}

export -f run_gpu_worker
export DATASETS_ALL DATASETS_WITH_IOB MANIFEST_DIR MANIFEST CONFIG_DIR WORKFLOW SKIP_STEPS

# ---------------------------------------------------------------------------
# Launch workers
# ---------------------------------------------------------------------------
declare -A PIDS
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    log_file="${LOG_DIR}/tft_sweep_gpu${gpu}_w${slot}.log"
    echo "Launching GPU ${gpu} worker ${slot} → ${log_file}"
    run_gpu_worker "$gpu" "$slot" "${SLOT_CONFIGS[$slot]}" \
        > "$log_file" 2>&1 &
    PIDS[$slot]=$!
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(tail -f logs/tft_sweep_gpu<N>_w<slot>.log to monitor)"
echo ""

OVERALL_FAIL=0
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    pid="${PIDS[$slot]}"
    if wait "$pid"; then
        echo "GPU ${gpu} worker ${slot}: SUCCESS"
    else
        echo "GPU ${gpu} worker ${slot}: FAILED  (see ${LOG_DIR}/tft_sweep_gpu${gpu}_w${slot}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== TFT sweep training complete  $(date) ==="
echo "  Manifest: ${MANIFEST}"
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} GPU worker(s) reported failures."
    exit 1
fi
echo "  All configs trained successfully."
