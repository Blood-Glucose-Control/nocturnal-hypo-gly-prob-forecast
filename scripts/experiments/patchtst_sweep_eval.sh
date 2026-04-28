#!/usr/bin/env bash
# patchtst_sweep_eval.sh
#
# Evaluates all PatchTST sweep configs (fine-tuned only — no zero-shot for
# trained-from-scratch models). JOBS_PER_GPU parallel workers per GPU.
#
# Checkpoint paths are read from the manifest written by patchtst_sweep_train.sh:
#   trained_models/artifacts/patchtst/sweep_manifest.txt
#
# Usage:
#   bash scripts/experiments/patchtst_sweep_eval.sh
#   GPUS="0 1" JOBS_PER_GPU=2 bash scripts/experiments/patchtst_sweep_eval.sh 2>&1 | tee patchtst_sweep_eval.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
DATASETS_WITH_IOB=(lynch_2022 aleppo_2017 brown_2019)
MANIFEST="trained_models/artifacts/patchtst/sweep_manifest.txt"
LOG_DIR="logs"

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
echo "=== PatchTST sweep eval  $(date) ==="
echo "  GPUs: ${GPUS_ARRAY[*]}  (${N_GPUS} total)"
echo "  Venv: ${PYTHON}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run patchtst_sweep_train.sh first."
    exit 1
fi

# Format: "stem|ctx_len|cov_cols|datasets_key"
CONFIG_META=(
    "00_bg_only|512||ALL"
    "01_bg_iob|512|iob|IOB"
    "02_bg_iob_high_lr|512|iob|IOB"
    "03_bg_iob_short_ctx|256|iob|IOB"
)

mkdir -p "$LOG_DIR"
DONE_FILE="${LOG_DIR}/patchtst_eval_done.log"
touch "$DONE_FILE"

# ---------------------------------------------------------------------------
# Build flat job list: "stem|ctx_len|cov_cols|dataset|checkpoint"
# ---------------------------------------------------------------------------
JOBS=()
for entry in "${CONFIG_META[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols datasets_key <<< "$entry"

    if [[ "$datasets_key" == "IOB" ]]; then
        ft_datasets=("${DATASETS_WITH_IOB[@]}")
    else
        ft_datasets=("${ALL_DATASETS[@]}")
    fi

    out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"
    checkpoint=""
    if [[ -n "$out_dir" ]]; then
        checkpoint="${PROJECT_ROOT}/${out_dir}"
    fi

    for dataset in "${ft_datasets[@]}"; do
        if [[ -z "$out_dir" ]]; then
            JOBS+=("${stem}|${ctx_len}|${cov_cols}|${dataset}|MISSING")
        elif [[ ! -d "$checkpoint" ]]; then
            JOBS+=("${stem}|${ctx_len}|${cov_cols}|${dataset}|MISSING")
        else
            JOBS+=("${stem}|${ctx_len}|${cov_cols}|${dataset}|${checkpoint}")
        fi
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Distribute jobs across GPU slots
# ---------------------------------------------------------------------------
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
N_SLOTS=$(( N_GPUS * JOBS_PER_GPU ))
echo "  Jobs per GPU: ${JOBS_PER_GPU}  (${N_SLOTS} total slots)"
echo ""

declare -A SLOT_CONFIGS
declare -A SLOT_GPU
for (( slot=0; slot<N_SLOTS; slot++ )); do
    SLOT_CONFIGS[$slot]=""
    SLOT_GPU[$slot]="${GPUS_ARRAY[$(( slot % N_GPUS ))]}"
done

for i in "${!JOBS[@]}"; do
    slot=$(( i % N_SLOTS ))
    SLOT_CONFIGS[$slot]+="${JOBS[$i]}"$'\n'
done

echo "Job distribution:"
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    n=$(echo "${SLOT_CONFIGS[$slot]}" | grep -c '|' || true)
    echo "  GPU ${gpu} worker ${slot}: ${n} jobs"
done
echo ""

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------
run_gpu_worker() {
    local gpu="$1"
    local slot="$2"
    local jobs_block="$3"
    local label="GPU${gpu}w${slot}"
    local pass=0 fail=0
    local failed=()

    echo "[${label}] Starting at $(date)"

    while IFS= read -r job; do
        [[ -z "$job" ]] && continue
        IFS='|' read -r stem ctx_len cov_cols dataset checkpoint <<< "$job"

        local model_config="configs/models/patchtst/${stem}.yaml"
        local data_config="${CONFIG_DIR}/${dataset}.yaml"
        local done_key="${stem}|${dataset}"

        if grep -qxF "$done_key" "$DONE_FILE" 2>/dev/null; then
            echo "[${label}] [SKIP] ${done_key} — already done"
            pass=$(( pass + 1 ))
            continue
        fi

        if [[ "$checkpoint" == "MISSING" ]]; then
            echo "[${label}] [SKIP] ${done_key} — checkpoint missing"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
            continue
        fi

        if [[ ! -f "$data_config" ]]; then
            echo "[${label}] [SKIP] ${done_key} — data config not found: ${data_config}"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
            continue
        fi

        echo ""
        echo "[${label}] ============================================"
        echo "[${label}]  Eval: patchtst / ${stem} / ${dataset}"
        echo "[${label}]  Checkpoint: ${checkpoint}"
        echo "[${label}] ============================================"

        eval_args=(
            --probabilistic
            --model-config "$model_config"
            --data-config "$data_config"
            --checkpoint "$checkpoint"
        )
        if [[ -n "$cov_cols" ]]; then
            # shellcheck disable=SC2206
            eval_args+=(--covariate-cols $cov_cols)
        fi

        if CUDA_VISIBLE_DEVICES="$gpu" \
           "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py "${eval_args[@]}"; then
            echo "[${label}] [OK] ${done_key}"
            echo "$done_key" >> "$DONE_FILE"
            pass=$(( pass + 1 ))
        else
            echo "[${label}] [FAIL] ${done_key}"
            fail=$(( fail + 1 ))
            failed+=("${done_key}")
        fi
    done <<< "$jobs_block"

    echo ""
    echo "[${label}] Done at $(date)  —  passed: ${pass} / $(( pass + fail ))"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "[${label}] Failed: ${failed[*]}"
        return 1
    fi
}

export -f run_gpu_worker
export PYTHON CONFIG_DIR DONE_FILE

declare -A PIDS
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    log_file="${LOG_DIR}/patchtst_eval_gpu${gpu}_w${slot}.log"
    echo "Launching GPU ${gpu} worker ${slot} → ${log_file}"
    run_gpu_worker "$gpu" "$slot" "${SLOT_CONFIGS[$slot]}" \
        > "$log_file" 2>&1 &
    PIDS[$slot]=$!
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(tail -f logs/patchtst_eval_gpu<N>_w<slot>.log to monitor)"
echo ""

OVERALL_FAIL=0
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    pid="${PIDS[$slot]}"
    if wait "$pid"; then
        echo "GPU ${gpu} worker ${slot}: SUCCESS"
    else
        echo "GPU ${gpu} worker ${slot}: FAILED  (see ${LOG_DIR}/patchtst_eval_gpu${gpu}_w${slot}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== PatchTST sweep eval complete  $(date) ==="
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} GPU worker(s) reported failures."
    exit 1
fi
echo "  All configs evaluated successfully."
