#!/usr/bin/env bash
# chronos2_step_sweep_eval.sh
#
# Evaluates IOB configs (04, 10, 11, 12) at intermediate training step
# checkpoints (2000, 4000, 6000, 8000) plus the final 10k checkpoint,
# across the three IOB datasets: lynch_2022, aleppo_2017, brown_2019.
#
# Zero-shot (step 0) results already exist in
#   experiments/nocturnal_forecasting_ctx_ablation/
# and are re-used by the plotting script — no re-evaluation needed.
#
# All runs use --episode-context-length 512 (fair fixed-episode set, same as
# the ctx-ablation experiment) so results are directly comparable across
# the four context-length configs.
#
# Results land in:
#   experiments/nocturnal_forecasting_step_sweep/{ctx}ctx_96fh/chronos2/...
# Done log:
#   logs/chronos2_step_sweep_done.log
#
# Usage:
#   bash scripts/experiments/chronos2_step_sweep_eval.sh
#   GPUS="0 1" bash scripts/experiments/chronos2_step_sweep_eval.sh
#   GPUS="0 1" JOBS_PER_GPU=4 bash scripts/experiments/chronos2_step_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
EVAL_SCRIPT="scripts/experiments/nocturnal_hypo_eval_ctx_ablation.py"
CONFIG_DIR="configs/data/holdout_10pct"
DATASETS=(lynch_2022 aleppo_2017 brown_2019)
MANIFEST="trained_models/artifacts/chronos2/sweep_manifest.txt"
LOG_DIR="logs"
DONE_FILE="${LOG_DIR}/chronos2_step_sweep_done.log"

EPISODE_CTX=512
# Steps to evaluate (snapshots directory names).
# Step 0 (zero-shot) is not evaluated here — use ctx-ablation results.
# Step 10000 is the same as model.pt; snapshot path is used for consistency.
STEPS=(2000 4000 6000 8000 10000)

# IOB configs: stem | ctx | covariate cols
CONFIG_META=(
    "04_bg_iob_ia_high_lr|512|iob insulin_availability"
    "10_bg_iob_ia_high_lr_ctx256|256|iob insulin_availability"
    "11_bg_iob_ia_high_lr_ctx128|128|iob insulin_availability"
    "12_bg_iob_ia_high_lr_ctx64|64|iob insulin_availability"
)

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
echo "=== Chronos-2 step-sweep eval  $(date) ==="
echo "  GPUs: ${GPUS_ARRAY[*]}  (${N_GPUS} total)"
echo "  Steps: ${STEPS[*]}"
echo "  Episode context length (filter): ${EPISODE_CTX}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: chronos2 venv not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    exit 1
fi

mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

# ---------------------------------------------------------------------------
# Build flat job list
# Format: "stem|ctx_len|cov_cols|dataset|checkpoint|step"
# ---------------------------------------------------------------------------
JOBS=()

for entry in "${CONFIG_META[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols <<< "$entry"

    out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { d=$2 } END { print d }' "$MANIFEST")"
    if [[ -z "$out_dir" ]]; then
        echo "[WARN] $stem — not in manifest, skipping"
        continue
    fi

    for step in "${STEPS[@]}"; do
        checkpoint="${PROJECT_ROOT}/${out_dir}/snapshots/step_${step}/model.pt"
        if [[ ! -d "$checkpoint" ]]; then
            echo "[WARN] $stem step_${step} — checkpoint dir missing: $checkpoint"
            continue
        fi

        for dataset in "${DATASETS[@]}"; do
            JOBS+=("${stem}|${ctx_len}|${cov_cols}|${dataset}|${checkpoint}|${step}")
        done
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Distribute round-robin across slots
# ---------------------------------------------------------------------------
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
N_SLOTS=$(( N_GPUS * JOBS_PER_GPU ))
echo "  Jobs per GPU: ${JOBS_PER_GPU}  (${N_SLOTS} total slots)"

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
        IFS='|' read -r stem ctx_len cov_cols dataset checkpoint step <<< "$job"

        local model_config="configs/models/chronos2/${stem}.yaml"
        local done_key="${stem}|step${step}|${dataset}"
        local job_label="chronos2 / ${stem} / step${step} / ${dataset}"

        if grep -qxF "$done_key" "$DONE_FILE" 2>/dev/null; then
            echo "[${label}] [SKIP] ${done_key} — already done"
            pass=$(( pass + 1 ))
            continue
        fi

        echo ""
        echo "[${label}] ============================================"
        echo "[${label}]  ${job_label}"
        echo "[${label}]  ctx=${ctx_len}  episode_ctx=${EPISODE_CTX}  step=${step}"
        echo "[${label}]  Checkpoint: ${checkpoint}"
        echo "[${label}] ============================================"

        local timestamp
        timestamp="$(date +%Y-%m-%d_%H%M%S)"
        local out_dir="${PROJECT_ROOT}/experiments/nocturnal_forecasting_step_sweep/${ctx_len}ctx_96fh/chronos2/${timestamp}_${dataset}_step${step}"

        CMD=(
            "$PYTHON" "$EVAL_SCRIPT"
            --model chronos2
            --model-config "$model_config"
            --dataset "$dataset"
            --config-dir "$CONFIG_DIR"
            --checkpoint "$checkpoint"
            --context-length "$ctx_len"
            --episode-context-length "$EPISODE_CTX"
            --forecast-length 96
            --cuda-device "$gpu"
            --probabilistic
            --output-dir "$out_dir"
        )

        if [[ -n "$cov_cols" ]]; then
            # shellcheck disable=SC2086
            CMD+=(--covariate-cols $cov_cols)
        fi

        if "${CMD[@]}"; then
            echo "[${label}] [OK] ${job_label}"
            echo "$done_key" >> "$DONE_FILE"
            pass=$(( pass + 1 ))
        else
            echo "[${label}] [FAIL] ${job_label}"
            fail=$(( fail + 1 ))
            failed+=("${job_label}")
        fi
    done <<< "$jobs_block"

    echo ""
    echo "[${label}] Done at $(date)  —  passed: ${pass} / $(( pass + fail ))"
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "[${label}] Failed:"
        for f in "${failed[@]}"; do
            echo "[${label}]   - $f"
        done
        return 1
    fi
}

export -f run_gpu_worker
export PYTHON EVAL_SCRIPT CONFIG_DIR DONE_FILE EPISODE_CTX PROJECT_ROOT

# ---------------------------------------------------------------------------
# Launch one background process per slot
# ---------------------------------------------------------------------------
declare -A PIDS
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    log_file="${LOG_DIR}/chronos2_step_sweep_gpu${gpu}_w${slot}.log"
    echo "Launching GPU ${gpu} worker ${slot} → ${log_file}"
    run_gpu_worker "$gpu" "$slot" "${SLOT_CONFIGS[$slot]}" \
        > "$log_file" 2>&1 &
    PIDS[$slot]=$!
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(tail -f logs/chronos2_step_sweep_gpu<N>_w<slot>.log to monitor)"
echo ""

# ---------------------------------------------------------------------------
# Wait and collect results
# ---------------------------------------------------------------------------
OVERALL_FAIL=0
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    pid="${PIDS[$slot]}"
    if wait "$pid"; then
        echo "GPU ${gpu} worker ${slot}: SUCCESS"
    else
        echo "GPU ${gpu} worker ${slot}: FAILED  (see ${LOG_DIR}/chronos2_step_sweep_gpu${gpu}_w${slot}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== Chronos-2 step-sweep eval complete  $(date) ==="
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} GPU worker(s) reported failures — check logs above."
    exit 1
fi
echo "  All jobs passed."
