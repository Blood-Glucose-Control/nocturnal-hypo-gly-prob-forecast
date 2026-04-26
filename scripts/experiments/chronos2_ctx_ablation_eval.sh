#!/usr/bin/env bash
# chronos2_ctx_ablation_eval.sh
#
# Context-window ablation evaluation for Chronos-2.
#
# Evaluates configs 04, 08, and 10-15 (plus zero-shot at all four context
# lengths) using nocturnal_hypo_eval_ctx_ablation.py, which filters episodes
# to only those valid at --episode-context-length 512 steps. This gives every
# context length (64/128/256/512) the exact same set of midnight anchors, so
# results are directly comparable without the confound of smaller windows
# qualifying more episodes from shorter patient sequences.
#
# Results land in:
#   experiments/nocturnal_forecasting_ctx_ablation/{ctx}ctx_96fh/chronos2/...
# Done log:
#   logs/chronos2_ctx_ablation_done.log
#
# Configs evaluated:
#   Fine-tuned:
#     IOB group  : 04 (ctx=512), 10 (256), 11 (128), 12 (64)  — lynch/aleppo/brown
#     COB group  : 08 (ctx=512), 13 (256), 14 (128), 15 (64)  — aleppo only
#   Zero-shot    : all four ctx lengths × all 4 datasets
#
# Usage:
#   bash scripts/experiments/chronos2_ctx_ablation_eval.sh
#   GPUS="0 1" bash scripts/experiments/chronos2_ctx_ablation_eval.sh
#   GPUS="0 1" JOBS_PER_GPU=4 bash scripts/experiments/chronos2_ctx_ablation_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
EVAL_SCRIPT="scripts/experiments/nocturnal_hypo_eval_ctx_ablation.py"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
DATASETS_WITH_IOB=(lynch_2022 aleppo_2017 brown_2019)
DATASETS_WITH_COB=(aleppo_2017)
MANIFEST="trained_models/artifacts/chronos2/sweep_manifest.txt"
LOG_DIR="logs"
DONE_FILE="${LOG_DIR}/chronos2_ctx_ablation_done.log"

# Episode context length used for all runs — ensures identical anchors.
EPISODE_CTX=512

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
echo "=== Chronos-2 ctx-ablation eval  $(date) ==="
echo "  GPUs: ${GPUS_ARRAY[*]}  (${N_GPUS} total)"
echo "  Episode context length (filter): ${EPISODE_CTX}"
echo ""

# Format: "stem|context_length|covariate cols (space-sep, empty=BG only)|datasets_key"
# datasets_key: ALL, IOB, or COB
CONFIG_META=(
    # --- IOB group: ctx=512 baseline + ablations 256/128/64 ---
    "04_bg_iob_ia_high_lr|512|iob insulin_availability|IOB"
    "10_bg_iob_ia_high_lr_ctx256|256|iob insulin_availability|IOB"
    "11_bg_iob_ia_high_lr_ctx128|128|iob insulin_availability|IOB"
    "12_bg_iob_ia_high_lr_ctx64|64|iob insulin_availability|IOB"
    # --- COB group: ctx=512 baseline + ablations 256/128/64 ---
    "08_bg_iob_cob_high_lr|512|iob cob|COB"
    "13_bg_iob_cob_high_lr_ctx256|256|iob cob|COB"
    "14_bg_iob_cob_high_lr_ctx128|128|iob cob|COB"
    "15_bg_iob_cob_high_lr_ctx64|64|iob cob|COB"
)

# Zero-shot context ablation: pre-trained model, no checkpoint, all 4 ctx levels.
# Use --covariate-cols iob insulin_availability to match config 04 feature set
# (zero-shot ignores unknown covariates, so results are identical regardless,
# but we keep covariate args for consistency with the fine-tuned runs).
ZS_CTXS=(512 256 128 64)

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: chronos2 venv not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run chronos2_sweep_train.sh first."
    exit 1
fi

mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

# ---------------------------------------------------------------------------
# Build flat job list
# Format: "mode|stem|ctx_len|cov_cols|dataset|checkpoint"
# ---------------------------------------------------------------------------
JOBS=()

for entry in "${CONFIG_META[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols datasets_key <<< "$entry"

    if [[ "$datasets_key" == "IOB" ]]; then
        ft_datasets=("${DATASETS_WITH_IOB[@]}")
    elif [[ "$datasets_key" == "COB" ]]; then
        ft_datasets=("${DATASETS_WITH_COB[@]}")
    else
        ft_datasets=("${ALL_DATASETS[@]}")
    fi

    out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"
    checkpoint=""
    if [[ -n "$out_dir" ]]; then
        checkpoint="${PROJECT_ROOT}/${out_dir}/model.pt"
    fi

    for dataset in "${ft_datasets[@]}"; do
        if [[ -z "$out_dir" ]]; then
            echo "[WARN] $stem — not in manifest, fine-tuned jobs will be skipped"
            JOBS+=("finetuned|${stem}|${ctx_len}|${cov_cols}|${dataset}|MISSING")
        elif [[ ! -d "$checkpoint" ]]; then
            echo "[WARN] $stem — checkpoint dir missing: $checkpoint"
            JOBS+=("finetuned|${stem}|${ctx_len}|${cov_cols}|${dataset}|MISSING")
        else
            JOBS+=("finetuned|${stem}|${ctx_len}|${cov_cols}|${dataset}|${checkpoint}")
        fi
    done
done

# Zero-shot jobs: 4 ctx levels × 4 datasets
for ctx_len in "${ZS_CTXS[@]}"; do
    for dataset in "${ALL_DATASETS[@]}"; do
        # Label zero-shot by ctx so done-log entries are unique
        JOBS+=("zeroshot|zs_ctx${ctx_len}|${ctx_len}|iob insulin_availability|${dataset}|")
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Distribute jobs round-robin across slots (JOBS_PER_GPU workers per GPU)
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
        IFS='|' read -r mode stem ctx_len cov_cols dataset checkpoint <<< "$job"

        local done_key="${stem}|${dataset}|${mode}"

        if grep -qxF "$done_key" "$DONE_FILE" 2>/dev/null; then
            echo "[${label}] [SKIP] ${done_key} — already in done log"
            pass=$(( pass + 1 ))
            continue
        fi

        if [[ "$mode" == "finetuned" ]]; then
            local model_config="configs/models/chronos2/${stem}.yaml"
            local job_label="chronos2 / ${stem} / ${dataset}"

            if [[ "$checkpoint" == "MISSING" ]]; then
                echo "[${label}] [SKIP] ${job_label} — checkpoint missing"
                fail=$(( fail + 1 ))
                failed+=("${job_label} (missing checkpoint)")
                continue
            fi

            echo ""
            echo "[${label}] ============================================"
            echo "[${label}]  Fine-tuned ctx-ablation eval: ${job_label}"
            echo "[${label}]  ctx=${ctx_len}  episode_ctx=${EPISODE_CTX}"
            echo "[${label}]  Checkpoint: ${checkpoint}"
            echo "[${label}] ============================================"

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
            )
        else
            local job_label="chronos2-zeroshot / ctx${ctx_len} / ${dataset}"

            echo ""
            echo "[${label}] ============================================"
            echo "[${label}]  Zero-shot ctx-ablation eval: ${job_label}"
            echo "[${label}]  episode_ctx=${EPISODE_CTX}"
            echo "[${label}] ============================================"

            # Use the IOB model config for zero-shot (arbitrary — ZS ignores covariates)
            local model_config="configs/models/chronos2/04_bg_iob_ia_high_lr.yaml"

            CMD=(
                "$PYTHON" "$EVAL_SCRIPT"
                --model chronos2
                --model-config "$model_config"
                --dataset "$dataset"
                --config-dir "$CONFIG_DIR"
                --context-length "$ctx_len"
                --episode-context-length "$EPISODE_CTX"
                --forecast-length 96
                --cuda-device "$gpu"
                --probabilistic
            )
        fi

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
export PYTHON EVAL_SCRIPT CONFIG_DIR DONE_FILE EPISODE_CTX

# ---------------------------------------------------------------------------
# Launch one background process per slot
# ---------------------------------------------------------------------------
declare -A PIDS
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    log_file="${LOG_DIR}/chronos2_ctx_ablation_gpu${gpu}_w${slot}.log"
    echo "Launching GPU ${gpu} worker ${slot} → ${log_file}"
    run_gpu_worker "$gpu" "$slot" "${SLOT_CONFIGS[$slot]}" \
        > "$log_file" 2>&1 &
    PIDS[$slot]=$!
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(tail -f logs/chronos2_ctx_ablation_gpu<N>_w<slot>.log to monitor)"
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
        echo "GPU ${gpu} worker ${slot}: FAILED  (see ${LOG_DIR}/chronos2_ctx_ablation_gpu${gpu}_w${slot}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== Chronos-2 ctx-ablation eval complete  $(date) ==="
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} GPU worker(s) reported failures — check logs above."
    exit 1
fi
echo "  All jobs passed."
