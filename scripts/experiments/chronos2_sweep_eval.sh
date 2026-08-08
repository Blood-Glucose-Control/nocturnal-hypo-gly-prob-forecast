#!/usr/bin/env bash
# chronos2_sweep_eval.sh
#
# Runs nocturnal_hypo_eval.py --probabilistic for Chronos-2 sweep configs 00, 06-15
# (configs 01-05 commented out), both fine-tuned and zero-shot, across the
# appropriate dataset subsets. JOBS_PER_GPU concurrent workers per GPU (default 4).
#
#   Fine-tuned:
#     IOB (3 datasets, excl. tamborlane): insulin covariate configs (10–12)
#     COB (1 dataset, aleppo only): carb covariate configs (06–09, 13–15)
#       — lynch carbs are all-zero so it is excluded from COB training and eval
#   Zero-shot: configs 06-15 × all 4 datasets
#
# Checkpoint paths are read from the manifest written by chronos2_sweep_train.sh:
#   trained_models/artifacts/chronos2/sweep_manifest.txt
# Format: <stem>\t<output_dir>   (tab-separated, one entry per config)
# If a stem appears more than once (re-run), the last entry wins.
#
# Usage:
#   bash scripts/experiments/chronos2_sweep_eval.sh
#   GPUS="0 1" bash scripts/experiments/chronos2_sweep_eval.sh
#   GPUS="0 1" JOBS_PER_GPU=4 bash scripts/experiments/chronos2_sweep_eval.sh 2>&1 | tee chronos2_sweep_eval.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/chronos2/bin/python"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
DATASETS_WITH_IOB=(lynch_2022 aleppo_2017 brown_2019)
DATASETS_WITH_COB=(aleppo_2017)  # lynch carbs are all-zero; only aleppo has real meal data
MANIFEST="trained_models/artifacts/chronos2/sweep_manifest.txt"
LOG_DIR="logs"
DONE_FILE="${LOG_DIR}/chronos2_eval_done.log"  # tracks completed stem|dataset|mode

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
echo "=== Chronos-2 sweep eval  $(date) ==="
echo "  GPUs: ${GPUS_ARRAY[*]}  (${N_GPUS} total)"
echo ""

# Format: "stem|context_length|space-separated covariate cols (empty = BG only)|datasets_key"
# datasets_key: ALL, IOB, or COB
# Note: 03_joint_bg_iob passes no --covariate-cols; joint_target_cols is handled inside the YAML.
CONFIG_META=(
    "00_bg_only|512||ALL"
    # "01_bg_iob_insulin_availability|512|iob insulin_availability|IOB"
    # "02_bg_iob|512|iob|IOB"
    # "03_joint_bg_iob|512||IOB"
    # "04_bg_iob_ia_high_lr|512|iob insulin_availability|IOB"
    # "05_bg_iob_short_ctx|288|iob insulin_availability|IOB"
    "06_bg_iob_cob|512|iob cob|COB"
    "07_bg_full_features|512|iob cob insulin_availability carb_availability|COB"
    "08_bg_iob_cob_high_lr|512|iob cob|COB"
    "09_bg_iob_cob_short_ctx|288|iob cob|COB"
    # Context window ablation — high-LR IOB (512 baseline = 04, new: 256/128/64)
    "10_bg_iob_ia_high_lr_ctx256|256|iob insulin_availability|IOB"
    "11_bg_iob_ia_high_lr_ctx128|128|iob insulin_availability|IOB"
    "12_bg_iob_ia_high_lr_ctx64|64|iob insulin_availability|IOB"
    # Context window ablation — high-LR COB (512 baseline = 08, new: 256/128/64)
    "13_bg_iob_cob_high_lr_ctx256|256|iob cob|COB"
    "14_bg_iob_cob_high_lr_ctx128|128|iob cob|COB"
    "15_bg_iob_cob_high_lr_ctx64|64|iob cob|COB"
)

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
touch "$DONE_FILE"  # ensure it exists for grep checks

# ---------------------------------------------------------------------------
# Build flat job list
# Format: "mode|stem|ctx_len|cov_cols|dataset|checkpoint"
# mode: finetuned or zeroshot
# checkpoint: absolute path for finetuned, empty for zeroshot
# ---------------------------------------------------------------------------
JOBS=()

for entry in "${CONFIG_META[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols datasets_key <<< "$entry"

    # --- Fine-tuned jobs ---
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

    # --- Zero-shot jobs (all 4 datasets) ---
    for dataset in "${ALL_DATASETS[@]}"; do
        JOBS+=("zeroshot|${stem}|${ctx_len}|${cov_cols}|${dataset}|")
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
# Worker function — runs all jobs assigned to one slot sequentially.
# Multiple slots may share the same GPU (run concurrently).
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

        local model_config="configs/models/chronos2/${stem}.yaml"
        local done_key="${stem}|${dataset}|${mode}"

        # Skip if already completed in a previous run
        if grep -qxF "$done_key" "$DONE_FILE" 2>/dev/null; then
            echo "[${label}] [SKIP] ${done_key} — already in done log"
            pass=$(( pass + 1 ))
            continue
        fi

        if [[ "$mode" == "finetuned" ]]; then
            local job_label="chronos2 / ${stem} / ${dataset}"

            if [[ "$checkpoint" == "MISSING" ]]; then
                echo "[${label}] [SKIP] ${job_label} — checkpoint missing"
                fail=$(( fail + 1 ))
                failed+=("${job_label} (missing checkpoint)")
                continue
            fi

            echo ""
            echo "[${label}] ============================================"
            echo "[${label}]  Fine-tuned eval: ${job_label}"
            echo "[${label}]  Checkpoint: ${checkpoint}"
            echo "[${label}] ============================================"

            CMD=(
                "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
                --model chronos2
                --model-config "$model_config"
                --dataset "$dataset"
                --config-dir "$CONFIG_DIR"
                --checkpoint "$checkpoint"
                --context-length "$ctx_len"
                --forecast-length 96
                --cuda-device "$gpu"
                --probabilistic
            )
        else
            local job_label="chronos2-zeroshot / ${stem} / ${dataset}"

            echo ""
            echo "[${label}] ============================================"
            echo "[${label}]  Zero-shot eval: ${job_label}"
            echo "[${label}] ============================================"

            CMD=(
                "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
                --model chronos2
                --model-config "$model_config"
                --dataset "$dataset"
                --config-dir "$CONFIG_DIR"
                --context-length "$ctx_len"
                --forecast-length 96
                --cuda-device "$gpu"
                --probabilistic
            )
        fi

        if [[ -n "$cov_cols" ]]; then
            # word-split intentional: cov_cols is space-separated
            # shellcheck disable=SC2086
            CMD+=(--covariate-cols $cov_cols)
        fi

        if "${CMD[@]}"; then
            echo "[${label}] [OK] ${job_label}"
            # Mark as done (atomic enough for sequential-within-worker appends)
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
export PYTHON CONFIG_DIR DONE_FILE

# ---------------------------------------------------------------------------
# Launch one background process per slot (JOBS_PER_GPU slots share each GPU)
# ---------------------------------------------------------------------------
declare -A PIDS
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    log_file="${LOG_DIR}/chronos2_sweep_eval_gpu${gpu}_w${slot}.log"
    echo "Launching GPU ${gpu} worker ${slot} → ${log_file}"
    run_gpu_worker "$gpu" "$slot" "${SLOT_CONFIGS[$slot]}" \
        > "$log_file" 2>&1 &
    PIDS[$slot]=$!
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(tail -f logs/chronos2_sweep_eval_gpu<N>_w<slot>.log to monitor)"
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
        echo "GPU ${gpu} worker ${slot}: FAILED  (see ${LOG_DIR}/chronos2_sweep_eval_gpu${gpu}_w${slot}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== Chronos-2 sweep eval complete  $(date) ==="
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} GPU worker(s) reported failures — check logs above."
    exit 1
fi
echo "  All jobs passed."
