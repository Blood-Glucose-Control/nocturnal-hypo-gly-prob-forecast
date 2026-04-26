#!/usr/bin/env bash
# chronos2_sweep_train.sh
#
# Fine-tunes all Chronos-2 sweep configs in parallel, JOBS_PER_GPU workers per GPU.
# Configs are distributed round-robin across all slots (GPU × worker). Workers
# sharing a GPU run concurrently; each worker's configs run sequentially.
#
# Dataset assignment by covariate class:
#   ALL: config 00       — aleppo_2017, brown_2019, lynch_2022, tamborlane_2008
#   IOB: configs 01–05   — aleppo_2017, brown_2019, lynch_2022
#   COB: configs 06–09   — aleppo_2017 only  (lynch carbs are all-zero; brown/tamborlane have no meal data)
#   IOB_CTX: configs 10–12 — aleppo_2017, brown_2019, lynch_2022  (high-LR IOB context ablation)
#   COB_CTX: configs 13–15 — aleppo_2017 only                     (high-LR COB context ablation)
#
# Usage:
#   bash scripts/experiments/chronos2_sweep_train.sh
#   GPUS="0 1 2 3" bash scripts/experiments/chronos2_sweep_train.sh
#   GPUS="0 1"     bash scripts/experiments/chronos2_sweep_train.sh
#   GPUS="0 1" JOBS_PER_GPU=2 bash scripts/experiments/chronos2_sweep_train.sh
#
# Per-GPU logs are written to logs/chronos2_sweep_gpu<N>.log
# A sweep manifest is written to trained_models/artifacts/chronos2/sweep_manifest.txt
# (used by chronos2_eval_long_run_checkpoints.sh to locate artifact dirs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

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
echo "=== Chronos-2 sweep training  $(date) ==="
echo "  GPUs: ${GPUS_ARRAY[*]}  (${N_GPUS} total)"
echo ""

# ---------------------------------------------------------------------------
# Config definitions — format: "config_path|datasets_key"
# ---------------------------------------------------------------------------
DATASETS_ALL="aleppo_2017 brown_2019 lynch_2022 tamborlane_2008"
DATASETS_WITH_IOB="aleppo_2017 brown_2019 lynch_2022"
DATASETS_WITH_COB="aleppo_2017"  # lynch carbs are all-zero; only aleppo has real meal data

CONFIGS=(
    "configs/models/chronos2/00_bg_only.yaml|ALL"
    "configs/models/chronos2/01_bg_iob_insulin_availability.yaml|IOB"
    "configs/models/chronos2/02_bg_iob.yaml|IOB"
    "configs/models/chronos2/03_joint_bg_iob.yaml|IOB"
    "configs/models/chronos2/04_bg_iob_ia_high_lr.yaml|IOB"
    "configs/models/chronos2/05_bg_iob_short_ctx.yaml|IOB"
    "configs/models/chronos2/06_bg_iob_cob.yaml|COB"
    "configs/models/chronos2/07_bg_full_features.yaml|COB"
    "configs/models/chronos2/08_bg_iob_cob_high_lr.yaml|COB"
    "configs/models/chronos2/09_bg_iob_cob_short_ctx.yaml|COB"
    # Context window ablation — high-LR IOB (mirrors 04 at 256/128/64 steps)
    "configs/models/chronos2/10_bg_iob_ia_high_lr_ctx256.yaml|IOB"
    "configs/models/chronos2/11_bg_iob_ia_high_lr_ctx128.yaml|IOB"
    "configs/models/chronos2/12_bg_iob_ia_high_lr_ctx64.yaml|IOB"
    # Context window ablation — high-LR COB (mirrors 08 at 256/128/64 steps)
    "configs/models/chronos2/13_bg_iob_cob_high_lr_ctx256.yaml|COB"
    "configs/models/chronos2/14_bg_iob_cob_high_lr_ctx128.yaml|COB"
    "configs/models/chronos2/15_bg_iob_cob_high_lr_ctx64.yaml|COB"
)

CONFIG_DIR="configs/data/holdout_10pct"
WORKFLOW="scripts/examples/run_holdout_generic_workflow.sh"
MANIFEST_DIR="trained_models/artifacts/chronos2"
# Skip holdout config generation (step 1) and validation (step 2) — same result
# every run since the 10pct configs are already committed and unchanged.
# Skip zero-shot eval (step 4) — not the metric we're tuning in the sweep.
# Skip resume training (step 7) — single long run with checkpoints is the strategy.
SKIP_STEPS="${SKIP_STEPS:-1 2 4 7}"
MANIFEST="${MANIFEST_DIR}/sweep_manifest.txt"
LOG_DIR="logs"

mkdir -p "$MANIFEST_DIR" "$LOG_DIR"
# Append to manifest (don't truncate — concurrent GPU processes write to it)
touch "$MANIFEST"

# ---------------------------------------------------------------------------
# Distribute configs round-robin across slots (JOBS_PER_GPU workers per GPU)
# ---------------------------------------------------------------------------
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
N_SLOTS=$(( N_GPUS * JOBS_PER_GPU ))
echo "  Jobs per GPU: ${JOBS_PER_GPU}  (${N_SLOTS} total slots)"

# SLOT_CONFIGS[$slot] = newline-delimited config entries for that slot
# SLOT_GPU[$slot]     = which GPU the slot runs on
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
# Worker function — runs all configs assigned to one slot sequentially.
# Multiple slots may share the same GPU (run concurrently).
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

        # Skip if already successfully trained (manifest entry + model.pt dir exist)
        local existing_dir
        existing_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST" 2>/dev/null || true)"
        if [[ -n "$existing_dir" && -d "${existing_dir}/model.pt" ]]; then
            echo "[${label}] [SKIP] ${stem} — already in manifest: $(basename "$existing_dir")"
            pass=$(( pass + 1 ))
            continue
        fi

        local datasets
        if [[ "$datasets_key" == "COB" ]]; then
            datasets="$DATASETS_WITH_COB"
        elif [[ "$datasets_key" == "ALL" ]]; then
            datasets="$DATASETS_ALL"
        else
            datasets="$DATASETS_WITH_IOB"
        fi

        local run_id
        run_id="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
        local out_dir="${MANIFEST_DIR}/$(date +%Y-%m-%d_%H%M)_RID${run_id}_${stem}"

        echo ""
        echo "[${label}] ============================================"
        echo "[${label}]  Training: chronos2 / ${stem}"
        echo "[${label}]  Config:   ${config}"
        echo "[${label}]  Datasets: ${datasets}"
        echo "[${label}]  Output:   ${out_dir}"
        echo "[${label}] ============================================"

        if CUDA_VISIBLE_DEVICES="$gpu" \
           MODEL_TYPE="chronos2" \
           MODEL_CONFIG="$config" \
           CONFIG_DIR="$CONFIG_DIR" \
           DATASETS="$datasets" \
           SKIP_TRAINING="false" \
           SKIP_STEPS="$SKIP_STEPS" \
           OUTPUT_BASE_DIR="$out_dir" \
           RUN_ID="$run_id" \
           "$WORKFLOW"; then
            echo "[${label}] [OK] ${stem}"
            # Append to shared manifest (echo is atomic for short lines)
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
export DATASETS_ALL DATASETS_WITH_IOB DATASETS_WITH_COB MANIFEST_DIR MANIFEST CONFIG_DIR WORKFLOW SKIP_STEPS

# ---------------------------------------------------------------------------
# Launch one background process per slot (JOBS_PER_GPU slots share each GPU)
# ---------------------------------------------------------------------------
declare -A PIDS
for (( slot=0; slot<N_SLOTS; slot++ )); do
    gpu="${SLOT_GPU[$slot]}"
    log_file="${LOG_DIR}/chronos2_sweep_gpu${gpu}_w${slot}.log"
    echo "Launching GPU ${gpu} worker ${slot} → ${log_file}"
    run_gpu_worker "$gpu" "$slot" "${SLOT_CONFIGS[$slot]}" \
        > "$log_file" 2>&1 &
    PIDS[$slot]=$!
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(tail -f logs/chronos2_sweep_gpu<N>_w<slot>.log to monitor)"
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
        echo "GPU ${gpu} worker ${slot}: FAILED  (see ${LOG_DIR}/chronos2_sweep_gpu${gpu}_w${slot}.log)"
        OVERALL_FAIL=$(( OVERALL_FAIL + 1 ))
    fi
done

echo ""
echo "=== Chronos-2 sweep training complete  $(date) ==="
echo "  Manifest: ${MANIFEST}"
if [[ $OVERALL_FAIL -gt 0 ]]; then
    echo "  ${OVERALL_FAIL} GPU worker(s) reported failures — check logs above."
    exit 1
fi
echo "  All configs trained successfully."
