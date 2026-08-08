#!/usr/bin/env bash
# tide_sweep_train.sh
#
# Trains all 7 TiDE sweep configs (00–06) in succession.
# Dataset assignment by covariate class:
#   ALL: bg-only config (00)  — tamborlane, brown, aleppo, lynch
#   IOB: insulin covariate configs (01, 02, 04–06) — excl. tamborlane (no insulin data)
#   COB: carb covariate config (03) — excl. brown + tamborlane (no meal data)
# Output dirs follow the project convention:
#   trained_models/artifacts/tide/<date>_RID<id>_holdout_workflow
# A manifest is written so tide_sweep_eval.sh can find the checkpoints.
#
# Usage:
#   bash scripts/experiments/tide_sweep_train.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/tide_sweep_train.sh
#   bash scripts/experiments/tide_sweep_train.sh 2>&1 | tee tide_sweep_train.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
ALL_DATASETS="lynch_2022 aleppo_2017 brown_2019 tamborlane_2008"
# Insulin covariates (iob, insulin_availability): tamborlane has no insulin data
DATASETS_WITH_IOB="lynch_2022 aleppo_2017 brown_2019"
# Carb covariates (cob, carb_availability): brown has no meal data; tamborlane has no covariates
DATASETS_WITH_COB="lynch_2022 aleppo_2017"
CONFIG_DIR="configs/data/holdout_10pct"
WORKFLOW="scripts/examples/run_holdout_generic_workflow.sh"
MANIFEST="trained_models/artifacts/tide/sweep_manifest.txt"

# Format: "config_path|datasets_key"
CONFIGS=(
    "configs/models/tide/00_bg_only.yaml|ALL"
    "configs/models/tide/01_bg_iob.yaml|IOB"
    "configs/models/tide/02_bg_iob_insulin_availability.yaml|IOB"
    "configs/models/tide/03_bg_all_covariates.yaml|COB"
    "configs/models/tide/04_large_arch.yaml|IOB"
    "configs/models/tide/05_bg_iob_ia_high_lr.yaml|IOB"
    "configs/models/tide/06_bg_iob_ia_short_ctx.yaml|IOB"
)

mkdir -p "trained_models/artifacts/tide"
# Append to manifest (allows re-runs to add new entries without wiping old ones)
touch "$MANIFEST"

PASS=0
FAIL=0
FAILED=()

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r config datasets_key <<< "$entry"
    stem="$(basename "$config" .yaml)"
    if [[ "$datasets_key" == "IOB" ]]; then
        datasets="$DATASETS_WITH_IOB"
    elif [[ "$datasets_key" == "COB" ]]; then
        datasets="$DATASETS_WITH_COB"
    else
        datasets="$ALL_DATASETS"
    fi
    # Generate RID matching the workflow's own convention: YYYYMMDD_HHMMSS_PID
    RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
    out_dir="trained_models/artifacts/tide/$(date +%Y-%m-%d_%H:%M)_RID${RUN_ID}_holdout_workflow"

    echo ""
    echo "============================================================"
    echo "  Training: tide / $stem"
    echo "  Config:   $config"
    echo "  Datasets: $datasets"
    echo "  Output:   $out_dir"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES="$GPU" \
       MODEL_TYPE="tide" \
       MODEL_CONFIG="$config" \
       CONFIG_DIR="$CONFIG_DIR" \
       DATASETS="$datasets" \
       SKIP_TRAINING="false" \
       SKIP_STEPS="7" \
       OUTPUT_BASE_DIR="$out_dir" \
       RUN_ID="$RUN_ID" \
       "$WORKFLOW"; then
        echo "[OK] tide / $stem"
        # Record stem → output dir so tide_sweep_eval.sh can find the checkpoint
        echo "${stem}	${out_dir}" >> "$MANIFEST"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] tide / $stem"
        FAIL=$((FAIL + 1))
        FAILED+=("tide / $stem")
    fi
done

echo ""
echo "============================================================"
echo "  TiDE sweep training complete  $(date)"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
