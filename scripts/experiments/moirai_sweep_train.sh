#!/usr/bin/env bash
# moirai_sweep_train.sh
#
# Fine-tunes all Moirai sweep configs (00–14) in succession.
# Dataset assignment by covariate class:
#   ALL: bg-only configs (00, 09)  — tamborlane, brown, aleppo, lynch
#   IOB: insulin covariate configs (01–04, 10–14) — excl. tamborlane (no insulin data)
#   COB: carb covariate configs (05–08) — excl. brown + tamborlane (no meal data)
# Output dirs follow the project convention:
#   trained_models/artifacts/moirai/<date>_RID<id>_holdout_workflow
# A manifest is written to trained_models/artifacts/moirai/sweep_manifest.txt
# so moirai_sweep_eval.sh can find the checkpoints without parsing dir names.
#
# Usage:
#   bash scripts/experiments/moirai_sweep_train.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/moirai_sweep_train.sh
#   bash scripts/experiments/moirai_sweep_train.sh 2>&1 | tee moirai_sweep_train.log

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
MANIFEST="trained_models/artifacts/moirai/sweep_manifest.txt"

# Format: "config_path|datasets_key"
# datasets_key: ALL, IOB, or COB
CONFIGS=(
    "configs/models/moirai/00_bg_only.yaml|ALL"
    "configs/models/moirai/01_bg_iob.yaml|IOB"
    "configs/models/moirai/02_bg_iob_ia.yaml|IOB"
    "configs/models/moirai/03_bg_iob_ia_high_lr.yaml|IOB"
    "configs/models/moirai/04_bg_iob_short_ctx.yaml|IOB"
    "configs/models/moirai/05_bg_iob_cob.yaml|COB"
    "configs/models/moirai/06_bg_full_features.yaml|COB"
    "configs/models/moirai/07_bg_iob_cob_high_lr.yaml|COB"
    "configs/models/moirai/08_bg_iob_cob_short_ctx.yaml|COB"
    "configs/models/moirai/09_bg_only_base.yaml|ALL"
    "configs/models/moirai/10_bg_iob_ia_head_only.yaml|IOB"
    "configs/models/moirai/11_bg_iob_ia_freeze_ffn.yaml|IOB"
    "configs/models/moirai/12_bg_iob_ia_patch8.yaml|IOB"
    "configs/models/moirai/13_bg_iob_ia_patch16.yaml|IOB"
    "configs/models/moirai/14_bg_iob_ia_patch32.yaml|IOB"
)

mkdir -p "trained_models/artifacts/moirai"
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

    RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
    out_dir="trained_models/artifacts/moirai/$(date +%Y-%m-%d_%H:%M)_RID${RUN_ID}_holdout_workflow"

    echo ""
    echo "============================================================"
    echo "  Training: moirai / $stem"
    echo "  Config:   $config"
    echo "  Datasets: $datasets"
    echo "  Output:   $out_dir"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES="$GPU" \
       MODEL_TYPE="moirai" \
       MODEL_CONFIG="$config" \
       CONFIG_DIR="$CONFIG_DIR" \
       DATASETS="$datasets" \
       SKIP_TRAINING="false" \
       SKIP_STEPS="7" \
       OUTPUT_BASE_DIR="$out_dir" \
       RUN_ID="$RUN_ID" \
       "$WORKFLOW"; then
        echo "[OK] moirai / $stem"
        echo "${stem}"$'\t'"${out_dir}" >> "$MANIFEST"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] moirai / $stem"
        FAIL=$((FAIL + 1))
        FAILED+=("moirai / $stem")
    fi
done

echo ""
echo "============================================================"
echo "  Moirai sweep training complete  $(date)"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
