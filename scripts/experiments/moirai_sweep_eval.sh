#!/usr/bin/env bash
# moirai_sweep_eval.sh
#
# Runs nocturnal_hypo_eval.py --probabilistic for every Moirai sweep config
# (00–14) × datasets.
#
# Checkpoint paths are read from the manifest written by moirai_sweep_train.sh:
#   trained_models/artifacts/moirai/sweep_manifest.txt
# Format: <stem>\t<output_dir>   (tab-separated, one entry per config)
# If a stem appears more than once (re-run), the last entry wins.
#
# Dataset assignment by covariate class:
#   ALL: bg-only configs (00, 09)
#   IOB: insulin covariate configs (01–04, 10–14) — excl. tamborlane (no insulin data)
#   COB: carb covariate configs (05–08) — excl. brown + tamborlane (no meal data)
#
# Usage:
#   bash scripts/experiments/moirai_sweep_eval.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/experiments/moirai_sweep_eval.sh
#   bash scripts/experiments/moirai_sweep_eval.sh 2>&1 | tee moirai_sweep_eval.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venvs/moirai/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_DIR="configs/data/holdout_10pct"
ALL_DATASETS=(lynch_2022 aleppo_2017 brown_2019 tamborlane_2008)
DATASETS_WITH_IOB=(lynch_2022 aleppo_2017 brown_2019)
DATASETS_WITH_COB=(lynch_2022 aleppo_2017)
MANIFEST="trained_models/artifacts/moirai/sweep_manifest.txt"

# Format: "stem|context_length|covariate_cols (space-sep, empty=none)|datasets_key"
# datasets_key: ALL, IOB, or COB
CONFIG_META=(
    "00_bg_only|512||ALL"
    "01_bg_iob|512|iob|IOB"
    "02_bg_iob_ia|512|iob insulin_availability|IOB"
    "03_bg_iob_ia_high_lr|512|iob insulin_availability|IOB"
    "04_bg_iob_short_ctx|288|iob|IOB"
    "05_bg_iob_cob|512|iob cob|COB"
    "06_bg_full_features|512|iob insulin_availability cob carb_availability|COB"
    "07_bg_iob_cob_high_lr|512|iob cob|COB"
    "08_bg_iob_cob_short_ctx|288|iob cob|COB"
    "09_bg_only_base|512||ALL"
    "10_bg_iob_ia_head_only|512|iob insulin_availability|IOB"
    "11_bg_iob_ia_freeze_ffn|512|iob insulin_availability|IOB"
    "12_bg_iob_ia_patch8|512|iob insulin_availability|IOB"
    "13_bg_iob_ia_patch16|512|iob insulin_availability|IOB"
    "14_bg_iob_ia_patch32|512|iob insulin_availability|IOB"
)

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: moirai venv not found at $PYTHON"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found at $MANIFEST"
    echo "       Run moirai_sweep_train.sh first."
    exit 1
fi

PASS=0
FAIL=0
FAILED=()

for entry in "${CONFIG_META[@]}"; do
    IFS='|' read -r stem ctx_len cov_cols datasets_key <<< "$entry"
    model_config="configs/models/moirai/${stem}.yaml"

    if [[ "$datasets_key" == "IOB" ]]; then
        datasets=("${DATASETS_WITH_IOB[@]}")
    elif [[ "$datasets_key" == "COB" ]]; then
        datasets=("${DATASETS_WITH_COB[@]}")
    else
        datasets=("${ALL_DATASETS[@]}")
    fi

    # Look up output dir from manifest (last match wins for re-runs)
    out_dir="$(awk -F'\t' -v s="$stem" '$1 == s { last=$2 } END { print last }' "$MANIFEST")"

    if [[ -z "$out_dir" ]]; then
        echo ""
        echo "[SKIP] moirai / $stem — not found in manifest: $MANIFEST"
        echo "       Run moirai_sweep_train.sh first."
        FAIL=$((FAIL + 1))
        FAILED+=("moirai / $stem (missing manifest entry)")
        continue
    fi

    # Moirai saves to a model.pt/ directory (base-class format); fall back to
    # a .ckpt file for externally trained checkpoints.
    checkpoint="${PROJECT_ROOT}/${out_dir}/model.pt"
    if [[ ! -d "$checkpoint" ]]; then
        checkpoint="${PROJECT_ROOT}/${out_dir}/model.ckpt"
    fi

    if [[ ! -d "$checkpoint" && ! -f "$checkpoint" ]]; then
        echo ""
        echo "[SKIP] moirai / $stem — checkpoint not found in: $out_dir"
        FAIL=$((FAIL + 1))
        FAILED+=("moirai / $stem (missing checkpoint)")
        continue
    fi

    for dataset in "${datasets[@]}"; do
        label="moirai / $stem / $dataset"
        echo ""
        echo "============================================================"
        echo "  Eval: $label"
        echo "  Checkpoint: $checkpoint"
        echo "============================================================"

        CMD=(
            "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py
            --model moirai
            --model-config "$model_config"
            --dataset "$dataset"
            --config-dir "$CONFIG_DIR"
            --checkpoint "$checkpoint"
            --context-length "$ctx_len"
            --forecast-length 96
            --cuda-device "$GPU"
            --probabilistic
        )
        if [[ -n "$cov_cols" ]]; then
            # word-split intentional: cov_cols is space-separated
            # shellcheck disable=SC2086
            CMD+=(--covariate-cols $cov_cols)
        fi

        if "${CMD[@]}"; then
            echo "[OK] $label"
            PASS=$((PASS + 1))
        else
            echo "[FAIL] $label"
            FAIL=$((FAIL + 1))
            FAILED+=("$label")
        fi
    done
done

echo ""
echo "============================================================"
echo "  Moirai sweep eval complete  $(date)"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"

[[ $FAIL -eq 0 ]]
