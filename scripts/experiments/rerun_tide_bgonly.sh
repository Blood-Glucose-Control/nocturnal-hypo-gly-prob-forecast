#!/usr/bin/env bash
# rerun_tide_bgonly.sh
# Re-evaluates tide on aleppo_2017 and lynch_2022 using the bg_only checkpoint
# (2026-04-17_16:52_RID20260417_165227_3405856_holdout_workflow).
# The previous "best" selections for these datasets used covariate-contaminated
# checkpoints (future leakage via AutoGluon known-future features) and are invalid.
#
# Usage:
#   bash scripts/experiments/rerun_tide_bgonly.sh
#   bash scripts/experiments/rerun_tide_bgonly.sh --dry-run

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venvs/tide/bin/python"
CKPT="$REPO_DIR/trained_models/artifacts/tide/2026-04-17_16:52_RID20260417_165227_3405856_holdout_workflow/model.pt"
LOG_DIR="$REPO_DIR/logs"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[DRY-RUN] Commands will be printed but not executed."
fi

mkdir -p "$LOG_DIR"

if [[ ! -e "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

JOBS=(
  "aleppo_2017|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-05-05_tide_bgonly_aleppo_2017"
  "lynch_2022|experiments/nocturnal_forecasting/512ctx_96fh/tide/2026-05-05_tide_bgonly_lynch_2022"
)

for entry in "${JOBS[@]}"; do
    IFS='|' read -r dataset out_dir <<< "$entry"
    cmd="$PYTHON scripts/experiments/nocturnal_hypo_eval.py \
--model tide \
--dataset $dataset \
--context-length 512 \
--forecast-length 96 \
--checkpoint $CKPT \
--output-dir $out_dir \
--probabilistic"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "--- $dataset ---"
        echo "$cmd"
        echo ""
    else
        logfile="$LOG_DIR/tide_bgonly_${dataset}.log"
        echo "[Running] tide / $dataset → $logfile"
        eval "$cmd" > "$logfile" 2>&1 && echo "  [OK]   $dataset" || echo "  [FAIL] $dataset (see $logfile)"
    fi
done

echo "Done."
