#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CHECKPOINT="trained_models/artifacts/statistical/2026-04-28_0318_RID20260428_031849_856335_01_autoarima_bg_iob"
PYTHON=".venvs/chronos2/bin/python"
CONFIG_DIR="configs/data/holdout_10pct"

for DATASET in aleppo_2017 brown_2019 lynch_2022; do
  echo "=== 01_autoarima_bg_iob | $DATASET ==="
  "$PYTHON" scripts/experiments/nocturnal_hypo_eval.py \
    --model statistical \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --config-dir "$CONFIG_DIR" \
    --context-length 512 \
    --forecast-length 96 \
    --covariate-cols iob \
    --probabilistic \
    --no-dilate \
    2>&1 | tail -30
  echo ""
done
echo "=== Done ==="
