#!/usr/bin/env bash
# Thin shell wrapper for Chronos-2 sweep training orchestration profile.
#
# Generic orchestration control flow is implemented in:
#   src/workflows/sweeps/train.py
# Chronos-2 profile wrapper:
#   scripts/experiments/chronos2_sweep_train.py
#
# Supported environment contract (preserved):
#   GPUS, JOBS_PER_GPU, CONFIG_DIR, SKIP_STEPS
#
# Optional fast path-check:
#   DRY_RUN=1 bash scripts/experiments/chronos2_sweep_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/experiments/chronos2_sweep_train.py "$@"
