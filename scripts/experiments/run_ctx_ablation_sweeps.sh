#!/usr/bin/env bash
# Thin compatibility launcher for the canonical context-ablation chain runner.
#
# Canonical path:
#   scripts/orchestration/sweeps/run_ctx_ablation_sweeps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/orchestration/sweeps/run_ctx_ablation_sweeps.sh "$@"
