#!/usr/bin/env bash
# Thin compatibility launcher for the canonical Chronos-2 sweep orchestrator.
#
# Canonical path:
#   scripts/orchestration/sweeps/chronos2_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/orchestration/sweeps/chronos2_sweep.sh "$@"
