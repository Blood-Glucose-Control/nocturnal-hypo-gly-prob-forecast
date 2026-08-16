#!/usr/bin/env bash
# Compatibility launcher for canonical training sweep script.
# Canonical path: scripts/training/sweeps/models/statistical_sweep_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/training/sweeps/models/statistical_sweep_train.sh "$@"
