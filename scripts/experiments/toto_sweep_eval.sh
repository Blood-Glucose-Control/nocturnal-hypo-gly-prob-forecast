#!/usr/bin/env bash
# Compatibility launcher for canonical evaluation sweep script.
# Canonical path: scripts/evaluation/sweeps/models/toto_sweep_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/evaluation/sweeps/models/toto_sweep_eval.sh "$@"
