#!/usr/bin/env bash
# Compatibility shim: canonical path is scripts/workflows/forecasting/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec bash scripts/workflows/forecasting/forecasting_workflow_regression_smoke.sh "$@"
