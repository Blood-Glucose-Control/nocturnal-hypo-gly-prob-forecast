#!/usr/bin/env bash
# Canonical workflow entrypoint (Wave 1 taxonomy).
# Executes the maintained implementation module.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

exec bash scripts/workflows/forecasting/run_forecasting_workflow_impl.sh "$@"
