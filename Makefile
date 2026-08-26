SHELL := /bin/bash
ROOT_DIR := $(shell pwd)
VENVS_DIR := $(ROOT_DIR)/.venvs

# ─────────────────────────────────────────────────────────────────────────────
# Model-specific test targets
# Most models have their own venv; AutoGluon-backed models share .venvs/autogluon.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test-ttm test-sundial test-timesfm test-autogluon test-chronos2 test-models test lint smoke-suite-aleppo smoke-suite-compare help

test-ttm:
	$(VENVS_DIR)/ttm/bin/python -m pytest tests/models/ -v -k ttm

test-sundial:
	$(VENVS_DIR)/sundial/bin/python -m pytest tests/models/ -v -k "not ttm and not timesfm and not chronos2"

test-timesfm:
	$(VENVS_DIR)/timesfm/bin/python -m pytest tests/models/ -v -k timesfm

test-autogluon:
	$(VENVS_DIR)/autogluon/bin/python -m pytest tests/models/ -v -k "chronos2 or autogluon_base or deepar or patchtst or tft or naive_baseline or statistical"

test-chronos2: test-autogluon

## Run all per-model tests sequentially with their correct venvs
test-models: test-ttm test-sundial test-timesfm test-autogluon

# ─────────────────────────────────────────────────────────────────────────────
# General tests (main venv — no model-specific deps required)
# ─────────────────────────────────────────────────────────────────────────────

test:
	.noctprob-venv/bin/python -m pytest tests/ -v --ignore=tests/models

## Run everything: common tests + all model tests
test-all: test test-models

# ─────────────────────────────────────────────────────────────────────────────
# Linting
# ─────────────────────────────────────────────────────────────────────────────

lint:
	.noctprob-venv/bin/python -m ruff check src/ tests/

## Run Aleppo one-epoch regression smoke suite across maintained models.
## Optional env:
##   SUITE_LABEL=pre_refactor_20260825
##   MODELS=ttm,chronos2,moment
##   FAIL_FAST=true
##   DRY_RUN=true
smoke-suite-aleppo:
	@FAIL_FAST_ARG=""; DRY_RUN_ARG=""; \
	if [ "$${FAIL_FAST:-}" = "true" ]; then FAIL_FAST_ARG="--fail-fast"; fi; \
	if [ "$${DRY_RUN:-}" = "true" ]; then DRY_RUN_ARG="--dry-run"; fi; \
	.noctprob-venv/bin/python scripts/workflows/forecasting/run_aleppo_model_regression_smoke_suite.py \
		--suite-label "$${SUITE_LABEL:-manual_$$(date +%Y%m%d_%H%M%S)}" \
		$${MODELS:+--models "$$MODELS"} \
		$$FAIL_FAST_ARG \
		$$DRY_RUN_ARG

## Compare pre/post suite manifests.
## Required env:
##   BASELINE=<path>/suite_manifest.json
##   CANDIDATE=<path>/suite_manifest.json
## Optional env:
##   REL_TOL=0.25
##   ABS_TOL=1e-6
##   REPORT_PATH=<path>/comparison_report.json
smoke-suite-compare:
	.noctprob-venv/bin/python scripts/workflows/forecasting/compare_regression_smoke_suites.py \
		--baseline "$$BASELINE" \
		--candidate "$$CANDIDATE" \
		$${REL_TOL:+--rel-tol "$$REL_TOL"} \
		$${ABS_TOL:+--abs-tol "$$ABS_TOL"} \
		$${REPORT_PATH:+--report-path "$$REPORT_PATH"}

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
	@echo ""
	@echo "Targets:"
	@echo "  test-ttm       Run TTM tests using .venvs/ttm"
	@echo "  test-sundial   Run Sundial tests using .venvs/sundial"
	@echo "  test-timesfm   Run TimesFM tests using .venvs/timesfm"
	@echo "  test-autogluon Run AutoGluon-family tests using .venvs/autogluon"
	@echo "  test-chronos2  Alias for test-autogluon (backward compatible)"
	@echo "  test-models    Run all model tests (all venvs)"
	@echo "  test           Run non-model tests (main venv)"
	@echo "  test-all       Run all tests"
	@echo "  lint           Run ruff linter"
	@echo "  smoke-suite-aleppo   Run Aleppo one-epoch pre/post smoke suite"
	@echo "  smoke-suite-compare  Compare two smoke suite manifests"
