SHELL := /bin/bash
ROOT_DIR := $(shell pwd)
VENVS_DIR := $(ROOT_DIR)/.venvs

# ─────────────────────────────────────────────────────────────────────────────
# Model-specific test targets
# Each model has its own venv due to conflicting dependency versions.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test-ttm test-sundial test-timesfm test-chronos2 test-models test lint summary grand-summary help

test-ttm:
	$(VENVS_DIR)/ttm/bin/python -m pytest tests/models/ -v -k ttm

test-sundial:
	$(VENVS_DIR)/sundial/bin/python -m pytest tests/models/ -v -k "not ttm and not timesfm and not chronos2"

test-timesfm:
	$(VENVS_DIR)/timesfm/bin/python -m pytest tests/models/ -v -k timesfm

test-chronos2:
	$(VENVS_DIR)/chronos2/bin/python -m pytest tests/models/ -v -k chronos2

## Run all per-model tests sequentially with their correct venvs
test-models: test-ttm test-sundial test-timesfm test-chronos2

# ─────────────────────────────────────────────────────────────────────────────
# General tests (main venv — no model-specific deps required)
# ─────────────────────────────────────────────────────────────────────────────

test:
	.noctprob-venv/bin/python -m pytest tests/ -v --ignore=tests/models

## Run everything: common tests + all model tests
test-all: test test-models

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: summary grand-summary

## Summarize nocturnal forecasting experiments
summary-rmse:
	.noctprob-venv/bin/python scripts/analysis/summarize_experiments.py --metric rmse

summary-wql:
	.noctprob-venv/bin/python scripts/analysis/summarize_experiments.py --metric wql

## Rebuild grand summary table (results/grand_summary/)
grand-summary:
	.noctprob-venv/bin/python scripts/analysis/build_grand_summary.py

# ─────────────────────────────────────────────────────────────────────────────
# Linting
# ─────────────────────────────────────────────────────────────────────────────

lint:
	.noctprob-venv/bin/python -m ruff check src/ tests/

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
	@echo ""
	@echo "Targets:"
	@echo "  test-ttm       Run TTM tests using .venvs/ttm"
	@echo "  test-sundial   Run Sundial tests using .venvs/sundial"
	@echo "  test-timesfm   Run TimesFM tests using .venvs/timesfm"
	@echo "  test-chronos2  Run Chronos-2 tests using .venvs/chronos2"
	@echo "  test-models    Run all model tests (all venvs)"
	@echo "  test           Run non-model tests (main venv)"
	@echo "  test-all       Run all tests"
	@echo "  lint           Run ruff linter"
	@echo "  summary-rmse   Summarize nocturnal forecasting experiments (RMSE)"
	@echo "  summary-wql    Summarize nocturnal forecasting experiments (WQL)"
	@echo "  grand-summary  Rebuild grand summary table"
