SHELL := /bin/bash
ROOT_DIR := $(shell pwd)
VENVS_DIR := $(ROOT_DIR)/.venvs
PYTHON ?= python3.11

# Per-extras virtualenvs (because of conflicting deps).
# Each name corresponds to an extras key in pyproject.toml.
# `autogluon` is a shared env covering deepar / patchtst / tft / naive_baseline
# / statistical / tide — they all use autogluon.timeseries 1.5.0 and don't
# conflict. chronos2 stays separate because it pins transformers==4.56.0.
MODEL_VENVS := ttm sundial chronos2 timegrad moment timesfm autogluon moirai toto

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  venv-base               Create main venv (.venv) for data + analysis code"
	@echo "  venv-<extras>           Create per-extras venv at .venvs/<extras>"
	@echo "                          (extras: $(MODEL_VENVS))"
	@echo "  venv-all-models         Create every per-extras venv listed above"
	@echo "  test                    Run the lightweight test suite in .venv"
	@echo "  summary                 Aggregate per-run nocturnal evaluations into summary.csv"
	@echo "  grand-summary           Rebuild results/grand_summary/ from experiments/"

# ─────────────────────────────────────────────────────────────────────────────
# Virtualenv bootstrap
# ─────────────────────────────────────────────────────────────────────────────
# .venv         → main env: data loading, evaluation, analysis only.
# .venvs/autogluon/ → shared env for AutoGluon-backed models (deepar,
#                    patchtst, tft, naive_baseline, statistical, tide).
# .venvs/<model>/   → isolated per-model env created from
#                    [project.optional-dependencies].<model> in pyproject.toml.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: venv-base venv-all-models $(addprefix venv-,$(MODEL_VENVS))

venv-base:
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -e .

# Pattern rule: `make venv-ttm` builds .venvs/ttm with the ttm extras, etc.
$(addprefix venv-,$(MODEL_VENVS)):
	$(eval MODEL := $(patsubst venv-%,%,$@))
	$(PYTHON) -m venv $(VENVS_DIR)/$(MODEL)
	$(VENVS_DIR)/$(MODEL)/bin/pip install --upgrade pip
	$(VENVS_DIR)/$(MODEL)/bin/pip install -e ".[$(MODEL)]"

venv-all-models: $(addprefix venv-,$(MODEL_VENVS))

# ─────────────────────────────────────────────────────────────────────────────
# Tests (run in main .venv)
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	./.venv/bin/python -m pytest tests/ -v

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

# `summary` walks experiments/nocturnal_forecasting/ and writes summary.csv —
# required intermediate step before `grand-summary`.
.PHONY: summary
summary:
	./.venv/bin/python scripts/analysis/summarize_experiments.py

.PHONY: grand-summary
grand-summary:
	./.venv/bin/python scripts/analysis/build_grand_summary.py
