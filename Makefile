SHELL := /bin/bash
ROOT_DIR := $(shell pwd)
VENVS_DIR := $(ROOT_DIR)/.venvs
PYTHON ?= python3.11

# Models that need their own virtualenv (because of conflicting deps).
# Each name corresponds to an extras key in pyproject.toml.
MODEL_VENVS := ttm sundial chronos2 timegrad moment timesfm tide moirai toto

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  venv-base               Create main venv (.venv) for data + analysis code"
	@echo "  venv-<model>            Create per-model venv at .venvs/<model> via extras"
	@echo "                          (models: $(MODEL_VENVS))"
	@echo "  venv-all-models         Create every per-model venv listed above"
	@echo "  test                    Run the lightweight test suite in .venv"
	@echo "  lint                    Run ruff over src/ and tests/"
	@echo "  grand-summary           Rebuild results/grand_summary/ from experiments/"

# ─────────────────────────────────────────────────────────────────────────────
# Virtualenv bootstrap
# ─────────────────────────────────────────────────────────────────────────────
# .venv  → main env: data loaders, evaluation, analysis, naive/statistical
#          baselines, plus DeepAR/PatchTST/TFT (autogluon/torch, no conflicts).
# .venvs/<model>/ → isolated per-model env created from
#          [project.optional-dependencies].<model> in pyproject.toml.
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

.PHONY: grand-summary
grand-summary:
	./.venv/bin/python scripts/analysis/build_grand_summary.py
