#!/usr/bin/env bash
# Creates or activates a model-specific virtual environment.
#
# Usage:
#   source scripts/setup_model_env.sh <model>
#
# Examples:
#   source scripts/setup_model_env.sh ttm
#   source scripts/setup_model_env.sh sundial
#
# Available models (defined as optional deps in pyproject.toml):
#   ttm, sundial

MODEL="${1:?Usage: source scripts/setup_model_env.sh <model>}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
VENVS_DIR="${REPO_ROOT}/.venvs"
VENV_PATH="${VENVS_DIR}/${MODEL}"

# Validate model name exists in pyproject.toml
if ! grep -qF "${MODEL} = [" "${REPO_ROOT}/pyproject.toml" 2>/dev/null; then
    echo "Error: Model '${MODEL}' not found in pyproject.toml [project.optional-dependencies]"
    echo "Available models:"
    grep -F ' = [' "${REPO_ROOT}/pyproject.toml" | sed 's/ = \[.*//'
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating new venv for '${MODEL}' at ${VENV_PATH}..."
    if ! python3 -m venv "${VENV_PATH}"; then
        echo "Error: Failed to create virtual environment."
        echo "You may need to install python3-venv:"
        echo "  apt install python3.12-venv  (or your Python version)"
        echo ""
        echo "Alternative: Install into your existing environment:"
        echo "  source .noctprob-venv/bin/activate && pip install -e '.[${MODEL}]'"
        return 1 2>/dev/null || exit 1
    fi
    source "${VENV_PATH}/bin/activate"
    pip install --upgrade pip
    echo "Installing project with [${MODEL}] dependencies..."
    pip install -e ".[${MODEL}]"
    echo ""
    echo "Done! Environment '${MODEL}' is ready and activated."
else
    source "${VENV_PATH}/bin/activate"
    echo "Activated existing '${MODEL}' environment."
    echo "To reinstall deps: pip install -e '.[${MODEL}]'"
fi

echo "Python: $(which python)"
echo "transformers version: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"
