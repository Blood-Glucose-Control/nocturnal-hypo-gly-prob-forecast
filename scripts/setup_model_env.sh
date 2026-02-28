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
#   ttm, sundial, timesfm, chronos2

MODEL="${1:?Usage: source scripts/setup_model_env.sh <model>}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "Error: Must be run from within a git repository"; return 1 2>/dev/null || exit 1; }
VENVS_DIR="${REPO_ROOT}/.venvs"
VENV_PATH="${VENVS_DIR}/${MODEL}"

# Model-specific Python version overrides
# Most models use 3.12; some need older versions for dependency compatibility.
case "${MODEL}" in
    timegrad) PYTHON_VERSION="3.11" ;;  # pytorchts requires pandas<2.0 (no 3.12 wheel)
    *)        PYTHON_VERSION="3.12" ;;
esac

# Find the required Python version
if command -v "python${PYTHON_VERSION}" &>/dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif [ "${PYTHON_VERSION}" = "3.12" ] && [ -x "${REPO_ROOT}/.noctprob-venv/bin/python" ]; then
    PYTHON_CMD="${REPO_ROOT}/.noctprob-venv/bin/python"
else
    echo "Error: Python ${PYTHON_VERSION} not found (required for ${MODEL})."
    echo "Install python${PYTHON_VERSION} or use: brew install python@${PYTHON_VERSION}"
    return 1 2>/dev/null || exit 1
fi

echo "Using Python: ${PYTHON_CMD} ($(${PYTHON_CMD} --version 2>&1))"

# Validate model name exists in pyproject.toml [project.optional-dependencies]
OPT_DEPS=$(sed -n '/^\[project.optional-dependencies\]/,/^\[/p' "${REPO_ROOT}/pyproject.toml" 2>/dev/null | tail -n +2)
if ! echo "${OPT_DEPS}" | grep -qF "${MODEL} = ["; then
    echo "Error: Model '${MODEL}' not found in pyproject.toml [project.optional-dependencies]"
    echo "Available models:"
    echo "${OPT_DEPS}" | grep -F ' = [' | sed 's/ = \[.*//'
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating new venv for '${MODEL}' at ${VENV_PATH}..."

    # Try venv first, fall back to virtualenv if ensurepip not available
    if ${PYTHON_CMD} -m venv "${VENV_PATH}" 2>/dev/null; then
        echo "Created environment using venv"
    elif command -v virtualenv &>/dev/null; then
        echo "venv failed (python3-venv not installed), using virtualenv..."
        virtualenv -p "${PYTHON_CMD}" "${VENV_PATH}" || {
            echo "Error: Failed to create virtual environment with virtualenv"
            return 1 2>/dev/null || exit 1
        }
    else
        echo "Error: Failed to create virtual environment."
        echo "venv requires python3-venv package, and virtualenv is not installed."
        echo ""
        echo "Options:"
        echo "  1. Install virtualenv: pip install virtualenv"
        echo "  2. Ask admin to install: sudo apt install python3.12-venv"
        echo "  3. Use conda: conda create -n ${MODEL} python=3.12 && conda activate ${MODEL} && pip install -e '.[${MODEL}]'"
        return 1 2>/dev/null || exit 1
    fi

    source "${VENV_PATH}/bin/activate"
    pip install --upgrade pip
    echo "Installing project with [${MODEL}] dependencies..."
    pip install -e ".[${MODEL}]" || { echo "Error: Failed to install dependencies"; return 1 2>/dev/null || exit 1; }
    echo ""
    echo "Done! Environment '${MODEL}' is ready and activated."
elif [ ! -f "${VENV_PATH}/bin/activate" ]; then
    # Directory exists but is broken (no activate script)
    echo "Warning: Found broken venv at ${VENV_PATH} (missing activate script)"
    echo "Removing and recreating..."
    rm -rf "${VENV_PATH}"
    # Re-run this script to create fresh
    source "${REPO_ROOT}/scripts/setup_model_env.sh" "${MODEL}"
    return $? 2>/dev/null || exit $?
else
    source "${VENV_PATH}/bin/activate"
    echo "Activated existing '${MODEL}' environment."
    echo "To reinstall deps: pip install -e '.[${MODEL}]'"
fi

echo "Python: $(which python)"
echo "transformers version: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"
