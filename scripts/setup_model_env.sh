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
# Available models:
#   - Optional dependency groups in pyproject.toml (e.g., ttm, sundial, moirai)
#   - AutoGluon-backed model aliases:
#       autogluon, chronos2, tide, deepar, patchtst, tft, naive_baseline, statistical
#     These all resolve to the shared .venvs/autogluon environment.
#   - Darts-backed model aliases:
#       darts, tsmixer
#     These resolve to the shared .venvs/darts environment.

MODEL="${1:?Usage: source scripts/setup_model_env.sh <model>}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "Error: Must be run from within a git repository"; return 1 2>/dev/null || exit 1; }
VENVS_DIR="${REPO_ROOT}/.venvs"

is_autogluon_alias() {
    case "${1}" in
        autogluon|chronos2|tide|deepar|patchtst|tft|naive_baseline|statistical) return 0 ;;
        *) return 1 ;;
    esac
}

is_darts_alias() {
    case "${1}" in
        darts|tsmixer) return 0 ;;
        *) return 1 ;;
    esac
}

if is_autogluon_alias "${MODEL}"; then
    VENV_NAME="autogluon"
    DEP_GROUP="autogluon"
elif is_darts_alias "${MODEL}"; then
    VENV_NAME="darts"
    DEP_GROUP="darts"
else
    VENV_NAME="${MODEL}"
    DEP_GROUP="${MODEL}"
fi

VENV_PATH="${VENVS_DIR}/${VENV_NAME}"

# Model-specific Python version overrides
# Most models use 3.12; some need older versions for dependency compatibility.
case "${DEP_GROUP}" in
    timegrad) PYTHON_VERSION="3.11" ;;  # pytorchts requires pandas<2.0 (no 3.12 wheel)
    *)        PYTHON_VERSION="3.12" ;;
esac

# Find the required Python version
if command -v "python${PYTHON_VERSION}" &>/dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif [ "${PYTHON_VERSION}" = "3.12" ] && [ -x "${REPO_ROOT}/.noctprob-venv/bin/python" ]; then
    PYTHON_CMD="${REPO_ROOT}/.noctprob-venv/bin/python"
else
    echo "Error: Python ${PYTHON_VERSION} not found (required for ${MODEL} / ${DEP_GROUP})."
    echo "Install python${PYTHON_VERSION} or use: brew install python@${PYTHON_VERSION}"
    return 1 2>/dev/null || exit 1
fi

echo "Using Python: ${PYTHON_CMD} ($(${PYTHON_CMD} --version 2>&1))"

install_model_dependencies() {
    if [ "${DEP_GROUP}" = "moment" ]; then
        echo "Installing project with [moment] dependencies..."
        if pip install -e ".[moment]"; then
            return 0
        fi

        # Fallback for environments where momentfm dependency resolution fails.
        # Keep project dependencies editable, then install momentfm without
        # re-resolving the full dependency graph.
        echo "Standard [moment] install failed. Applying momentfm fallback..."
        pip install -e . || return 1
        pip install --no-deps momentfm || return 1
        return 0
    fi

    echo "Installing project with [${DEP_GROUP}] dependencies..."
    pip install -e ".[${DEP_GROUP}]" || return 1
}

# Validate model name exists in pyproject.toml [project.optional-dependencies]
OPT_DEPS=$(sed -n '/^\[project.optional-dependencies\]/,/^\[/p' "${REPO_ROOT}/pyproject.toml" 2>/dev/null | tail -n +2)
if ! echo "${OPT_DEPS}" | grep -qF "${DEP_GROUP} = ["; then
    echo "Error: Model '${MODEL}' resolved to dependency group '${DEP_GROUP}', which is not found in pyproject.toml [project.optional-dependencies]"
    echo "Available models:"
    echo "${OPT_DEPS}" | grep -F ' = [' | sed 's/ = \[.*//'
    echo ""
    echo "AutoGluon aliases (all use .venvs/autogluon):"
    echo "  chronos2 tide deepar patchtst tft naive_baseline statistical"
    echo "Darts aliases (all use .venvs/darts):"
    echo "  tsmixer"
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "${VENV_PATH}" ]; then
    if [ "${MODEL}" = "${VENV_NAME}" ]; then
        echo "Creating new venv for '${MODEL}' at ${VENV_PATH}..."
    else
        echo "Creating shared '${VENV_NAME}' venv for model '${MODEL}' at ${VENV_PATH}..."
    fi

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
        echo "  3. Use conda: conda create -n ${VENV_NAME} python=${PYTHON_VERSION} && conda activate ${VENV_NAME} && pip install -e '.[${DEP_GROUP}]'"
        return 1 2>/dev/null || exit 1
    fi

    source "${VENV_PATH}/bin/activate"
    pip install --upgrade pip
    install_model_dependencies || {
        echo "Error: Failed to install dependencies"
        return 1 2>/dev/null || exit 1
    }
    echo ""
    if [ "${MODEL}" = "${VENV_NAME}" ]; then
        echo "Done! Environment '${VENV_NAME}' is ready and activated."
    else
        echo "Done! Shared environment '${VENV_NAME}' is ready and activated for model '${MODEL}'."
    fi
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
    if [ "${MODEL}" = "${VENV_NAME}" ]; then
        echo "Activated existing '${VENV_NAME}' environment."
    else
        echo "Activated existing shared '${VENV_NAME}' environment for model '${MODEL}'."
    fi
    if [ "${DEP_GROUP}" = "moment" ]; then
        echo "To reinstall deps: pip install -e '.[moment]' (fallback: pip install -e . && pip install --no-deps momentfm)"
    else
        echo "To reinstall deps: pip install -e '.[${DEP_GROUP}]'"
    fi
fi

echo "Python: $(which python)"
echo "transformers version: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"
