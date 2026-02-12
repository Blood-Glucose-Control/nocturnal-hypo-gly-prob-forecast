# Contributing to Nocturnal Hypo-Gly Prob Forecast

Thank you for your interest in contributing to the Nocturnal Hypo-Gly Prob Forecast project! 🩸📈

This project focuses on blood glucose control and forecasting research using foundation models like Time Series Transformers (TTM).

## Quick Start

1. **Fork and clone** the repository
2. **Set up environment**: `python -m venv .noctprob-venv && source .noctprob-venv/bin/activate`
3. **Install dependencies**: `pip install -e .`
4. **Run tests**: `pytest tests/`
5. **Create feature branch**: `git checkout -b feature/your-feature`
6. **Make changes** and commit with conventional commit format
7. **Submit pull request**

## Environment Setup

This project uses **two types of virtual environments**:

### General Development (`.noctprob-venv`)
For notebooks, data exploration, and non-model work:
```bash
python -m venv .noctprob-venv
source .noctprob-venv/bin/activate
pip install -e .
```

### Model-Specific Environments (`.venvs/<model>/`)
Different foundation models require different package versions (e.g., `transformers`). Use model-specific environments for training and inference:

```bash
# Set up and activate a model environment (creates venv on first run)
source scripts/setup_model_env.sh ttm
source scripts/setup_model_env.sh sundial

# Deactivate when done
deactivate
```

Available models are defined in `pyproject.toml` under `[project.optional-dependencies]`. To add a new model, add an entry there and run:
```bash
source scripts/setup_model_env.sh <new-model>
```

**When to use which:**
| Task | Environment |
|------|-------------|
| Data exploration, notebooks | `.noctprob-venv` |
| Running tests | `.noctprob-venv` |
| TTM model training/inference | `source scripts/setup_model_env.sh ttm` |
| Other model training | `source scripts/setup_model_env.sh <model>` |

## What Can You Contribute?

### 🤖 **Model Development**
- Implement new foundation model architectures
- Improve TTM training pipelines
- Add new evaluation metrics
- Optimize model performance

### 📊 **Data Science**
- Add support for new diabetes datasets
- Improve data preprocessing
- Enhance feature engineering
- Add data validation

### 🔧 **Infrastructure**
- Improve CI/CD pipelines
- Enhance caching systems
- Add monitoring and logging
- Development workflow improvements

### 📚 **Documentation & Research**
- Write tutorials and guides
- Conduct hyperparameter studies
- Add benchmarking datasets
- Improve evaluation methodologies

## Development Guidelines

- **Follow the [Foundation Model Template](docs/foundation_model_template.md)** for new architectures
- **Use conventional commits**: `feat:`, `fix:`, `docs:`, etc.
- **Add tests** for new functionality
- **Update documentation** as needed
- **Run code formatting**: `ruff format .` and `ruff check .`
- **Configure VS Code properly**: Use `pyrightconfig.json` for Pylance settings, not VS Code `diagnosticSeverityOverrides`

## Code Quality Standards

- ✅ All tests pass (`pytest`)
- ✅ Code follows style guidelines (`ruff check`)
- ✅ Type hints for function signatures
- ✅ Docstrings for public functions
- ✅ No merge conflicts with main branch

## Issue Reporting

- **Bugs**: Use the bug report template
- **Features**: Use the feature request template
- **Questions**: Use GitHub Discussions

## Recognition

Contributors are recognized in:
- Project README
- Release notes
- Research publications (for significant contributions)

---

📖 **Full Contributing Guide**: See [docs/contributing.md](docs/contributing.md) for detailed guidelines.

🚀 **Getting Started**: Check out the [TTM Reorganization Plan](docs/ttm_reorganization_plan.md) to understand the current architecture improvements.

For questions, open an issue or reach out to the maintainers. Thank you for helping advance diabetes research! 🙏
