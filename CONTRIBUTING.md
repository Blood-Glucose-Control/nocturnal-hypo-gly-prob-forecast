# Contributing to Nocturnal Hypo-Gly Prob Forecast

Thank you for your interest in contributing to the Nocturnal Hypo-Gly Prob Forecast project! 🩸📈

This project focuses on blood glucose control and forecasting research using foundation models like Time Series Transformers (TTM).

## Quick Start

1. **Fork and clone** the repository
2. **Set up environment**: `python -m venv .noctprob-venv && source .noctprob-venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run tests**: `pytest tests/`
5. **Create feature branch**: `git checkout -b feature/your-feature`
6. **Make changes** and commit with conventional commit format
7. **Submit pull request**

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
