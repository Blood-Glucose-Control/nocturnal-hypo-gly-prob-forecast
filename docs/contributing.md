# Contributing to Nocturnal Hypo-Gly Prob Forecast

We welcome contributions to the Nocturnal Hypo-Gly Prob Forecast project! This document provides guidelines for contributing to this blood glucose control and forecasting research project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Contributing Guidelines](#contributing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Testing](#testing)
8. [Code Style](#code-style)
9. [Documentation](#documentation)
10. [Issue Reporting](#issue-reporting)
11. [Development Workflow](#development-workflow)
12. [Model Training Guidelines](#model-training-guidelines)
13. [Data Handling Guidelines](#data-handling-guidelines)

---

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- Be respectful and inclusive in language and actions
- Focus on constructive feedback and collaboration
- Respect different viewpoints and experiences
- Show empathy towards other community members
- Handle disagreements professionally

---

## Getting Started

### Prerequisites

- Python 3.8 or higher (Python 3.12 recommended)
- Git
- CUDA-capable GPU (recommended for model training)
- Sufficient disk space for datasets and model outputs

### Quick Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/nocturnal-hypo-gly-prob-forecast.git
   cd nocturnal-hypo-gly-prob-forecast
   ```

2. **Set up Environment**
   ```bash
   # Create and activate virtual environment (recommended)
   python -m venv .noctprob-venv
   source .noctprob-venv/bin/activate  # Linux/Mac
   # .noctprob-venv\Scripts\activate  # Windows
   
   # Alternative: Using conda (if you prefer)
   conda create -n nocturnal python=3.12
   conda activate nocturnal
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verify Setup**
   ```bash
   python -m pytest tests/ -v
   ```

---

## Development Setup

### Environment Configuration

1. **Create `.env` file** (copy from `.env.example` if available)
   ```bash
   # Add any necessary environment variables
   TTM_DEBUG=false
   MLFLOW_TRACKING_URI=./mlflow_experiments
   ```

2. **Set up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

3. **Configure MLflow** (for experiment tracking)
   ```bash
   # Start MLflow server locally
   cd mlflow_experiments
   mlflow server --host 127.0.0.1 --port 8080
   ```

### IDE Configuration

**VS Code (Recommended)**
- Install Python extension (includes Pylance language server)
- Install Ruff extension for linting/formatting
- Configure Python interpreter to your environment (`.noctprob-venv/bin/python`)
- Use provided workspace settings (`.vscode/settings.json`)
- Pylance configuration is provided in `pyrightconfig.json`

**Important**: Do not add `python.analysis.diagnosticSeverityOverrides` to VS Code settings when `pyrightconfig.json` exists, as this will cause configuration conflicts.

**PyCharm**
- Set Python interpreter to your venv environment
- Install Ruff plugin for linting/formatting
- Enable pytest as default test runner
- Configure code style to match project standards

---

## Project Structure

```
nocturnal-hypo-gly-prob-forecast/
├── src/                          # Source code
│   ├── data/                     # Data loading and processing
│   ├── train/                    # Model training pipelines
│   ├── eval/                     # Evaluation and metrics
│   ├── tuning/                   # Hyperparameter tuning
│   └── utils/                    # Utility functions
├── tests/                        # Test suite
├── configs/                      # Configuration files
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── models/                       # Trained model artifacts
├── results/                      # Experiment results
└── cache/                        # Data cache
```

### Key Components

- **TTM Training**: Time Series Transformer Model training pipeline
- **Data Pipeline**: Diabetes dataset processing and caching
- **Evaluation**: Model evaluation and benchmarking
- **Cache System**: Efficient data storage and retrieval

---

## Contributing Guidelines

### Types of Contributions

1. **Bug Fixes**: Fix issues in existing code
2. **Feature Development**: Add new functionality
3. **Model Improvements**: Enhance existing models or add new architectures
4. **Data Pipeline**: Improve data processing and loading
5. **Documentation**: Improve docs, add examples, write tutorials
6. **Testing**: Add or improve tests
7. **Performance**: Optimize code performance

### Contribution Areas

#### 🤖 **Model Development**
- Implement new foundation model architectures
- Improve existing TTM training pipeline
- Add new evaluation metrics
- Optimize model performance

#### 📊 **Data Science**
- Add support for new diabetes datasets
- Improve data preprocessing pipelines
- Enhance feature engineering
- Add data validation and quality checks

#### 🔧 **Infrastructure**
- Improve CI/CD pipelines
- Enhance caching systems
- Add monitoring and logging
- Optimize development workflows

#### 📚 **Research**
- Conduct hyperparameter studies
- Compare different model architectures
- Add new benchmarking datasets
- Improve evaluation methodologies

---

## Pull Request Process

### Before Starting

1. **Check existing issues** - Look for related work
2. **Create an issue** - Discuss your proposed changes
3. **Get feedback** - Ensure your approach aligns with project goals

### Development Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Follow the [code style guidelines](#code-style)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test categories
   pytest tests/data/
   pytest tests/train/
   
   # Run with coverage
   pytest tests/ --cov=src --cov-report=html
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new TTM architecture support"
   # Follow conventional commit format
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Requirements

- [ ] **Tests pass**: All existing tests continue to pass
- [ ] **New tests added**: For new functionality
- [ ] **Documentation updated**: README, docstrings, etc.
- [ ] **Code style**: Follows project formatting standards
- [ ] **No conflicts**: Branch is up-to-date with main
- [ ] **Descriptive PR**: Clear title and description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

---

## Testing

### Test Structure

```
tests/
├── test_data/           # Data pipeline tests
├── test_train/          # Training pipeline tests
├── test_eval/           # Evaluation tests
├── test_utils/          # Utility function tests
└── integration/         # End-to-end tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data/test_loaders.py

# Run with coverage
pytest --cov=src

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"

# Run in parallel (faster)
pytest -n auto
```

### Writing Tests

#### Unit Tests
```python
import pytest
from src.data.loaders import DataLoader

class TestDataLoader:
    def test_load_kaggle_data(self):
        loader = DataLoader("kaggle_brisT1D")
        data = loader.load()
        assert data is not None
        assert len(data) > 0

    def test_invalid_source_raises_error(self):
        with pytest.raises(ValueError):
            DataLoader("invalid_source")
```

#### Integration Tests
```python
@pytest.mark.slow
def test_end_to_end_training():
    """Test complete training pipeline with small dataset."""
    config = create_test_config()
    trainer = TTMTrainer(config)
    metrics = trainer.train()
    assert "eval_loss" in metrics
```

### Test Guidelines

- **Fast by default**: Unit tests should run quickly
- **Mark slow tests**: Use `@pytest.mark.slow` for integration tests
- **Use fixtures**: For common test data and setups
- **Test edge cases**: Include error conditions and boundary cases
- **Mock external dependencies**: Use mocking for external APIs/services

---

## Code Style

### Python Code Style

We use **Ruff** for linting and formatting:

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .
```

**Note**: Ruff is configured in your VS Code settings to run automatically on save. Pylance (Python language server) provides type checking and IntelliSense, configured via `pyrightconfig.json`.

#### Pylance Configuration

The project uses a **two-file configuration** for Pylance:

1. **`.vscode/settings.json`** - IDE-specific settings:
   - Python interpreter path
   - Ruff integration
   - Terminal configuration
   - Editor behavior

2. **`pyrightconfig.json`** - Language server settings:
   - Type checking rules
   - Diagnostic severity levels
   - Include/exclude paths
   - Python version and platform

**⚠️ Important**: When `pyrightconfig.json` exists, all diagnostic settings (`reportXxx` rules) must be configured there, not in VS Code settings. Adding `python.analysis.diagnosticSeverityOverrides` to VS Code settings will cause a configuration conflict.

### Style Guidelines

#### General Principles
- **PEP 8 compliant**: Follow Python style guide
- **Type hints**: Use type annotations for function signatures
- **Docstrings**: Document all public functions and classes
- **Clear naming**: Use descriptive variable and function names

#### Code Examples

```python
# Good
def process_patient_data(
    patient_id: str, 
    data: pd.DataFrame,
    config: ProcessingConfig
) -> ProcessedData:
    """Process patient time series data for model training.
    
    Args:
        patient_id: Unique identifier for patient
        data: Raw patient time series data
        config: Processing configuration parameters
        
    Returns:
        ProcessedData object ready for model training
        
    Raises:
        ValueError: If patient_id is invalid
        DataProcessingError: If data processing fails
    """
    if not patient_id:
        raise ValueError("Patient ID cannot be empty")
    
    try:
        processed = apply_preprocessing(data, config)
        return ProcessedData(patient_id, processed)
    except Exception as e:
        raise DataProcessingError(f"Failed to process {patient_id}") from e

# Avoid
def proc(id, df, cfg):  # Bad: unclear names, no types, no docs
    return stuff
```

#### Import Organization
```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third party
import numpy as np
import pandas as pd
import torch
from transformers import Trainer

# Local imports
from src.data.loaders import DataLoader
from src.utils.logging import get_logger
```

### Configuration Files

#### YAML Style
```yaml
# Use consistent indentation (2 spaces)
model:
  type: "ttm"
  path: "ibm-granite/granite-timeseries-ttm-r2"
  config:
    context_length: 512
    forecast_length: 96

# Use descriptive keys
training:
  batch_size: 128
  learning_rate: 1e-4
  num_epochs: 10
```

---

## Documentation

### Documentation Types

1. **API Documentation**: Function and class docstrings
2. **User Guides**: How-to guides and tutorials
3. **README Files**: Module and project overviews
4. **Configuration Docs**: YAML configuration guides

### Docstring Format

Use **Google style** docstrings:

```python
def fine_tune_model(
    model_path: str,
    data_config: DataConfig,
    training_config: TrainingConfig
) -> Dict[str, float]:
    """Fine-tune a foundation model on patient data.
    
    This function implements the complete fine-tuning pipeline including
    data loading, model setup, training, and evaluation.
    
    Args:
        model_path: Path to pre-trained model or HuggingFace identifier
        data_config: Configuration for data loading and preprocessing
        training_config: Configuration for training hyperparameters
        
    Returns:
        Dictionary containing final training metrics including:
        - eval_loss: Final validation loss
        - eval_accuracy: Final validation accuracy
        - training_time: Total training time in seconds
        
    Raises:
        ModelNotFoundError: If model_path is invalid
        DataLoadingError: If data loading fails
        TrainingError: If training process fails
        
    Example:
        >>> data_config = DataConfig(source="kaggle_brisT1D")
        >>> train_config = TrainingConfig(epochs=10, batch_size=64)
        >>> metrics = fine_tune_model("ttm-model", data_config, train_config)
        >>> print(f"Final loss: {metrics['eval_loss']}")
    """
```

### Documentation Guidelines

- **Clear and concise**: Explain what, why, and how
- **Include examples**: Show practical usage
- **Document edge cases**: Explain error conditions
- **Keep updated**: Update docs when code changes

---

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment information**
   - Python version
   - Operating system
   - GPU information (if relevant)
   - Package versions (`pip list`)

2. **Steps to reproduce**
   - Minimal code example
   - Input data characteristics
   - Configuration used

3. **Expected vs actual behavior**
   - What should happen
   - What actually happens
   - Error messages/tracebacks

4. **Additional context**
   - Screenshots if relevant
   - Log files
   - Performance implications

### Feature Requests

For new features:

1. **Clear problem statement**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Use cases**: Who would use this and how?
4. **Implementation ideas**: Technical approach (optional)

### Using Issue Templates

Use the provided GitHub issue templates:
- Bug Report: `.github/ISSUE_TEMPLATE/bug_report.md`
- Feature Request: `.github/ISSUE_TEMPLATE/feature_request.md`

---

## Development Workflow

### Git Workflow

We use a **feature branch** workflow:

```bash
# 1. Start from main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/new-model-architecture

# 3. Make changes and commit
git add .
git commit -m "feat: add LSTM architecture support"

# 4. Push branch
git push origin feature/new-model-architecture

# 5. Create PR on GitHub

# 6. After merge, cleanup
git checkout main
git pull origin main
git branch -d feature/new-model-architecture
```

### Commit Message Format

Use **Conventional Commits**:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix  
- `docs`: Documentation changes
- `style`: Formatting changes
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Build/tool changes

**Examples:**
```bash
feat(data): add support for Aleppo dataset
fix(training): resolve TTM memory leak during evaluation
docs(api): update model configuration guide
refactor(cache): restructure data caching system
test(integration): add end-to-end training tests
```

### Branch Naming

- `feature/description`: New features
- `fix/issue-description`: Bug fixes
- `docs/topic`: Documentation updates
- `refactor/component`: Code restructuring

---

## Model Training Guidelines

### Adding New Model Architectures

When contributing new models, follow the [Foundation Model Template](./foundation_model_template.md):

1. **Use the standard structure**:
   ```
   src/train/your_model/
   ├── core/trainer.py
   ├── data/loaders.py
   ├── evaluation/metrics.py
   ├── config/schema.py
   └── cli/runner.py
   ```

2. **Follow the base interfaces**:
   ```python
   class YourModelTrainer(BaseFoundationTrainer):
       def _create_data_loader(self): ...
       def _create_model_factory(self): ...
       def _create_evaluator(self): ...
   ```

3. **Provide comprehensive configuration**:
   ```yaml
   # configs/models/your_model_default.yaml
   model:
     type: "your_model"
     path: "path/to/pretrained/model"
   
   data:
     source_name: "kaggle_brisT1D"
     batch_size: 64
   
   training:
     num_epochs: 10
     learning_rate: 1e-4
   ```

### Experiment Management

- **Use descriptive names**: `ttm_kaggle_finetuning_v1`
- **Track all parameters**: Model, data, hyperparameters
- **Save reproducible configs**: YAML files with exact settings
- **Document results**: README in results directory

### Performance Guidelines

- **Profile memory usage**: Monitor GPU memory during training
- **Benchmark training speed**: Track time per epoch
- **Test on multiple datasets**: Verify generalization
- **Compare baselines**: Include comparison with existing models

---

## Data Handling Guidelines

### Data Privacy and Security

- **No personal data**: Ensure all datasets are properly anonymized
- **Respect licenses**: Check data usage rights and attribution
- **Secure storage**: Use appropriate access controls for sensitive data

### Data Pipeline Best Practices

1. **Validate inputs**: Check data format and quality
2. **Handle missing data**: Implement robust imputation strategies
3. **Document preprocessing**: Clear documentation of all transformations
4. **Cache efficiently**: Use the project's caching system
5. **Version datasets**: Track data versions and changes

### Adding New Datasets

1. **Create data loader**:
   ```python
   class NewDatasetLoader(BaseDataLoader):
       def load_raw_data(self): ...
       def preprocess(self, data): ...
       def validate(self, data): ...
   ```

2. **Add configuration**:
   ```yaml
   # configs/datasets/new_dataset.yaml
   source_name: "new_dataset"
   format: "csv"
   columns:
     timestamp: "datetime"
     target: "bg_level"
   ```

3. **Update documentation**: Add dataset description and usage

---

## Troubleshooting Common Issues

### VS Code / Pylance Configuration Conflicts

**Problem**: Error message `'python.analysis.diagnosticSeverityOverrides' cannot be set when a pyrightconfig.json or pyproject.toml is being used.`

**Solution**: 
1. Remove any `python.analysis.diagnosticSeverityOverrides` from `.vscode/settings.json`
2. Configure diagnostic settings in `pyrightconfig.json` instead
3. Reload VS Code window: `Ctrl+Shift+P` → "Developer: Reload Window"

**Problem**: Red underlines everywhere / Import resolution issues

**Solution**:
1. Ensure your Python interpreter is set to `.noctprob-venv/bin/python`
2. Check that `pyrightconfig.json` includes your `src` directory
3. Restart Python language server: `Ctrl+Shift+P` → "Python: Restart Language Server"

### Virtual Environment Issues

**Problem**: VS Code not finding your virtual environment

**Solution**:
1. Create the environment: `python -m venv .noctprob-venv`
2. Activate it: `source .noctprob-venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Set interpreter in VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter"

### Testing Issues

**Problem**: Tests not running or pytest not found

**Solution**:
1. Ensure pytest is installed: `pip install pytest`
2. Check VS Code test configuration in settings.json
3. Refresh test discovery: `Ctrl+Shift+P` → "Test: Refresh Tests"

---

## Getting Help

### Resources

- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` and `scripts/` directories
- **Tests**: Look at test files for usage examples
- **Issues**: Search existing GitHub issues

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Comments**: For code-specific questions

### Maintainer Response Times

- **Bug reports**: Within 48-72 hours
- **Feature requests**: Within 1 week
- **Pull reviews**: Within 3-5 business days

---

## Recognition

### Contributors

All contributors will be:
- Listed in the project README
- Recognized in release notes
- Credited in any resulting publications (for significant contributions)

### Types of Recognition

- **Code contributors**: Implementation and bug fixes
- **Documentation contributors**: Improving docs and tutorials
- **Research contributors**: Experimental validation and analysis
- **Community contributors**: Helping other users and maintaining discussions

---

Thank you for contributing to the Nocturnal Hypo-Gly Prob Forecast project! Your contributions help advance blood glucose control research and improve outcomes for diabetes patients.

For questions about contributing, please open an issue or reach out to the maintainers.