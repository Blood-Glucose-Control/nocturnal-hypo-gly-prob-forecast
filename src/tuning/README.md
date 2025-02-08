# Hyperparameter YAML Configuration Guide

This README explains how to define hyperparameters for each forecasting model in YAML format.

## General Structure of a YAML Entry

Each model should be added as a top-level key in the YAML file, with parameters specified as nested keys.

```yaml
ModelName:
  parameter_name:
    type: data_type
    values: [...]  # or range: [start, end] for numerical params
```

- **Use `values` for lists of discrete options** (categorical variables).
- **Use `range` for numerical parameters** (integers or floats).

## Supported Parameter Types & Best Practices

### List (Categorical Parameters)
- Use when the parameter takes a predefined set of values.
- Best for options like trend types, seasonal components, or model selection.

Example:
```yaml
AutoARIMA:
  seasonal_order:
    type: list
    values: [[0, 1, 1, 7], [1, 1, 1, 12]]
```
Best Practice:
- Include values that align with common seasonal cycles (e.g., 7 for weekly data, 12 for monthly).

### Integer (Discrete Ranges)
- Use when the parameter is a whole number within a logical range.
- Best for order selection (e.g., AR, MA terms in ARIMA), tree depths, or iteration counts.

Example:
```yaml
AutoARIMA:
  max_order:
    type: int
    range: [1, 5]
```
Best Practice:
- Choose a **reasonable range** (avoid excessive values that may cause overfitting or slow training).

---

### Float (Continuous Ranges)
- Use when the parameter can take decimal values within a range.
- Best for learning rates, smoothing factors, or penalty terms.

Example:
```yaml
ExpSmoothing:
  smoothing_level:
    type: float
    range: [0.1, 1.0, 0.1]  # Start, End, Step
```
Best Practice:
- Keep the **step size** small enough to explore the space efficiently but not too small (e.g., 0.01 is fine for tuning but 0.0001 may be excessive).

---

### Boolean (True/False Options)
- Use for settings that toggle features on or off.
- Best for enabling/disabling trend components, intercepts, or constraints.

Example:
```yaml
AutoARIMA:
  with_intercept:
    type: bool
```
Best Practice:
- Use booleans only when a setting is explicitly binary.
