# LoRA Compatibility Notes

LoRA support in this repository is **model-family specific**, not a universal
base-framework feature.

## Current state

- The shared base class does **not** expose a generic `supports_lora()` contract.
- Any LoRA behavior is implemented inside specific model-family code paths.
- Non-transformer families (for example mixer-style architectures) should use
  standard fine-tuning controls such as learning rate, batch size, and optional
  backbone freezing where supported.

## Practical guidance

Before enabling LoRA for a family, verify the implementation in that family's
`config.py` and `model.py` modules rather than assuming support from the base
framework.

For schema-routed workflows, ensure any LoRA-related config keys are explicitly
represented in:

- [`src/config/schemas/model_configs.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/src/config/schemas/model_configs.py)
- the family runtime adapter in the same module
- focused tests in
  [`tests/workflows/forecasting/test_model_config_schema_loader.py`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/tests/workflows/forecasting/test_model_config_schema_loader.py)
