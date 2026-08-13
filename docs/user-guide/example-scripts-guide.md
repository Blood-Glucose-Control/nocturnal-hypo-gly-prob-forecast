# Example Scripts Guide

This guide lists the maintained scripts in `scripts/examples/` and
`scripts/experiments/` and when to use each one.

## Active examples

| Script | Purpose | Typical use |
|---|---|---|
| `example_holdout_generic_workflow.py` | End-to-end holdout workflow across model families | Main example for training/eval workflow |
| `run_holdout_generic_workflow.sh` (`scripts/experiments/`) | Local shell wrapper for the generic holdout workflow | Repeatable local runs with environment variables |
| `example_chronos2_finetune.py` | Focused Chronos-2 fine-tuning walkthrough | Model-specific Chronos-2 experimentation |
| `example_data_holdout_system.py` | Demonstrates holdout config generation/validation/loading APIs | Data split and holdout debugging |
| `example_load_holdout_data.py` | Minimal holdout loading example | Quick data-access sanity checks |
| `ttm_holdout_workflow.py` | TTM-specific workflow variant | Legacy TTM-specific experimentation |

## Canonical workflow example

Use the generic workflow as the default entrypoint:

```bash
python scripts/examples/example_holdout_generic_workflow.py \
  --model-type chronos2 \
  --datasets brown_2019 lynch_2022 \
  --config-dir configs/data/holdout_10pct
```

Or with the local wrapper:

```bash
MODEL_TYPE=chronos2 \
DATASETS="brown_2019 lynch_2022" \
CONFIG_DIR="configs/data/holdout_10pct" \
bash scripts/experiments/run_holdout_generic_workflow.sh
```

## Notes on legacy examples

Several older example scripts were removed during P1 scripts cleanup because they
depended on model-base surfaces that were already pruned from runtime. If you need
the old behavior, use git history for reference and prefer rebuilding on top of
the maintained scripts listed above.
