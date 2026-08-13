# SLURM Training Scripts

This directory contains legacy SLURM launchers that are currently being rewired
as part of the P1 scripts reorganization.

## Current status

- `single_gpu.sh`, `multi_gpu.sh`, and `adaptive_resources.sh` now **fail fast**
  with a clear deprecation message.
- Reason: these launchers previously depended on legacy example entrypoints that
  were removed during scripts cleanup.
- Follow-up work will reconnect these launchers to a maintained training
  entrypoint.

## Temporary workflow

Until the SLURM launchers are rewired, use the maintained generic workflow script:

```bash
bash scripts/examples/run_holdout_generic_workflow.sh
```

Or call Python directly:

```bash
python scripts/examples/example_holdout_generic_workflow.py --help
```

## Other scripts in this folder

Model-specific SLURM scripts such as:

- `chronos2_eval.sh`
- `chronos2_finetune.sh`
- `chronos2_holdout_workflow.sh`
- `chronos2_time_covariate.sh`
- `per_patient_finetune.sh`
- `per_patient_finetune_all.sh`
- `toto_eval.sh`

remain available and should be treated as model/experiment-specific launchers.
