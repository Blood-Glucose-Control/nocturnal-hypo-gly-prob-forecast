# Data Inventory Summary (Post-Prune + Policy Validation)
**Date**: 2026-08-07 03:26 UTC
**Status**: ✅ `optimizer.pt` + `train.pkl` pruned; inventories regenerated

## 1) Snapshot (Current Files)

| Inventory File | Rows (incl. header) | Data Rows | Total Size Tracked | Scope |
|---|---:|---:|---:|---|
| `trained_models_inventory.csv` | 19,031 | 19,030 | 129.56 GiB | `trained_models/` plus a small `experiments/` subset |
| `experiments_inventory.csv` | 669 | 668 | 1.72 GiB | `experiments/nocturnal_forecasting*` only |

**Scope note**: `trained_models_inventory.csv` currently includes **673** `experiments/` files (1.72 GiB). Avoid double-counting with `experiments_inventory.csv`.

## 2) Cleanup Executed

Prune was executed via [prune_retention_artifacts.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/prune_retention_artifacts.py):

- Removed files: **1,637**
- Space reclaimed: **136.81 GiB**
- Removed `optimizer.pt`: **1,528 files / 81.59 GiB**
- Removed `train.pkl`: **109 files / 55.23 GiB**
- Failures: **0**

Report: [retention_prune_report.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/retention_prune_report.json)

## 3) Trained Model Storage (trained_models/ only)

### Retention Tier Breakdown

| Tier | Files | Size |
|---|---:|---:|
| paper-critical | 6,198 | 100.73 GiB |
| active-work | 11,959 | 26.99 GiB |
| archivable | 43 | 0.10 GiB |
| deletable | 157 | 0.01 GiB |
| **Total** | **18,357** | **127.84 GiB** |

### Model Types by Size (largest first)

| Model Type | Files | Size |
|---|---:|---:|
| TimesFM | 592 | 64.15 GiB |
| Toto | 444 | 29.34 GiB |
| Moment | 637 | 12.74 GiB |
| PatchTST | 615 | 7.38 GiB |
| TTM | 11,157 | 4.77 GiB |
| DeepAR | 615 | 2.78 GiB |
| Chronos2 | 2,156 | 2.27 GiB |
| Temporal Fusion Transformer | 734 | 1.46 GiB |
| MOIRAI | 488 | 1.25 GiB |
| TIDE | 565 | 0.77 GiB |

## 4) Composition (trained_models/)

- Binary artifacts (`.pt`, `.pth`, `.ckpt`, `.pkl`, `.safetensors`, `.bin`): **127.41 GiB** across 9,218 files
- Non-binary metadata/results/config/logs: **0.43 GiB** across 9,139 files

Residual high-impact target:
- `cached_predictions.pkl`: **101 files / 7.54 GiB**

## 5) Single Retention Policy (Spec + Validator)

Added canonical policy spec:
- [model_artifact_retention_policy.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/configs/retention/model_artifact_retention_policy.json)

Added validator:
- [validate_model_artifact_retention.py](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/validate_model_artifact_retention.py)

Validation run result (after prune):
- Runs checked: **215**
- Compliant: **178**
- With violations: **37**
- Forbidden files found (`optimizer.pt`, `train.pkl`): **0**

Report: [retention_policy_validation_report.json](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/retention_policy_validation_report.json)

Most common remaining violations are missing top-level run artifacts (`training_metadata.json`, `forecasts`, `predictions`, `split_metadata.json`, `model_config.yaml`) on older/incomplete runs.

## 6) Operational Commands

Dry-run prune:
```bash
python scripts/prune_retention_artifacts.py
```

Execute prune:
```bash
python scripts/prune_retention_artifacts.py --execute
```

Run policy validation:
```bash
python scripts/validate_model_artifact_retention.py
```

Fail CI/pre-upload when violations exist:
```bash
python scripts/validate_model_artifact_retention.py --fail-on-violations
```
