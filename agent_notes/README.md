# Agent Notes Directory

This folder is now ordered by project priority using `P#-##` prefixes.

## Active Notes (Priority-Ordered)

### P0
- `P0-02_PII_SECRETS_AUDIT_REPORT.md`
- `P0-03_DATA_INVENTORY_SUMMARY.md`
- `P0-04_BACKUP_RECOVERY.md`
- `P0-04_DATA_MIRROR_COMPLETION_REPORT.md`
- `P0-06_P0-07_SETUP_COMPLETE.md`
- `P0-07_SECURITY_SETTINGS.md`
- `P0-08_BRANCH_CLEANUP_PLAN.md`
- `P0-09_P0-10_SESSION_ENDPOINT_HANDOFF.md`
- `P0-11_EXPERIMENTS_ARCH_RFC.md`
- `P0-P3_MASTER_REORG_PLAN.md` (cross-phase roadmap)

### P1
- `P1-05_P1-06_MLFLOW_OPTUNA_STORAGE_ARCHITECTURE.md`
- `P1-12_P1-20_EXECUTION_PLAN.md`
- `P1-12_PRIVATE_FORK_SYNC_POLICY.md`
- `P1-13_PRIVATE_GOVERNANCE_COST_PLAN.md`
- `P1-14_MODEL_STACK_DEEP_DIVE.md`
- `P1-15_PYDANTIC_SCHEMA_MIGRATION_KICKOFF.md`
- `P1-21_MODEL_RUNTIME_FLOW_AUDIT.md`
- `P1-30_CLEANUP_VALIDATION_MATRIX_KICKOFF.md`
- `P1-32_PYLANCE_RUNTIME_DIAGNOSTICS_TRIAGE_HANDOFF.md`
- `P1-34_WORKSPACE_TRIAGE_AND_SIMPLIFICATION_PLAN.md`
- `P1-37_CANONICAL_HANDOFF_AND_STATUS.md` (canonical consolidated handoff/status note)

## Historical / Archived Notes

Stale handoff and interim planning artifacts were preserved (not deleted) under:

- `archive_2026-08-07/`
- `archive_2026-08-18/p1-37/` (superseded P1-37 handoff/plan variants)

These are kept for traceability but should not be treated as current runbooks.

## Naming Convention

- `P0-04_...` = task tied to P0 item 4 in `project_tracking.csv`
- `P1-05_P1-06_...` = doc spanning multiple adjacent tasks
- `P0-P3_...` = roadmap across phases

## Maintenance Rule

When task state changes materially, update the corresponding `P#-##` file (or create it if missing) and archive obsolete variants instead of deleting them.
