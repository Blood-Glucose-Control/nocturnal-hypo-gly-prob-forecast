# P1 Execution Plan (Active P1 Tasks)

**Date:** 2026-08-13
**Scope:** Complete all P1 tasks in [`project_tracking.csv`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv)
**Inputs:** [`P0-11_EXPERIMENTS_ARCH_RFC.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P0-11_EXPERIMENTS_ARCH_RFC.md), [`P1-05_P1-06_MLFLOW_OPTUNA_STORAGE_ARCHITECTURE.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-05_P1-06_MLFLOW_OPTUNA_STORAGE_ARCHITECTURE.md)

---

## 1) P1 objective

Finish the minimum reliable platform needed for repeatable experiment execution and evaluation:

- private/public workflow governance is explicit,
- model/config/run interfaces are typed and stable,
- model runtime surface is minimal and unambiguous (non-runtime surfaces pruned by default),
- tracking/tuning are integrated (MLflow + Optuna),
- statistical rigor modules are productionized,
- experiment surfaces are collapsed per RFC.

---

## 2) Ordered work plan

### Phase A — Governance + interface contracts (Week 1)

1. **`private-fork-setup`**
   - Finalize sync policy doc (upstream→private cadence, private→upstream PR gate). ✅
   - Operationalize sync script + runbook. ✅
   - Adjust remote target to personal private repo workflow (Option C) and validate collaborator model.

2. **`private-governance-cost-plan`**
   - Finalize affordable governance decision (A/B/C options). ✅
   - Record owner decision + operational checklist. ✅
   - Current decision: **Option C** (personal private repo workflow).

3. **`model-adapter-protocol`**
   - Initial protocol artifacts shipped (`src/models/adapter.py`, naive baseline reference port).
   - Deep-dive architecture hardening required before close:
     - unify constructor path (`factory.py` vs registry),
     - reduce dead/legacy base exports,
     - define single contributor extension contract.
   - See [`P1-14_MODEL_STACK_DEEP_DIVE.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-14_MODEL_STACK_DEEP_DIVE.md).

4. **`model-runtime-flow-audit`**
   - Trace actual runtime call/import paths from training/eval entrypoints. ✅
   - Classify model-stack surfaces into keep / wire-with-owner / remove. ✅
   - Completion artifact: [`P1-21_MODEL_RUNTIME_FLOW_AUDIT.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-21_MODEL_RUNTIME_FLOW_AUDIT.md).

5. **`model-runtime-surface-prune`**
   - Apply runtime-first pruning policy from [`P1-14_MODEL_STACK_DEEP_DIVE.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-14_MODEL_STACK_DEEP_DIVE.md).
   - Remove or un-export non-runtime surfaces unless explicitly owned and test-backed.

6. **`model-extension-contract-doc`**
   - Publish one canonical contributor contract for adding models.
   - Make factory vs registry responsibilities explicit.

7. **`pydantic-config-schemas`**
   - Add typed schemas for model/data/eval configs.
   - Enforce validation on load paths.

### Phase B — Reproducibility + tracking (Week 2)

8. **`run-manifest-schema`**
   - Define required manifest contract fields.
   - Write manifest from all training/eval entrypoints.

9. **`mlflow-integration`**
   - Wire all entrypoints to MLflow tracking URI/project conventions.
   - Store artifact pointers + manifest linkage.

10. **`optuna-integration`**
   - Add `src/tuning/` with reusable search orchestration.
   - Connect trials to MLflow run IDs and manifests.

### Phase C — Evaluation hardening + architecture collapse (Week 3)

11. **`stats-rigor-module`**
   - Promote rebuttal A1–A9 code to durable public-safe module naming.
   - Separate generic statistical utilities from project-specific loaders.
   - Add focused tests + CI-fast subset.

12. **`experiments-collapse`**
   - Implement RFC mapping from [`P0-11_EXPERIMENTS_ARCH_RFC.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P0-11_EXPERIMENTS_ARCH_RFC.md).
   - Establish canonical flow:
     - `configs/experiments/*` typed specs,
     - `src/pipelines/*` runners,
     - `runs/` canonical artifacts (with compatibility support for existing `experiments/` for one release).

---

## 3) Dependencies / critical path

Critical path:

`model-adapter-protocol` → `model-runtime-flow-audit` → `model-runtime-surface-prune` → `model-extension-contract-doc` → `pydantic-config-schemas` → `run-manifest-schema` → (`mlflow-integration` + `optuna-integration`) → `experiments-collapse`

Parallelizable side lane:

`private-fork-setup` + `private-governance-cost-plan` can run in parallel with early technical tasks but must be finished before broad team rollout.

---

## 4) Definition of done for P1

P1 is done only when all conditions hold:

1. All active P1 task rows in [`project_tracking.csv`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/project_tracking.csv) are `done` (including runtime-flow audit/prune/contract tasks).
2. At least one end-to-end training/eval pipeline run:
   - validated config schema,
   - generated run manifest,
   - logged to MLflow,
   - (if tuning path) registered Optuna trial linkage.
3. Stats module migration merged with passing targeted tests.
4. Experiment architecture collapse merged with compatibility shim strategy documented.

---

## 5) Risks and mitigations

- **Scope creep from branch leftovers**
  Mitigation: keep P1 scoped to tracked tasks only; no opportunistic refactors outside dependencies.

- **Config migration churn**
  Mitigation: schema introduction first, auto-conversion helpers second, strict CI validation last.

- **Tracking inconsistency across scripts**
  Mitigation: centralize runner interface in `src/pipelines/` before broad script updates.

- **Pruning removes a hidden dependency**
  Mitigation: runtime flow audit first; prune only after keep/wire/remove classification; run focused model constructor/load/predict regression tests per removal PR.

- **Path/artifact breakage during collapse**
  Mitigation: one-release compatibility layer for legacy `experiments/` path and import shims.

---

## 6) Immediate next action

Start a dedicated implementation branch for P1 kickoff and complete:

1. `private-fork-setup` residual: confirm `private` remote points to chosen personal private repo and test sync job.
2. `model-runtime-surface-prune` first PR (remove non-runtime surface with tests).
3. `model-extension-contract-doc` draft so contributor path is unambiguous.
4. `pydantic-config-schemas` first implementation slice (schema package + one pilot model family).
5. `run-manifest-schema` draft contract so MLflow/Optuna wiring can start immediately after.
