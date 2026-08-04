# Nocturnal Hypo-Gly Prob Forecast — Cleanup Week + New Phase Plan

**Status:** v1 — locked charter (A2) + license direction (B1); remaining decisions (D3–D7) captured as recommended defaults, revisable during execution.
**Repo:** [Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast](https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast) — currently on `neurips-rebuttal`
**Author view:** Written from a seasoned ML Engineering Manager's perspective.

---

## 1. Problem statement

The repo is (a) publication-critical work that just went through NeurIPS rebuttal, (b) a public research artifact that is *de facto* the team's day-to-day workspace, and (c) the intended future host of a HuggingFace-published benchmark. These three roles are in tension:

- The public repo currently exposes team activity (branches, artifacts, scripts) that the team wants private going forward.
- 246 GB of trained models and 1.8 GB of experiment outputs live only on the cluster with no immutable backup — high loss risk.
- The publication-critical statistical rigor pipeline (A1–A9 with Friedman/Wilcoxon/block-bootstrap/CD-diagrams) lives only on the `neurips-rebuttal` branch — not on `main`, not tagged, not CI'd.
- The benchmark's stated future (HuggingFace-hostable, external researchers plugging in private models, hyperparameter budget tables, statistical CIs baked in) requires infrastructure that mostly doesn't exist yet: no Optuna, no MLflow integration, no `ModelAdapter` public contract, no data-version pinning, no run manifest schema.

The goal is one week of safety-first cleanup + governance, followed by a phase of building the benchmark platform properly.

---

## 2. Guiding principles

1. **Preserve first, restructure second.** Nothing gets reorganized before the rebuttal artifacts are tagged and the 246 GB is inventoried and mirrored.
2. **Public repo is a showroom, not a workshop.** Team's day-to-day moves to a private fork; public repo contains the harness, adapter API, small examples, and the benchmark surface.
3. **Every run is a manifest.** Code SHA, data version, seed, hyperparameters, and budget (wall time, GPU-hours, FLOPs) captured automatically on every training/eval run.
4. **Bootstrap the future in the old world.** Land the rebuttal statistical pipeline into `main` as first-class modules with CI, so the paper's rigor becomes the benchmark's default.
5. **Agent-safe by construction.** Explicit read-only paths, protected mounts, and `AGENTS.md`/`CLAUDE.md` conventions at every level that matters.

---

## 3. Decisions

| # | Decision | Chosen | Notes |
|---|---|---|---|
| D1 | Public repo charter | **A2 — Reference implementation** | Harness + adapter API + 2–3 reference models (naive baseline, Chronos-2, TimesFM). Competitive models stay in private fork. |
| D2 | License strategy | **B1 — Apache-2.0 for code** | Code Apache-2.0. Each open T1D dataset stays under its original license (Aleppo, Brown, Lynch, Tamborlane, Colas, Hall, Kaggle BrisT1D). Leaderboard entries CC-BY-4.0. Gluroo data never touches this repo. |
| D3 | Tracking stack | **C1 — Optuna + MLflow** | Self-hosted MLflow (already scaffolded). Optuna for HP search with pruning. W&B mirror only if we want polished paper reports later. |
| D4 | Data version control | **D1 public + D3 private** | HF Datasets for public benchmark splits (commit-pinned). Custom parquet manifest for team's private data. |
| D5 | Model interface | **E1 short-term → E2 by benchmark release** | Keep per-model venvs as pip extras now; containerize per model family before public benchmark launch. |
| D6 | Private fork model | **GitHub fork under Blood-Glucose-Control** | Cleanest upstream sync via GitHub's fork mechanism. Nightly upstream→fork sync action; curated fork→upstream PRs only. |
| D7 | 246 GB retention | **Tiered** | paper-critical (immutable, mirrored forever) / active-work (mirrored, retained 6 months rolling) / archivable (mirrored, retained 90 days) / deletable (removed after inventory review). |

Revisit D3/D4/D5 during execution if constraints change (e.g., W&B credits become available, HF datasets access limits hit, upstream `transformers` versions converge).

---

## 4. Prioritized todo list (SQL-tracked)

Todos are stored in the SQL `todos` table with dependencies in `todo_deps`. See §5 for status tracking guidance. The IDs below match SQL rows.

### P0 — This week (safety + governance)
- `paper-tag-immutable`: Tag `paper-v1` on `neurips-rebuttal`, push to backup remote(s), verify tag SHA is preserved outside the repo (e.g., cluster README, external doc).
- `pii-secrets-audit`: `git log --all -p` scan for PHI/tokens/keys across history; enable GitHub secret scanning + push protection; document any incidents and remediation.
- `data-inventory`: Produce `trained_models_inventory.csv` (path, size, model, dataset, run_id, retention_tier, sha256) covering all of `trained_models/` and `experiments/`. Classify into paper-critical / active-work / archivable / deletable.
- `data-mirror-plan`: Choose object-storage target (institutional S3, HF private repo, or cluster mirror), write mirror script, dry-run on paper-critical subset.
- `land-rebuttal-analyses`: Cherry-pick/rebase `scripts/analysis/rebuttal_neurips2026/` + `src/evaluation/storage.py` deltas from `neurips-rebuttal` to `main` via a documented PR series; keep `METHODOLOGY.md` in the tree.
- `agent-safety-conventions`: Add repo-root `AGENTS.md` (mirrored as `CLAUDE.md`) declaring read-only paths (`trained_models/`, `cache/data/*/`, `experiments/`, `results/`, `mlflow/`); add per-directory `.agent-forbidden` markers where appropriate.
- `repo-governance`: Enable branch protection on `main` (required review, no force push, required checks); add `CODEOWNERS`; enable Dependabot + secret scanning + Copilot review on PRs.
- `stale-branch-triage`: Snapshot list of 50+ branches; tag `archive/<branch-name>` for anything not merged; open a tracking issue for delete-after-2-weeks.
- `experiments-arch-rfc`: One-page RFC choosing a single meaning for "experiment" (recommend: `configs/` declarative → `src/pipelines/` runners → `runs/` outputs) and mapping the four current experiment folders onto it. No code change yet — just the decision.

### P1 — Next 2–3 weeks (unblock new phase)
- `private-fork-setup`: Create private mirror/fork; document sync policy (upstream → fork nightly; fork → upstream via curated PRs only); move active branches into private.
- `model-adapter-protocol`: Design the ~150 LoC `ModelAdapter` protocol (fit/predict/save/load + config schema + metadata); ship as `src/models/adapter.py`; port 1 model to it as reference (recommend Chronos-2 or naive baseline first).
- `pydantic-config-schemas`: Introduce Pydantic v2 schemas for model configs, sweep configs, data configs, evaluation configs; validate on load; auto-generate JSON schema for docs; migrate all existing YAML in `configs/`.
- `run-manifest-schema`: Define `run_manifest.json` written by every training/eval run (code SHA, data version, seed, hyperparameters, budget metrics, environment fingerprint); wire into existing 3-tier storage.
- `optuna-integration`: Add `src/tuning/` implementation on top of Optuna + MLflow; support pruning, distributed workers, and re-use existing YAML search spaces via Pydantic; automatic search-space + budget table generation (LaTeX + Markdown).
- `mlflow-integration`: Wire MLflow tracking into every training/eval entrypoint; capture manifest, artifacts, metrics; document self-hosted server operations.
- `stats-rigor-module`: Promote `scripts/analysis/rebuttal_neurips2026/` A1–A9 into `src/evaluation/stats/` as first-class importable modules with unit tests + CI job that runs the fast ones on sample data on every PR.
- `experiments-collapse`: Execute the RFC from `experiments-arch-rfc`; unify `experiments/`, `src/experiments/`, `configs/experiments/`, `scripts/experiments/` into the chosen layout; keep symlinks or shim modules for backward compat for 1 release.

### P2 — Weeks 4–6 (public phase)
- `hf-benchmark-packaging`: Build the HF-hostable benchmark: `datasets` loader for public splits, evaluation script contract, leaderboard-ready output schema; publish a private draft first.
- `hf-space-leaderboard`: HF Space that reads leaderboard CSV from a datasets repo; simple table + CD diagram; instructions for benchmark submission.
- `license-migration`: Relicense code to Apache-2.0. Update `LICENSE`, `pyproject.toml` classifiers, per-file headers, README license section. Publish a `RELICENSE_NOTES.md` explaining the split: code Apache-2.0 / each dataset under its own upstream license / leaderboard entries CC-BY-4.0 / Gluroo data explicitly off-limits.
- `public-repo-slim`: Remove/relocate anything that makes the public repo look "in use": scratch scripts, temp files, in-flight notebook dumps, competition submission CSVs; verify against `.gitignore`.
- `docs-consolidation`: Single mkdocs site with four audiences (user, contributor, adapter-author, benchmark-submitter); retire `docs-internal/`; archive exploratory notebooks to a separate research-notebooks repo.
- `reproducibility-lock`: Introduce `uv` (or pip-tools) lockfile; document seed policy; add reproducibility CI job that installs from lockfile and runs smoke tests.
- `container-per-model`: Docker/Singularity image per model family + one CPU-baselines image; publish to a container registry; wire into HF Space + local dev.

### P3 — Later (nice to have)
- `launcher-abstraction`: Slurm/cloud/local launcher abstraction (Ray, Modal, or custom); Colab-friendly examples.
- `transformer-version-harmonize`: Track upstream `transformers` compatibility; collapse per-model envs when possible.
- `wandb-mirror`: Optional W&B mirror for polished reviewer/paper artifacts.

---

## 5. Execution notes

- Todos are tracked in the SQL `todos` table; deps in `todo_deps`. Update `status` (`pending` → `in_progress` → `done`/`blocked`) as work progresses.
- P0 todos are strict-order because of data-loss risk: `paper-tag-immutable` → `pii-secrets-audit` → `data-inventory` → `data-mirror-plan` → everything else.
- Any todo that touches history (secrets scrub, license migration) requires a signed-off migration doc before execution.
- Weekly review: reconcile plan against reality; move items between priorities as needed.

---

## 6. Assumptions to confirm during P0

- The `Blood-Glucose-Control` GitHub org owns the repo and has admin capacity to enable branch protection, secret scanning, and Dependabot.
- The team has access to at least one object-storage target (institutional S3, HF private, or backed-up cluster mount) for the 246 GB mirror.
- Apache-2.0 relicensing has legal signoff available (Gluroo commercial relationship is compatible with Apache-2.0 code + private data — nearly always the case, but confirm).
- GitHub fork under `Blood-Glucose-Control` (private) is acceptable as the team's day-to-day work location.
- Each open T1D dataset's license permits redistribution as a benchmark configuration (typically yes for research use; verify per-dataset before publishing the HF loader).
- Nothing already on public branches contains PHI (to be verified in `pii-secrets-audit`).
