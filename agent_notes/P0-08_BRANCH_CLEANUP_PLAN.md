# P0-08 Branch Triage Completion Record

**Date:** 2026-08-08
**Status:** ✅ Complete (public stale-branch cleanup complete; hold set intentionally parked)

Tracking issue: https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/issues/422

---

## Final Snapshot

- Public remote (`origin`) active branches:
  - `main`
  - **1** non-main branch (`anonneurips26`)
- Private remote (`private`) retains historical work and all archived/deleted public branches.
- Archive tags for all deleted branches exist on both remotes and were SHA-verified before deletion.

---

## Completed Cleanup Waves

### Wave A (completed)
- `copilot/sub-pr-236`
- `chronos_finetune`
- `ttm-kaggle-submit`
- `ay-exploration`
- `nocturnal_eval_metrics`
- `pinball-loss`

### Wave B delete-ready (completed)
- `chronos-adapt`
- `tide_impl`
- `tide_testing`
- `ss-chronos-2`
- `ss-base-class-redesign`

### Wave B review-first (completed after owner sign-off)
- `toto-finetuning`
- `ss-tide-validation`
- `fix/chronos2-cross-learning`
- `two-stage-ft`

---

## Remaining non-main public branches (intentionally kept)

These are not stale-cleanup targets right now:

1. `feat/autogluon-baselines`
2. `anonneurips26`

---

## Focused audit: `feat/autogluon-baselines` (2026-08-08)

### Snapshot

- Divergence vs `origin/main`: **11 commits ahead, 3 commits behind**
- Scope: **179 files changed** (`+18,278 / -253,879`)
- Change mix: model-family additions, experiment summarization tooling, large generated result artifacts, and rerun scripts with machine-local paths.

### Confirmed high-value public-safe content

These match the branch owner's intent and are good extraction candidates:

1. **AutoGluon baseline model integration**
   - `src/models/autogluon_base.py`
   - `src/models/{naive_baseline,statistical,deepar,patchtst,tft}/`
   - `src/models/factory.py` registrations + loading paths
   - `tests/models/test_{autogluon_base,naive_baseline,statistical,deepar,patchtst,tft}.py`
   - `configs/models/{naive_baseline,statistical,deepar,patchtst,tft}/`

2. **Nocturnal summary utilities (candidate, but separate PR)**
   - `src/experiments/nocturnal/grand_summary.py`
   - `src/experiments/nocturnal/grand_summary_split.py`
   - `src/experiments/nocturnal/episode_classification.py`
   - `src/experiments/nocturnal/holdout_split_analysis.py`
   - `scripts/analysis/build_grand_summary.py` (+ companion summary scripts)

### Content that should **not** be merged directly to public main

1. **Generated artifacts / publication outputs**
   - `results/grand_summary/*`
   - edited `experiments/**/summary.csv` and `best_*.csv`
   - figure exports (`.pdf`, `.png`)

2. **Environment-coupled rerun scripts and manifests**
   - `scripts/experiments/rerun_best_models.sh`
   - `scripts/experiments/rerun_10_failed.sh`
   - `scripts/experiments/rerun_manifest.txt`
   - `scripts/experiments/rerun_failed.txt`
   - These include hardcoded local paths (e.g., `/data/home/...`) and dated run-specific checkpoint references.

3. **Accidental destructive config deletions**
   - deletions under `configs/data/holdout*/*.yaml`
   - these should be excluded from any merge.

### Extraction plan (recommended)

1. **PR-A (highest value): AutoGluon baseline core only**
   - cherry-pick/extract model code + tests + model configs.
   - explicitly exclude results/experiments CSV churn and data-config deletions.
2. **PR-B: Nocturnal summary/aggregation modules**
   - land reusable summary code only after API review.
3. **PR-C (optional): sweep orchestration scripts**
   - only scripts that are portable; strip local absolute paths first.
4. After PR-A/PR-B/PR-C decisions are complete:
   - archive-tag `feat/autogluon-baselines` on both remotes,
   - delete public branch.

### Decision status

- `feat/autogluon-baselines` remains an intentional hold branch **only as an extraction source**.
- Do **not** merge branch wholesale.

---

## Hold-Branch Policy (next review trigger)

Park these 2 branches for now and continue with other P0 work.

Revisit these branches when any of the following occurs:
- branch is inactive for >120 days,
- branch owner confirms it is no longer needed on public.

When revisiting, apply the same safety flow:
1. create/push `archive/*` tag on both remotes,
2. verify SHA on both remotes,
3. delete from `origin`,
4. re-snapshot.

---

## Outcome

P0-08 objective achieved:
- stale public branches were triaged and cleaned up,
- recovery guarantees are preserved (archive tags + private retention),
- remaining public branches are active or tied to open PRs and intentionally parked.

---

## 2026-08-13 closeout audit: `autogluon-baselines-extraction`

### Completion verdict

- **PR-A core extraction is complete on `main`** (model families/configs/tests/factory wiring present).
- Targeted validation rerun passed on `main`: **70 passed** (`test_autogluon_base`, model-family tests, `test_registry`).
- Remaining `feat/autogluon-baselines` deltas are intentionally excluded from public merge:
  - generated artifacts and experiment CSV churn,
  - environment-coupled rerun scripts with absolute local paths,
  - destructive holdout config deletions,
  - a broad-exception regression in `src/models/autogluon_base.py` (`except Exception`) vs bounded exceptions on `main`.

### Exact closeout commands (archive-first, then delete public branch)

```bash
# 0) sync refs
git fetch --all --prune

# 1) create archive tags from current public branch tips
git tag -a archive/feat-autogluon-baselines-20260813 origin/feat/autogluon-baselines -m "Archive feat/autogluon-baselines before public branch deletion"
git tag -a archive/p0-autogluon-pr-a-20260813 origin/p0-autogluon-pr-a -m "Archive p0-autogluon-pr-a before public branch deletion"

# 2) push tags to both remotes (private retains historical recovery path)
git push origin archive/feat-autogluon-baselines-20260813 archive/p0-autogluon-pr-a-20260813
git push private archive/feat-autogluon-baselines-20260813 archive/p0-autogluon-pr-a-20260813

# 3) verify SHA parity for archive tags
git rev-parse archive/feat-autogluon-baselines-20260813
git ls-remote --tags origin archive/feat-autogluon-baselines-20260813
git ls-remote --tags private archive/feat-autogluon-baselines-20260813

git rev-parse archive/p0-autogluon-pr-a-20260813
git ls-remote --tags origin archive/p0-autogluon-pr-a-20260813
git ls-remote --tags private archive/p0-autogluon-pr-a-20260813

# 4) delete public branches after verification
git push origin --delete feat/autogluon-baselines
git push origin --delete p0-autogluon-pr-a

# 5) optional local cleanup
git branch -d feat/autogluon-baselines || true
git branch -d p0-autogluon-pr-a || true
git fetch origin --prune
```

### Executed closeout result (2026-08-13)

- Created archive tag `archive/feat-autogluon-baselines-20260813` from `origin/feat/autogluon-baselines`.
- Pushed the tag to both `origin` and `private`.
- Verified commit SHA parity on both remotes:
  - tag target commit: `baeee053fc9ea15ce45332e1f97341e0fc440f01`
- Deleted public branch `origin/feat/autogluon-baselines`.
- Deleted local branch `feat/autogluon-baselines`.
- Verified `private/feat/autogluon-baselines` remains preserved.
