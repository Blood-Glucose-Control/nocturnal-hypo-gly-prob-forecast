# Branch Inventory for Backup (2026-08-05)

## Local-Only Branches Requiring Backup

### 1. `neurips-rebuttal` ⚠️ CRITICAL
- **Status**: Has 2 unpushed commits not on any remote
- **Purpose**: NeurIPS 2026 paper submission + rebuttal analysis pipeline
- **Value**: CRITICAL - Contains publication-critical statistical rigor pipeline
- **Commits not on remote**:
  - 27980c6 Planning docs updates
  - 261f179 Storing so I can restart session in another window
- **Notable files**:
  - `reorg_plan.md` (106 lines - planning document)
  - `.github/prompts/plan-neuripsRebuttal.prompt.md`
  - `scripts/analysis/rebuttal_neurips2026/` (statistical analysis)
  - Data config registry backups
- **Action**: MUST be backed up before any cleanup. Tag as `paper-v1` after backup.
- **Secrets scan**: No obvious secrets in file names, but needs full `git log -p` scan per plan

### 2. `metabonet-integration`
- **Status**: Has 1 unpushed commit beyond origin/feat/autogluon-baselines
- **Purpose**: MetaboNet 2026 dataloader with dual split modes
- **Value**: ACTIVE WORK - Competition preparation
- **Commit**: 5420d92 feat: Add MetaboNet 2026 dataloader with dual split modes
- **Notable changes**:
  - Modified: `src/data/dataset_configs.py`, `src/data/diabetes_datasets/data_loader.py`
  - Deleted: MetaboNet 2026 loader files (appears to be cleanup/refactor)
- **Action**: Backup to private fork - contains competition-related work
- **Secrets scan**: No obvious secrets in file names

### 3. `anonneurips26`
- **Status**: Has 1 unpushed commit beyond origin/anonneurips26
- **Purpose**: Branch cleanup + initial public release preparation
- **Value**: ACTIVE WORK - Cleanup preparation
- **Commits not on remote**: 0198a9b Branch cleanup
- **Notable changes**: Large diff including:
  - Modified Makefile, README.md, pyproject.toml
  - Added `scripts/analysis/summarize_experiments.py`
  - Removed transformer modules (`src/data/transformers/`)
  - Empty __init__.py files (cleanup)
- **Action**: Backup to private fork - contains cleanup work
- **Secrets scan**: No obvious secrets in file names

## Remote Branches Already Backed Up
- Total: 74 branches on `origin` (already pushed to public GitHub)
- These are safe and don't need immediate action beyond the cleanup plan

## Recommended Backup Strategy

### Step 1: Create Private Fork
- Create private fork under `Blood-Glucose-Control` org
- Name suggestion: `nocturnal-hypo-gly-prob-forecast-private`
- Purpose: Team workspace for competition work, model development

### Step 2: Add Private Remote
```bash
git remote add private <private-fork-url>
```

### Step 3: Backup Local Branches
```bash
# Push all 3 local-only branches to private fork
git push private neurips-rebuttal:neurips-rebuttal
git push private metabonet-integration:metabonet-integration
git push private anonneurips26:anonneurips26

# Verify all commits are backed up
git fetch private
git log --oneline neurips-rebuttal ^private/neurips-rebuttal  # Should be empty
git log --oneline metabonet-integration ^private/metabonet-integration  # Should be empty
git log --oneline anonneurips26 ^private/anonneurips26  # Should be empty
```

### Step 4: Tag neurips-rebuttal
```bash
git tag -a paper-v1 neurips-rebuttal -m "NeurIPS 2026 submission version with rebuttal analysis"
git push origin paper-v1
git push private paper-v1
```

### Step 5: Verify SHA Preservation
Document the commit SHAs externally (cluster README, this file) as immutable reference:
- neurips-rebuttal HEAD: 27980c6
- paper-v1 tag: (will be 27980c6 after tagging)

## Next Steps After Backup
1. Full PII/secrets audit: `git log --all -p` scan across entire history
2. Branch cleanup plan: Review 74 remote branches, create archive/* tags, delete stale
3. Move active work to private fork
4. Configure GitHub branch protection on main

## Notes
- No obvious PHI/secrets detected in quick file name scan
- Full `git log -p` audit still required per reorg plan
- All 3 local branches contain valuable work - none should be deleted without backup
- Priority order matches reorg plan P0: tag → audit → inventory → backup → cleanup
