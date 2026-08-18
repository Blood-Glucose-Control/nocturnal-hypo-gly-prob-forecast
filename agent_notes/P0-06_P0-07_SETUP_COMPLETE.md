# Agent Safety & Repository Governance - Setup Complete

**Date**: 2026-08-06
**Status**: ✅ COMPLETE

---

## ✅ Completed Tasks

### 1. Agent Safety Conventions
Created files that tell AI assistants which directories to never modify:

- ✅ **AGENTS.md** - Main convention file at repo root
- ✅ **CLAUDE.md** - Mirror of AGENTS.md for Claude-specific tools
- ✅ **.agent-forbidden markers** in 6 critical directories:
  - `trained_models/.agent-forbidden`
  - `cache/data/.agent-forbidden`
  - `experiments/.agent-forbidden`
  - `results/.agent-forbidden`
  - `mlflow/.agent-forbidden`
  - `.stash-backups/.agent-forbidden`

### 2. Repository Governance

#### CODEOWNERS File
- ✅ Created `.github/CODEOWNERS`
- ✅ Automatically requests @cjrisi for review on all PRs
- ✅ Specific rules for ML code, configs, documentation, and protected directories

#### GitHub Security Settings (Verified Enabled)
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot malware alerts
- ✅ Dependabot security updates
- ✅ Secret scanning
- ✅ Push protection

### 3. Documentation Organization

#### agent_notes/ Directory
Created and populated with documentation files:
- ✅ `README.md` - Directory purpose and guidelines
- ✅ `BACKUP_RECOVERY.md` - 311 GB backup restore instructions
- ✅ `PII_SECRETS_AUDIT_REPORT.md` - Security audit results
- ✅ `SECURITY_SETTINGS.md` - GitHub settings status
- ✅ `branch_cleanup_plan.md` - Branch archival strategy
- ✅ `reorg_plan.md` - Overall cleanup roadmap

---

## 📝 Key Protections in Place

### Protected Directories (AI Read-Only)
These directories are now marked as read-only for AI assistants:

1. **trained_models/** - 246 GB of trained models (paper-critical)
2. **cache/data/** - Downloaded and preprocessed datasets
3. **experiments/** - Experimental results and logs
4. **results/** - Analysis outputs
5. **mlflow/** - MLflow tracking database
6. **agent_notes/** - Human-readable documentation

### Protected Branches
Based on your confirmation, `main` branch has protection enabled requiring:
- Pull request reviews before merging
- No direct pushes allowed

### Code Review Automation
CODEOWNERS file ensures @cjrisi is automatically requested for review on:
- All PRs (default rule)
- ML code changes (`/src/models/`, `/src/evaluation/`, etc.)
- Configuration changes (`/configs/`)
- Documentation changes (`/docs/`, `/README.md`)
- CI/CD changes (`/.github/workflows/`)
- Dependency changes (`pyproject.toml`, `requirements*.txt`)
- Protected directory changes (extra scrutiny)

---

## 🔧 About the gh CLI Issue

### What Happened
When trying `gh auth login`, you got an error. This is because you have a **different tool** installed called `gh` (GitHub browser opener), not the official GitHub CLI.

### The Tool You Have
```bash
$ gh --help
Github browser opener
```
This tool opens GitHub pages in your browser, but can't interact with the GitHub API.

### The Tool We Need (Optional)
The official GitHub CLI is called `gh` too, but it's different:
- Repository: https://cli.github.com/
- Purpose: Interact with GitHub API from terminal
- Install: https://github.com/cli/cli#installation

### Do You Need It?
**No!** We've completed all the tasks. The official `gh` CLI would have let me check branch protection status automatically, but you confirmed it manually, so we're all set.

If you want it later:
```bash
# Remove the conflicting gh (browser opener)
conda remove gh

# Install official GitHub CLI
# Follow: https://github.com/cli/cli/blob/trunk/docs/install_linux.md
```

---

## 🎯 What's Next?

With agent safety and governance complete, the next P0 tasks are:

1. **data-inventory** - Create CSV inventory of 246 GB trained models
2. **data-mirror-plan** - Choose object storage and write mirror script
3. **land-rebuttal-analyses** - Cherry-pick rebuttal stats to main
4. **stale-branch-triage** - Archive/delete old branches
5. **experiments-arch-rfc** - Design single experiment architecture

---

## 📊 Progress Summary

### P0 Tasks (Safety + Governance)
- ✅ paper-tag-immutable - Tag verified
- ✅ pii-secrets-audit - No secrets/PHI found
- ✅ agent-safety-conventions - AGENTS.md, CLAUDE.md, markers created
- ✅ repo-governance - CODEOWNERS, security settings verified
- ⏳ data-inventory - Ready to start
- ⏳ data-mirror-plan - Blocked by data-inventory
- ⏳ land-rebuttal-analyses - Ready to start
- ⏳ stale-branch-triage - Ready to start
- ⏳ experiments-arch-rfc - Ready to start

### Files Created
- `AGENTS.md` - 3.1 KB
- `CLAUDE.md` - 3.1 KB (mirror)
- `.github/CODEOWNERS` - 1.2 KB
- `agent_notes/README.md` - 1.5 KB
- `agent_notes/SECURITY_SETTINGS.md` - 2.3 KB
- 6x `.agent-forbidden` marker files

---

**Status**: ✅ Agent safety and governance setup COMPLETE
**Next**: Data inventory or branch cleanup
