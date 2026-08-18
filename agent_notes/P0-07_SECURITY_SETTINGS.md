# GitHub Security Settings Status

**Date**: 2026-08-06
**Repository**: Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast
**Verified By**: Repository Administrator

---

## ✅ Currently Enabled

The following GitHub security features are **ENABLED** as of 2026-08-06:

### Dependency Management
- ✅ **Dependency graph** - Visualizes Python dependencies
- ✅ **Dependabot alerts** - Notifies of vulnerable dependencies
- ✅ **Dependabot malware alerts** - Detects malicious packages
- ✅ **Dependabot security updates** - Auto-creates PRs to fix vulnerabilities

### Secret Protection
- ✅ **Secret scanning** - Detects committed secrets (API keys, tokens)
- ✅ **Push protection** - Blocks pushes containing secrets

---

## ⚠️ Status Unknown (Need Manual Check)

The following features may or may not be enabled (AI cannot check):

### Branch Protection
- ❓ **Branch protection rules on `main`** - Need to verify at:
  - https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/settings/branches
- Expected settings:
  - Require pull request reviews before merging
  - Require status checks to pass
  - Prohibit force pushes

### Code Scanning
- ❓ **CodeQL analysis** - Advanced security scanning (optional)
  - Free for public repos
  - Check at: https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/settings/security_analysis

---

## 📝 Recommendations

### Already Implemented ✅
All critical security features are enabled! Well done.

### Private fork governance constraint (2026-08-07)
- Team access was added to the private fork, but enforcing `main` branch protection
  there is currently blocked by GitHub plan limits (rulesets on private repos require
  paid plan features for this org).
- Decision: defer paid-plan change for now; continue with compensating controls and
  track this as a future governance task.
- Interim controls for private repo:
  1. Keep write/admin access limited to core maintainers only.
  2. Use PR-only workflow by team policy (even if not hard-enforced).
  3. Keep public cleanup work preserving all history in private.
- Future options to evaluate:
  - Stay on Free with strict collaborator permissions and manual PR discipline.
  - Move private work to a separate small paid org (only essential maintainers billed).
  - Use a personal private repo for active private work and mirror back as needed.

#### Decision checklist (`private-governance-cost-plan`)
- [ ] Confirm required private collaborators for next 3 months (names + permission level).
- [ ] Estimate monthly cost for each option:
  - [ ] Option A (Free/manual policy): $0 direct cost
  - [ ] Option B (small paid org): seats × monthly rate
  - [ ] Option C (personal private repo): seats × monthly rate (if applicable)
- [ ] Confirm minimum governance requirements:
  - [ ] PR-only flow required
  - [ ] Required approvals needed? (1 or 2)
  - [ ] Required CI checks needed? (yes/no)
- [ ] Choose one option (A/B/C) with owner approval and target decision date.
- [ ] Document final operating model:
  - [ ] Who has admin/write access
  - [ ] How upstream sync works
  - [ ] How private-to-public PR promotion works
- [ ] Implement chosen model and record completion in `project_tracking.csv`.

**Update 2026-08-13:** Decision completed in `private-governance-cost-plan` as **Option C (personal private repo workflow)**. See [`P1-13_PRIVATE_GOVERNANCE_COST_PLAN.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-13_PRIVATE_GOVERNANCE_COST_PLAN.md).

### Optional Enhancements
1. **Enable CodeQL scanning** (if not already enabled)
   - Finds security vulnerabilities in Python code
   - Free for public repositories
   - Settings → Code security → Set up code scanning

2. **Review branch protection rules**
   - Verify `main` branch is protected
   - Consider protecting other important branches (e.g., `neurips-rebuttal`)

---

## 🔗 Quick Links

- Security settings: https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/settings/security_analysis
- Branch protection: https://github.com/Blood-Glucose-Control/nocturnal-hypo-gly-prob-forecast/settings/branches
- CODEOWNERS: [/.github/CODEOWNERS](/.github/CODEOWNERS)
- Agent safety: [/AGENTS.md](/AGENTS.md)

---

**Last Updated**: 2026-08-06
**Next Review**: Check quarterly or after major changes
