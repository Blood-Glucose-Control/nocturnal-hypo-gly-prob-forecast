# P1-13 Private Governance Cost Plan

**Date:** 2026-08-13
**Task ID:** `private-governance-cost-plan`
**Status:** Decision recorded

---

## Decision

Adopt **Option C (personal private repo workflow)** for the next operating window through end of September review period.

Rationale:

- avoids organization-seat and premium governance costs that do not match your student-rotation model,
- preserves broad public repo access for continuity across dozens of mentees,
- limits private write access to the 4-5 actively building each semester,
- remains compatible with current branch-cleanup and archive-first operational discipline.

Decision checkpoint:

- Re-evaluate after late-September review period, or earlier if collaborator count or compliance requirements increase.

---

## Cost/operations comparison

| Option | Direct Cost | Governance Strength | Operational Burden | Decision |
|---|---:|---|---|---|
| A. Free + manual policy (org private repo) | $0 direct, but limited enforceable controls | Medium (policy-enforced) | Medium | Defer |
| B. Small paid org | seats × monthly plan | High (enforced rulesets/protection) | Low-Medium | Not affordable now |
| C. Personal private repo workflow | low / bounded by active collaborators only | Medium-High (small trusted maintainer set) | Medium | **Selected now** |

---

## Minimum governance controls (Option C)

1. Keep personal private repo admin/write access limited to current active maintainers.
2. Treat students not actively shipping private work as public-repo-only contributors.
3. PR-only merge policy for private `main` (human-enforced).
4. Require one reviewer sign-off by team policy.
5. Use archive-first protocol before any destructive branch deletion.
6. Keep public promotion curated (private → public via explicit PR/cherry-pick only).
7. Nightly upstream sync via [sync_upstream_to_private.sh](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/sync_upstream_to_private.sh).

---

## Trigger to revisit governance model

Move from Option C to another model (A or B) if any of these become true:

- active private-maintainer set grows enough that personal-repo admin overhead becomes brittle,
- mandatory branch/ruleset enforcement is required by policy/compliance,
- repeated policy drift incidents (unreviewed merges, unsafe branch operations).

---

## Implementation checklist

- [x] Governance option selected and documented.
- [x] Sync policy documented and automated.
- [x] Review-period re-evaluation trigger defined.
- [x] Public/private promotion constraints documented.
