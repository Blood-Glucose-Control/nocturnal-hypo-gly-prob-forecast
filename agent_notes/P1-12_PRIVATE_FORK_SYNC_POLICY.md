# P1-12 Private Fork Sync Policy

**Date:** 2026-08-13
**Task ID:** `private-fork-setup`
**Status:** Implemented

---

## Objective

Define and operationalize the upstream/public to private sync model while keeping private-to-public promotion curated.

---

## Current governance choice (linked to cost plan)

Per [`P1-13_PRIVATE_GOVERNANCE_COST_PLAN.md`](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/agent_notes/P1-13_PRIVATE_GOVERNANCE_COST_PLAN.md), the current operating choice is:

- **Option C: personal private repo workflow** for active private development,
- while preserving broad public-repo access for the larger rotating student group.

In this policy, the remote name `private` is a logical alias and can point to either:
- a personal private repo (current choice), or
- an organization private repo (if governance model changes later).

---

## Plain-language explanation (why this design exists)

If you're not deep in Git/release engineering, here's the core idea:

- We have a **public repo** (`origin`) that should stay clean and stable.
- We have a **private repo** (`private`) where active team work can move faster.
- We still want them connected so important public updates flow into private work.

### What "promotion direction" means

Think of "promotion" as "what gets allowed to move into the public release lane."

- **Normal sync direction:** public `main` → private `main`
  - private always gets the latest public baseline
- **Promotion direction:** private → public only through reviewed PRs/cherry-picks
  - we do **not** auto-push private work into public

Why: this prevents accidental publication of internal-only experiments, local-path scripts, or noisy artifacts.

### What "fast-forward only" means

Fast-forward means Git can move a branch pointer straight ahead without creating a merge commit.

In practical terms:

- If private `main` is behind public `main`, we can safely move private forward.
- If histories diverged (both sides have different new commits), auto-sync stops and asks for manual review.

Why: this avoids hidden merges that can mix histories in surprising ways and makes repo state easier to reason about.

### Why this policy is safer for this project

1. Reduces risk of leaking private/in-progress work into the public repo.
2. Keeps release history cleaner and easier to audit.
3. Forces explicit human review when histories diverge.
4. Supports your current workflow: private iteration first, curated public release later.

### Flow diagram (at a glance)

```text
                    Nightly sync (fast-forward only)
   Public repo                                        Private repo
  origin/main  ----------------------------------->  private/main
                                                       |
                                                       | Team experiments / iteration
                                                       v
                                              private feature branches
                                                       |
                                                       | Curated promotion only
                                                       | (reviewed PR or cherry-pick)
                                                       v
                                                   origin/main

Rule: never auto-push private branches directly into public main.
```

---

## Remotes and branch roles

- Public upstream remote: `origin` (`main` is public release branch)
- Private development remote: `private` (`main` is private integration branch)

Policy:

1. **Nightly direction:** `origin/main` → `private/main` (fast-forward only).
2. **Promotion direction:** private work → public only via reviewed PRs/cherry-picks.
3. **No force pushes** in either direction as part of sync automation.

---

## Automation

Script added:

- [sync_upstream_to_private.sh](/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/scripts/sync_upstream_to_private.sh)

Behavior:

- fetches both remotes
- verifies refs exist
- pushes `origin/main` to `private/main` only when fast-forward-safe
- exits with error on divergence to force explicit human resolution

Example cron (local runner with repo checkout):

```bash
0 3 * * * cd /data/home/cjrisi/nocturnal-hypo-gly-prob-forecast && ./scripts/sync_upstream_to_private.sh >> /tmp/private-sync.log 2>&1
```

---

## Manual runbook

```bash
cd /data/home/cjrisi/nocturnal-hypo-gly-prob-forecast
./scripts/sync_upstream_to_private.sh
```

If divergence is reported:

1. pause sync,
2. review commit graph (`git log --left-right --graph origin/main...private/main`),
3. resolve with explicit merge/rebase decision,
4. rerun sync script.

---

## Acceptance criteria met

- [x] Sync direction documented.
- [x] Automation script present and executable.
- [x] Divergence guardrails documented.
- [x] Private→public remains curated PR-only.
