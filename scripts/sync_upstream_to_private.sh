#!/usr/bin/env bash
set -euo pipefail

# Sync public upstream main -> private main with divergence guardrails.
#
# Usage:
#   ./scripts/sync_upstream_to_private.sh
#
# Expected remotes:
#   origin  -> public repo
#   private -> private repo

PUBLIC_REMOTE="${PUBLIC_REMOTE:-origin}"
PRIVATE_REMOTE="${PRIVATE_REMOTE:-private}"
PUBLIC_BRANCH="${PUBLIC_BRANCH:-main}"
PRIVATE_BRANCH="${PRIVATE_BRANCH:-main}"

echo "Fetching remotes..."
git fetch "$PUBLIC_REMOTE" --prune
git fetch "$PRIVATE_REMOTE" --prune

public_ref="refs/remotes/${PUBLIC_REMOTE}/${PUBLIC_BRANCH}"
private_ref="refs/remotes/${PRIVATE_REMOTE}/${PRIVATE_BRANCH}"

if ! git show-ref --verify --quiet "$public_ref"; then
  echo "ERROR: Missing $public_ref" >&2
  exit 1
fi
if ! git show-ref --verify --quiet "$private_ref"; then
  echo "ERROR: Missing $private_ref" >&2
  exit 1
fi

public_sha="$(git rev-parse "$public_ref")"
private_sha="$(git rev-parse "$private_ref")"

echo "Public  ${PUBLIC_REMOTE}/${PUBLIC_BRANCH}: ${public_sha}"
echo "Private ${PRIVATE_REMOTE}/${PRIVATE_BRANCH}: ${private_sha}"

if git merge-base --is-ancestor "$private_sha" "$public_sha"; then
  echo "Fast-forward sync is safe. Pushing ${PUBLIC_REMOTE}/${PUBLIC_BRANCH} -> ${PRIVATE_REMOTE}/${PRIVATE_BRANCH} ..."
  git push "$PRIVATE_REMOTE" "${PUBLIC_REMOTE}/${PUBLIC_BRANCH}:${PRIVATE_BRANCH}"
  echo "Sync complete."
  exit 0
fi

if git merge-base --is-ancestor "$public_sha" "$private_sha"; then
  echo "No sync needed: private/main is ahead of public/main."
  echo "Use curated private->public PR flow for promotion."
  exit 0
fi

echo "ERROR: public/main and private/main have diverged."
echo "Resolve manually with explicit merge/rebase strategy before syncing."
exit 1
