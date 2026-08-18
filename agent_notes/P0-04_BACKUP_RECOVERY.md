# Full Local Backup - Pre-Cleanup Safety Net
**Date**: 2026-08-05 19:13 UTC
**Status**: ✅ COMPLETE AND VERIFIED

## Backup Location
`/data/home/cjrisi/backups/nocturnal-hypo-gly-backup-2026-08-05/`

## What's Backed Up
- ✅ 973,973 files (311 GB total)
- ✅ 246 GB trained models
- ✅ 1.8 GB experiments
- ✅ 2.8 GB cache
- ✅ All git history (.git/)
- ✅ All stash patches (.stash-backups/)

## Quick Restore

### Restore entire directory (e.g., trained_models/)
```bash
rsync -av /data/home/cjrisi/backups/nocturnal-hypo-gly-backup-2026-08-05/trained_models/ \
          /data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/trained_models/
```

### Restore single file
```bash
cp /data/home/cjrisi/backups/nocturnal-hypo-gly-backup-2026-08-05/path/to/file \
   /data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/path/to/file
```

### Full repository restore
```bash
cd /data/home/cjrisi
cp -a backups/nocturnal-hypo-gly-backup-2026-08-05 nocturnal-hypo-gly-prob-forecast-restored
```

## SAFE Git Commands (Don't Touch Local Files)
- `git push origin --delete branch-name` ✅
- `git tag archive/name branch` ✅
- `git push origin tag-name` ✅

## DANGEROUS Commands (NEVER USE)
- `git clean -fdx` ❌ Deletes all .gitignored files!
- `git reset --hard` ⚠️ Can lose uncommitted work

**Backup verified. Safe to proceed with branch cleanup.**
