# Cloud Backup Options - Decision Guide

**Context**: User stuck on Backblaze reCAPTCHA during signup (2026-08-06)

---

## Quick Decision Matrix

| Option | Cost/Month | Setup Time | Complexity | Recommendation |
|--------|-----------|------------|------------|----------------|
| **Backblaze B2** | $0.01 (2.27GB)<br>$1.50 (300GB) | 15 min | Easy | ⭐ Best if signup works |
| **Wasabi** | $5.99 minimum | 20 min | Easy | ⭐ Try if B2 fails |
| **AWS S3** | $0.52 (2.27GB)<br>$6.90 (300GB) | 15 min | Medium | Use if org pays |
| **Defer** | $0 | 0 min | N/A | ✅ **Recommended for now** |

---

## Recommendation: Defer and Move Forward

### Why Defer Makes Sense

**You already have excellent protection**:
1. ✅ **311 GB local backup** - Complete repository snapshot on `/data/home/cjrisi/backups/`
2. ✅ **Private GitHub repo** - All branches backed up remotely
3. ✅ **ZFS-locked training data** - Immutable, cannot be accidentally deleted
4. ✅ **Structured inventories** - Know exactly what's paper-critical (2.27 GB)

**Cloud backup is nice-to-have, not blocking**:
- Paper is already accepted at NeurIPS 2026
- Rebuttal analyses are version-controlled in git
- Trained models can be regenerated if needed (code + data preserved)
- MLflow will eventually sync to cloud, but works fine locally first

**Better use of time**:
- 2 more P0 tasks remain: land-rebuttal-analyses, stale-branch-triage
- P1 MLflow setup doesn't require cloud storage initially
- Can set up cloud backup later when signup issues resolve

---

## If You Want to Push Through

### Option 1: Troubleshoot Backblaze Signup

**Common reCAPTCHA issues**:
- VPN/proxy blocking: Try disabling
- Browser fingerprinting: Try different browser or incognito
- Server/institutional IP flagged: Try from personal device/network
- Cookie issues: Clear cookies and try again

**Steps to try**:
1. Different browser (Chrome → Firefox → Safari)
2. Different device (desktop → phone → tablet)
3. Different network (work → home → mobile hotspot)
4. Incognito/private browsing mode
5. Wait 24 hours and try again (sometimes rate-limited)

### Option 2: Switch to Wasabi

**Wasabi advantages**:
- Usually easier signup (fewer anti-bot measures)
- No egress fees (B2 charges for downloads after free tier)
- S3-compatible (same tools, easy migration)

**Wasabi setup**:
```bash
# 1. Sign up at https://wasabi.com/sign-up/
# 2. Create bucket: mlflow-nocturnal-hypo
# 3. Create access key (Account → Access Keys)
# 4. Install AWS CLI
pip install awscli

# 5. Configure with Wasabi credentials
aws configure
# AWS Access Key ID: <your-wasabi-access-key>
# AWS Secret Access Key: <your-wasabi-secret-key>
# Default region: us-east-1
# Default output format: json

# 6. Test connection
aws s3 ls --endpoint-url=https://s3.us-east-1.wasabisys.com

# 7. Modify upload script to use aws s3 sync instead of b2 sync
```

### Option 3: Use AWS S3 (If You Have Credits)

**When AWS makes sense**:
- You have AWS credits from institution
- You're already using AWS for other research
- You want enterprise SLA guarantees

**AWS cost** (more expensive but more features):
- $0.023/GB/month for S3 Standard
- 2.27 GB paper-critical = $0.52/month
- 300 GB full mirror = $6.90/month

---

## My Recommendation

**Defer the cloud backup for now**. Here's why:

1. **You're 62.5% done with P0 tasks** - maintain momentum
2. **Cloud backup is the most optional P0 task** - everything else is higher priority
3. **Your local + git backups are solid** - cloud is redundancy, not primary
4. **Signup issues are outside your control** - don't let this block progress

**Next steps**:
1. Mark P0-4 as "blocked" in tracking
2. Move to P0-5: land-rebuttal-analyses (high value, clear path)
3. Then P0-8: stale-branch-triage (cleanup momentum)
4. Come back to P0-4 in a few days when:
   - Backblaze signup works (reCAPTCHA issue resolved)
   - You have time to try Wasabi alternative
   - You're ready to explore institutional S3/storage options

---

## To Proceed with Deferral

Run these commands:

```bash
# Update project tracking to mark data-mirror-plan as blocked
sed -i 's/^data-mirror-plan,P0,pending,/data-mirror-plan,P0,blocked,/' project_tracking.csv

# Note the blocker
echo "Note: Blocked on cloud provider signup (reCAPTCHA issues)" >> project_tracking.csv
```

Then proceed with:
- **P0-5**: land-rebuttal-analyses (cherry-pick to main)
- **P0-8**: stale-branch-triage (archive old branches)

---

## Summary

| Choice | Action |
|--------|--------|
| **Defer (Recommended)** | Mark blocked, do P0-5 and P0-8, return later |
| **Try Wasabi** | https://wasabi.com/sign-up/ + modify upload script |
| **Troubleshoot B2** | Different browser/device/network |
| **Use AWS S3** | If you have credits or institutional account |

**What would you like to do?**
