# Data Mirror Plan - Implementation Guide

**Task**: P0-4 data-mirror-plan
**Date**: 2026-08-06
**Status**: Ready for implementation

---

## Phase 1: Cloud Storage Setup (15 minutes)

### Option A: Backblaze B2 (Recommended for small datasets)

**Having reCAPTCHA issues?** Try Option B (Wasabi) or Option C (Defer) below.

#### Backblaze B2 Setup

### Step 1: Create Backblaze Account (5 min)

1. Go to: https://www.backblaze.com/b2/sign-up.html
2. Fill out registration form
3. Verify email address
4. Login to dashboard

**Pricing to expect**:
- First 10 GB: FREE
- After that: $0.005/GB/month ($1.50 for 300 GB)
- Our immediate need: **2.27 GB paper-critical = ~$0.01/month** (basically free!)

### Step 2: Create a Bucket (2 min)

1. In B2 dashboard, click **"Buckets"** in left menu
2. Click **"Create a Bucket"**
3. Settings:
   - **Bucket Unique Name**: `mlflow-nocturnal-hypo` (or adjust if taken - must be globally unique)
   - **Files in Bucket**: **Private** (NOT public!)
   - **Default Encryption**: **Enabled** (recommended for PHI-adjacent research data)
   - **Object Lock**: **Disabled** (not needed)
4. Click **"Create a Bucket"**

### Step 3: Create Application Key (3 min)

1. Go to **"Account"** → **"Application Keys"**
2. Click **"Add a New Application Key"**
3. Settings:
   - **Name**: `mlflow-nocturnal-hypo-cli`
   - **Allow access to bucket(s)**: Select your bucket (`mlflow-nocturnal-hypo`)
   - **Type of Access**: **Read and Write**
   - **Allow List All Bucket Names**: **Yes**
   - **File name prefix**: Leave empty (full bucket access)
   - **Duration**: Leave blank (no expiration, but plan to rotate annually)
4. Click **"Create New Key"**
5. **IMPORTANT**: Copy both values NOW (you won't see them again!):
   - **keyID**: (starts with something like `005abc123def...`)
   - **applicationKey**: (long random string)

---

### Option B: Wasabi (Alternative - Easier Signup)

If Backblaze reCAPTCHA is blocking you, try Wasabi instead:

**Why Wasabi**:
- Often easier signup process (no reCAPTCHA issues reported)
- S3-compatible (same tools work)
- No egress fees (vs B2's free tier)
- Cost: $5.99/month minimum (first 1TB) - slightly more expensive but still cheap

**Wasabi Setup**:
1. Go to: https://wasabi.com/sign-up/
2. Create account (usually smoother signup process)
3. Create bucket: `mlflow-nocturnal-hypo` (us-east-1 or us-west-1 region)
4. Create access key: Account → Access Keys → Create New Access Key
5. Save credentials same way as B2 (see Step 4 below)
6. Install AWS CLI instead: `pip install awscli`
7. Configure: `aws configure` (enter Wasabi keys, region)
8. Upload command changes from `b2 sync` to `aws s3 sync` (same syntax otherwise)

**Upload script changes for Wasabi**:
- Replace `b2 sync` commands with `aws s3 sync`
- Use endpoint: `--endpoint-url=https://s3.us-east-1.wasabisys.com`
- Bucket URL: `s3://mlflow-nocturnal-hypo/`

---

### Option C: Defer Cloud Backup (Focus on Other P0 Tasks)

**If both providers are giving signup issues**, you can defer this task:

**You already have excellent backups**:
- ✅ 311 GB local backup on `/data/home/cjrisi/backups/`
- ✅ Private GitHub repo with all branches
- ✅ ZFS-locked data (immutable training data)

**Defer strategy**:
1. Complete other P0 tasks first (land-rebuttal-analyses, stale-branch-triage)
2. Set up MLflow with local PostgreSQL only (P1 task)
3. Return to cloud backup once provider signup issues resolved
4. Cloud backup is nice-to-have, not blocking for paper publication

**To defer**: Mark P0-4 as "blocked" and move on to P0-5 (land-rebuttal-analyses)

---

### Step 4: Save Credentials Securely (2 min)

**For B2 or Wasabi** - DO NOT commit these to git!

**DO NOT commit these to git!**

Save to a secure location on the server:

```bash
# Create secure credentials file (readable only by you)
mkdir -p ~/.b2_credentials
chmod 700 ~/.b2_credentials

cat > ~/.b2_credentials/mlflow-nocturnal-hypo << 'EOF'
# Backblaze B2 credentials for mlflow-nocturnal-hypo bucket
# Created: 2026-08-06
# Rotate: 2027-08-06 (annual rotation)

B2_KEY_ID="your_keyID_here"
B2_APPLICATION_KEY="your_applicationKey_here"
B2_BUCKET_NAME="mlflow-nocturnal-hypo"
EOF

chmod 600 ~/.b2_credentials/mlflow-nocturnal-hypo
```

### Step 5: Install B2 CLI (3 min)

```bash
# Activate your Python environment
conda activate env-in-conda  # or whichever env you use

# Install B2 SDK with CLI
pip install b2[full] # note its b2sdk NOT b2-sdk

# Verify installation
b2 version
```

### Step 6: Authorize B2 CLI (1 min)

```bash
# Load credentials
source ~/.b2_credentials/mlflow-nocturnal-hypo

# Authorize (this stores auth in ~/.config/b2/account_info)
b2 account authorize "$B2_KEY_ID" "$B2_APPLICATION_KEY"

# Test - verify bucket access (if using a restricted key, you'll see a notice that it's restricted to your bucket - this is expected and secure!)
b2 bucket get mlflow-nocturnal-hypo

# List files in your bucket
b2 ls b2://mlflow-nocturnal-hypo/
```

✅ If you see your bucket info and can list files, you're authorized correctly!

**Note:** If your application key is restricted to a specific bucket (recommended for security), you may see a message like "Application key is restricted to buckets: ['mlflow-nocturnal-hypo']" - this is normal and means your key is properly scoped.

---

## Phase 2: Upload Paper-Critical Artifacts (30-60 minutes)

### Step 1: Dry Run (Test Without Uploading)

```bash
cd /data/home/cjrisi/nocturnal-hypo-gly-prob-forecast

# Test upload script without actually uploading
DRY_RUN=true B2_BUCKET_NAME="mlflow-nocturnal-hypo" ./scripts/upload_to_b2.sh
```

This will show you:
- How many files will be uploaded
- Total size
- What would be uploaded

Review the output to make sure it looks correct.

### Step 2: Upload Paper-Critical Artifacts (Live Run)

```bash
# Load credentials
source ~/.b2_credentials/mlflow-nocturnal-hypo

# Live upload (paper-critical only: 2.27 GB)
B2_BUCKET_NAME="mlflow-nocturnal-hypo" ./scripts/upload_to_b2.sh
```

**Expected**:
- 876 files (228 from trained_models + 648 from experiments)
- 2.27 GB total
- Time: 30-60 minutes depending on upload speed

The script will:
- ✅ Skip files already uploaded (resume-safe)
- ✅ Verify file sizes
- ✅ Report progress
- ✅ Ask before uploading archivable tier (4 GB)

### Step 3: Verify Upload

```bash
# List uploaded files
b2 ls --recursive $B2_BUCKET_NAME

# Check total size
b2 ls --recursive --long $B2_BUCKET_NAME | awk '{sum+=$1} END {printf "%.2f GB\n", sum/1024/1024/1024}'

# Download a test file
b2 download-file-by-name $B2_BUCKET_NAME \
  "paper-critical/experiments/experiments/nocturnal_forecasting/best_by_model.csv" \
  /tmp/test_restore.csv

# Verify checksum matches
diff <(sha256sum /tmp/test_restore.csv | cut -d' ' -f1) \
     <(sha256sum experiments/nocturnal_forecasting/best_by_model.csv | cut -d' ' -f1)
```

---

## Phase 3: Document and Update Tracking

### Update Documentation

```bash
# Add B2 bucket info to BACKUP_RECOVERY.md
cat >> agent_notes/BACKUP_RECOVERY.md << 'EOF'

## Cloud Backup (Backblaze B2)

**Date**: 2026-08-06
**Bucket**: mlflow-nocturnal-hypo
**Region**: us-west-002 (or your selected region)

### Paper-Critical Artifacts (2.27 GB)
- 228 files from trained_models/
- 648 files from experiments/
- Path: `paper-critical/`

### Restore Commands

List all backed up files:
```bash
b2 ls --recursive mlflow-nocturnal-hypo
```

Download specific file:
```bash
b2 download-file-by-name mlflow-nocturnal-hypo \
  "paper-critical/experiments/nocturnal_forecasting/best_by_model.csv" \
  ./restored_file.csv
```

Restore entire paper-critical directory:
```bash
b2 sync b2://mlflow-nocturnal-hypo/paper-critical ./restored_paper_critical/
```

### Monthly Cost
- Storage: $0.01/month (2.27 GB × $0.005/GB)
- Egress: $0 (within 3x free tier)
- Total: ~$0.01/month
EOF
```

### Mark Task Complete

```bash
# Update project tracking
# (I'll do this via SQL)
```

---

## What You've Accomplished

After completing these steps:

✅ **2.27 GB of irreplaceable paper artifacts backed up to cloud**
- NeurIPS rebuttal experiment results
- Paper-critical model checkpoints
- Configuration files and logs

✅ **3-2-1 backup strategy achieved**
- 3 copies: Original (server) + Local backup (311GB backup) + Cloud (B2)
- 2 different media: Local disk + Cloud storage
- 1 offsite: Backblaze B2

✅ **Cost-effective solution**
- $0.01-0.05/month for current needs
- Scales linearly as you add more artifacts

✅ **Foundation for MLflow integration**
- B2 is S3-compatible, works with MLflow artifact storage
- Can point MLflow to B2 for automatic artifact archival (P1 task)

---

## Troubleshooting

### "Bucket name already taken"
- Bucket names are globally unique across all B2 users
- Try: `mlflow-nocturnal-hypo-robotpsych` or `nocturnal-hypo-cjrisi`
- Update `B2_BUCKET_NAME` in your commands

### "Not authorized"
- Run: `b2 authorize-account "$B2_KEY_ID" "$B2_APPLICATION_KEY"`
- Check credentials file has correct values
- Verify no spaces/newlines in key strings

### Upload fails partway through
- Script is resume-safe! Just rerun it
- Already-uploaded files are skipped (by size check)
- Continue from where it left off

### Can't find files
- Verify inventories exist: `ls -lh *_inventory.csv`
- Check file paths in inventory match actual locations
- Run inventory scripts if needed

---

## Next Steps (After This Is Complete)

1. ✅ Mark P0-4 (data-mirror-plan) as DONE
2. ⏭️ Move to P0-5 (land-rebuttal-analyses) or P0-8 (stale-branch-triage)
3. 🔮 Later: P1 MLflow setup (will use this B2 bucket!)

---

## Questions Before You Start?

- Do you want to use a different bucket name?
- Should we include archivable artifacts (4 GB) in first upload?
- Any concerns about costs or process?

**Ready to begin?** Start with Phase 1, Step 1 above!
