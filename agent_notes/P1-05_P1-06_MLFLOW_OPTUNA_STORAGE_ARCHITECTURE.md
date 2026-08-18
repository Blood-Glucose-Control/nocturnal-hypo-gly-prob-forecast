# MLflow + Optuna Storage Architecture Design

**Created**: 2026-08-06 20:58 UTC
**For**: P0-4 data-mirror-plan + P1 MLflow/Optuna integration
**Status**: DRAFT - awaiting user approval

---

## Problem Statement

**Current state**:
- 290 GB trained models in local filesystem
- No centralized experiment tracking
- Manual CSV inventories
- No query interface for results
- No automated artifact management

**Environment**:
- 🖥️ SSH server environment (not always-on)
- 🔒 Training/eval data behind ZFS-lock (secure, can't be moved)
- 🔄 Server can be restarted/shut down periodically
- ✅ No need for 24/7 uptime

**Requirements**:
1. ✅ Gluroo training/eval data MUST stay on this server (PHI/proprietary + ZFS-locked)
2. ✅ MLflow tracking for experiments (P1 task)
3. ✅ Optuna for hyperparameter tuning (P1 task)
4. ✅ Large file storage (100+ GB models)
5. ✅ Safe, recoverable, queryable
6. ✅ Cost-effective (<$5/month ideally)
7. ✅ More organized than current scattered directories
8. ✅ Resilient to server restarts (no data loss)

---

## Recommended Architecture: Hybrid Local + Cloud

```
┌────────────────────────────────────────────────────────┐
│ Research Server (this machine)                         │
├────────────────────────────────────────────────────────┤
│                                                         │
│  PostgreSQL Database                                   │
│  ├─ MLflow Tracking (experiments, metrics, params)     │
│  └─ Optuna Studies (trials, hyperparameters)           │
│                                                         │
│  MLflow Tracking Server (self-hosted, port 5000)       │
│  └─ Web UI for querying experiments                    │
│                                                         │
│  Local Artifact Storage (fast disk)                    │
│  ├─ Active experiments (hot storage)                   │
│  ├─ Recent models (<3 months)                          │
│  ├─ Training/eval datasets (Gluroo - MUST STAY HERE)   │
│  └─ Cache/temp files                                   │
│                                                         │
└────────────────────────────────────────────────────────┘
           │
           ├──> Local Backup Disk
           │    - Daily PostgreSQL dumps
           │    - Rsync of critical artifacts
           │    - Full repo backup (current 311GB backup location)
           │
           └──> S3-Compatible Cold Storage (Backblaze B2 or Wasabi)
                - Paper-critical artifacts (2.3 GB)
                - Archivable models (4 GB)
                - Old experiment results (>3 months)
                - Automatic lifecycle: hot → warm → cold
                - Cost: ~$2-3/month for 300GB
```

---

## Component Details

### 1. PostgreSQL Database (Free, on-server)

**Why PostgreSQL over SQLite?**
- Multi-process safe (MLflow + Optuna can run simultaneously)
- Better query performance for large experiment histories
- Supports concurrent writes from distributed training
- Easy to backup/restore

**Setup**:
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create databases
sudo -u postgres createdb mlflow_tracking
sudo -u postgres createdb optuna_studies

# Create user
sudo -u postgres createuser --interactive cjrisi
```

**Backup strategy**:
```bash
# Daily cron job
pg_dump mlflow_tracking > /backup/mlflow_$(date +%Y%m%d).sql
pg_dump optuna_studies > /backup/optuna_$(date +%Y%m%d).sql
```

**Cost**: $0 (self-hosted)
**Storage**: ~1-10 GB for tracking data (metadata only, not artifacts)

---

### 2. MLflow Tracking Server (Free, on-server)

**Configuration**:
```bash
# Start MLflow server
mlflow server \
  --backend-store-uri postgresql://cjrisi@localhost/mlflow_tracking \
  --default-artifact-root /data/mlflow_artifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Server Restart Resilience**:
Since the server can be shut down/restarted:
- PostgreSQL automatically starts on boot (systemd service)
- MLflow server can be started on-demand (doesn't need 24/7 uptime)
- All data persists in PostgreSQL (no data loss on restart)
- Experiments can query historical data even when server is down

**Systemd Service for Easy Management**:
```bash
# /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=postgresql.service

[Service]
Type=simple
User=cjrisi
WorkingDirectory=/data/home/cjrisi
Environment="PATH=/home/cjrisi/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/cjrisi/miniconda3/bin/mlflow server \
  --backend-store-uri postgresql://cjrisi@localhost/mlflow_tracking \
  --default-artifact-root /data/mlflow_artifacts \
  --host 0.0.0.0 \
  --port 5000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Usage**:
```bash
# Start when needed
sudo systemctl start mlflow

# Check status
sudo systemctl status mlflow

# Stop when done
sudo systemctl stop mlflow

# Auto-start on boot (optional)
sudo systemctl enable mlflow
```

**Features you get**:
- ✅ Web UI at http://localhost:5000 (or SSH tunnel: `ssh -L 5000:localhost:5000 cjrisi@server`)
- ✅ Search/filter experiments by metrics, params, tags
- ✅ Compare multiple runs side-by-side
- ✅ Model registry with versioning
- ✅ Automatic lineage tracking (data → model → results)
- ✅ API for programmatic access (works even when server is stopped - reads from PostgreSQL)
- ✅ Survives server restarts (all data in PostgreSQL)

**In your training code**:
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nocturnal_forecasting")

with mlflow.start_run(run_name="chronos2_rebuttal_aleppo"):
    mlflow.log_params({"model": "chronos2", "dataset": "aleppo_2017"})
    mlflow.log_metrics({"auroc": 0.85, "auprc": 0.72})
    mlflow.log_artifact("forecasts.npz")
    mlflow.log_artifact("best_worst_forecasts.png")
```

**Cost**: $0 (self-hosted)

---

### 3. Local Artifact Storage (Hot Storage)

**Location**: `/data/mlflow_artifacts/` (or existing `/data/home/cjrisi/nocturnal-hypo-gly-prob-forecast/trained_models/`)

**What stays local**:
- ✅ **ALL training/eval data** (Gluroo MUST stay - PHI/proprietary + ZFS-locked; others for speed)
- ✅ **Active experiments** (last 3 months)
- ✅ **Recent models** being developed/tuned
- ✅ **Cache files** (preprocessing, tokenization)
- ✅ **Fast-access artifacts** (daily analysis, debugging)

**ZFS-lock benefits**:
- Data can't accidentally be moved/deleted
- Snapshot capability for point-in-time recovery
- Built-in compression and checksumming
- Perfect for PHI/proprietary data security

**Lifecycle automation** (via cron or MLflow plugin):
```python
# Pseudo-code for artifact lifecycle
if artifact.age > 90_days and artifact.tier == "archivable":
    upload_to_cold_storage(artifact)
    if verify_upload_checksum():
        archive_local_copy(artifact)  # compress or delete
```

**Estimated size**: 50-100 GB active working set

---

### 4. S3-Compatible Cold Storage (Cheap Cloud)

**Recommended: Backblaze B2** (most cost-effective)

**Cost comparison** (for 300 GB):
| Provider | Storage Cost/mo | Egress | Total/mo |
|----------|----------------|--------|----------|
| **Backblaze B2** | $1.50 | $0.01/GB (first 3x storage free) | **~$1.50** |
| Wasabi | $1.77 | $0 (no egress fees) | **~$1.77** |
| AWS S3 Standard | $6.90 | $0.09/GB | $6.90+ |
| AWS S3 Glacier | $0.30 | $0.03/GB + retrieval fees | $0.30+ (slow) |
| AWS S3 Intelligent-Tier | $4.50 | $0.09/GB | $4.50+ |

**Winner: Backblaze B2**
- $0.005/GB/month storage
- 3x free egress (900GB/month free for 300GB storage)
- S3-compatible API (works with MLflow)
- Fast retrieval (not like Glacier)
- Easy Python SDK

**What goes to cold storage**:
- Paper-critical artifacts (2.3 GB) - **IMMEDIATE PRIORITY**
- Archivable models (4 GB)
- Experiment results >90 days old
- Completed study checkpoints
- Historical baselines

**MLflow integration**:
```bash
# Configure MLflow to use B2 for old artifacts
mlflow server \
  --backend-store-uri postgresql://cjrisi@localhost/mlflow_tracking \
  --default-artifact-root s3://mlflow-nocturnal-hypo/artifacts \
  --host 0.0.0.0 \
  --port 5000

# Set B2 credentials as S3-compatible
export AWS_ACCESS_KEY_ID=<b2_key_id>
export AWS_SECRET_ACCESS_KEY=<b2_application_key>
export MLFLOW_S3_ENDPOINT_URL=https://s3.us-west-002.backblazeb2.com
```

**Alternative: MinIO self-hosted**
- $0 cost but requires reliable storage hardware
- Full S3-compatible API
- Can run on the research server or separate NAS
- Good if you have spare hardware + backup strategy

**Cost**: ~$1.50-2/month for 300GB on Backblaze B2

---

### 5. Optuna Integration

**Configuration**:
```python
import optuna
from optuna.storages import RDBStorage

# Shared PostgreSQL backend
storage = RDBStorage(
    url="postgresql://cjrisi@localhost/optuna_studies",
    engine_kwargs={"pool_size": 20}
)

study = optuna.create_study(
    study_name="chronos2_hyperparameter_search",
    storage=storage,
    direction="maximize",
    load_if_exists=True
)

# Integrate with MLflow
import mlflow
from optuna.integration.mlflow import MLflowCallback

mlflow.set_tracking_uri("http://localhost:5000")
mlflc = MLflowCallback(tracking_uri="http://localhost:5000")

study.optimize(objective, n_trials=100, callbacks=[mlflc])
```

**Benefits**:
- ✅ Distributed parallel tuning (multiple workers on same study)
- ✅ Resume interrupted searches
- ✅ Pruning underperforming trials
- ✅ All trials logged to MLflow automatically
- ✅ Visual dashboard in Optuna-Dashboard package

**Cost**: $0 (uses same PostgreSQL)

---

## Migration Strategy

### Phase 1: Setup Infrastructure (Week 1)
1. Install PostgreSQL
2. Create databases (mlflow_tracking, optuna_studies)
3. Start MLflow server
4. Test with toy experiment
5. Set up Backblaze B2 account + bucket

### Phase 2: Migrate Paper-Critical Artifacts (Week 1-2)
1. Upload 2.3 GB paper-critical to B2 (from inventory)
2. Verify checksums
3. Keep local copies (double-stored for now)
4. Document restore procedure
5. Update BACKUP_RECOVERY.md

### Phase 3: Retrofit Existing Experiments (Week 2-3)
1. Write script to import experiments_inventory.csv into MLflow
2. Bulk-register existing models in MLflow Model Registry
3. Backfill metrics from results_summary.json files
4. Tag paper-critical runs

### Phase 4: Integrate New Experiments (Week 3-4)
1. Update training scripts to log to MLflow
2. Add Optuna to hyperparameter search scripts
3. Implement automatic artifact lifecycle (hot → cold)
4. Set up monitoring dashboard

### Phase 5: Archive Old Artifacts (Ongoing)
1. Move archivable (4GB) to B2
2. Compress or delete local copies after verification
3. Set up automated lifecycle policy

---

## Query Examples (What You'll Be Able to Do)

### MLflow UI Queries:
```
# Find best AUROC across all models for Aleppo dataset
Filter: params.dataset = "aleppo_2017"
Sort by: metrics.auroc DESC

# Compare all rebuttal runs
Filter: tags.experiment = "rebuttal"
Compare: side-by-side metrics table

# Find all Chronos2 models trained in April 2026
Filter: params.model = "chronos2" AND start_time >= "2026-04-01"
```

### Programmatic Queries:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://localhost:5000")

# Get best model for production
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.auroc > 0.85",
    order_by=["metrics.auprc DESC"],
    max_results=1
)

best_run = runs[0]
model_uri = f"runs:/{best_run.info.run_id}/model"
```

### Optuna Queries:
```python
# Get best hyperparameters from completed study
study = optuna.load_study(
    study_name="chronos2_hyperparameter_search",
    storage=storage
)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Plot optimization history
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

---

## Cost Breakdown

### Monthly Costs:
| Component | Cost | Notes |
|-----------|------|-------|
| PostgreSQL | $0 | Self-hosted |
| MLflow Server | $0 | Self-hosted |
| Local Storage | $0 | Already have disk |
| Backblaze B2 (300GB) | $1.50 | $0.005/GB/month |
| Egress | $0 | 3x free (900GB/month) |
| **Total** | **$1.50/month** | **Scales linearly with storage** |

### One-time Costs:
- Setup time: ~8 hours
- Learning curve: ~2 days
- Migration scripts: ~4 hours

### Scaling:
- 500 GB: $2.50/month
- 1 TB: $5.00/month
- 10 TB: $50/month (but you'd want cheaper cold storage like Glacier at that point)

---

## Disaster Recovery

### Backup Strategy (3-2-1 Rule):
1. **3 copies**: Original (server) + Local backup disk + Cloud (B2)
2. **2 different media**: Disk (local) + Cloud (B2)
3. **1 offsite**: B2 (geographically distributed)

### Recovery Procedures:

**Database loss**:
```bash
# Restore from daily dump
psql mlflow_tracking < /backup/mlflow_20260806.sql
psql optuna_studies < /backup/optuna_20260806.sql
```

**Artifact loss**:
```bash
# Restore from B2
b2 sync b2://mlflow-nocturnal-hypo/paper-critical /data/mlflow_artifacts/
```

**Complete server failure**:
1. Provision new machine
2. Install PostgreSQL + MLflow
3. Restore database dumps
4. Download artifacts from B2
5. Resume experiments (Optuna studies are resumable!)

**Recovery Time Objective (RTO)**: 4-8 hours
**Recovery Point Objective (RPO)**: 24 hours (daily backups)

---

## Additional Considerations

### Security:
- ✅ MLflow runs on localhost (not exposed to internet)
- ✅ SSH tunnel for web UI access: `ssh -L 5000:localhost:5000 cjrisi@server`
- ✅ B2 bucket is private (authenticated access only)
- ✅ PostgreSQL local connections only
- ✅ ZFS-lock protects training/eval data from accidental deletion/modification
- ⚠️ Consider VPN if accessing from other machines
- ⚠️ Rotate B2 application keys annually

### Server Restart Resilience:
**What happens during shutdown/restart?**
- ✅ PostgreSQL: Auto-starts on boot (systemd), all data persists
- ✅ MLflow server: Start on-demand when needed, no data loss
- ✅ Training jobs: Can reconnect to MLflow after restart
- ✅ Optuna studies: Fully resumable (trials stored in PostgreSQL)
- ✅ Local artifacts: Persist on disk
- ✅ B2 artifacts: Unaffected (already in cloud)

**Best practices**:
- Start MLflow when beginning research session: `sudo systemctl start mlflow`
- Stop MLflow when done: `sudo systemctl stop mlflow` (saves resources)
- Optuna trials automatically resume after restart (just reload the study)
- Training scripts should log to MLflow incrementally (not just at end)

### Monitoring:
- Disk space alerts (set threshold at 80%)
- PostgreSQL backup verification
- B2 upload success monitoring
- MLflow server uptime check

### Documentation Needs:
1. MLflow server startup procedure (systemd service)
2. B2 bucket management (lifecycle policies)
3. Artifact migration checklist
4. Common query examples
5. Troubleshooting guide

---

## Decision Points for User

### 1. Cold Storage Provider:
- [ ] **Backblaze B2** (recommended: $1.50/mo for 300GB, S3-compatible)
- [ ] **Wasabi** (alternative: $1.77/mo, no egress fees)
- [ ] **MinIO self-hosted** (free but need hardware + backup strategy)
- [ ] **AWS S3 Glacier** (cheapest $0.30/mo but slow retrieval)

### 2. Artifact Lifecycle:
- [ ] Move to cold storage after **90 days** (recommended)
- [ ] Move to cold storage after **180 days** (keep more local)
- [ ] Manual triage (no automatic lifecycle)

### 3. Migration Timing:
- [ ] **Immediate**: Set up MLflow + B2, migrate paper-critical now
- [ ] **Gradual**: Set up MLflow first, add B2 later
- [ ] **Defer**: Complete other P0 tasks first

### 4. Existing Artifacts:
- [ ] Import existing experiments into MLflow (recommended for queryability)
- [ ] Leave existing as-is, only track new experiments in MLflow
- [ ] Hybrid: Import paper-critical only

---

## Recommendation Summary

**For your use case** (researcher, tight budget, need robustness):

✅ **Do this**:
1. Install PostgreSQL + MLflow on research server (free)
2. Sign up for Backblaze B2 ($1.50/month for 300GB)
3. Migrate paper-critical artifacts (2.3GB) to B2 immediately
4. Retrofit existing experiments into MLflow for queryability
5. Integrate Optuna with MLflow for future hyperparameter searches
6. Set up daily PostgreSQL backups

✅ **Benefits**:
- Queryable experiment history via web UI
- Automatic versioning and lineage tracking
- Cheap cloud backup ($1.50/month)
- Resumable hyperparameter searches
- Professional-grade tracking (same tools used at Meta, Databricks, etc.)

✅ **Avoid**:
- ❌ AWS S3 Standard (too expensive: $6.90/month)
- ❌ Storing everything in cold storage (need fast local access)
- ❌ SQLite for MLflow backend (not multi-process safe)
- ❌ No tracking system (you'll hit the same chaos in 6 months)

---

## Next Steps

If you approve this architecture:

1. I'll update the P0-4 data-mirror-plan to use Backblaze B2
2. I'll create setup scripts for PostgreSQL + MLflow
3. I'll write the B2 upload script for paper-critical artifacts
4. I'll document the full setup procedure
5. We can add P1 tasks for MLflow integration and Optuna setup

**Does this architecture align with your needs?** Any changes or concerns?
