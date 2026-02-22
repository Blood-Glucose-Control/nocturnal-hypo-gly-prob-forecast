### This is for running TimescaleDB in an Apptainer container (mostly used for HPC clusters)
See timescaledb.md for more details on running TimescaleDB in a Docker container.


## To symlink the cache directory to the shared cache directory
At the root of the project:
```bash
ln -s /data/shared/cache cache
```

## Setup Instructions

### 1. Create the data directory
```bash
mkdir -p ~/timescaleDB
```

### 2. Start TimescaleDB (runs in background)
```bash
nohup apptainer run \
  --bind ~/timescaleDB:/pgdata \
  --env PGDATA=/pgdata \
  --env POSTGRES_PASSWORD=password \
  --env POSTGRES_DB=gluroo_datasets \
  --env POSTGRES_HOST_AUTH_METHOD=trust \
  --writable-tmpfs \
  docker://timescale/timescaledb-ha:pg18 \
  -c listen_addresses='*' \
  > ~/timescale.log 2>&1 &
```

**Note:** The `-c listen_addresses='*'` is critical for network connectivity. Without it, PostgreSQL only listens on Unix sockets.

### 5. Connect to the database

Since `psql` is not installed on the host (Maybe we can ask Indy about it), use it from within the container:
```bash
apptainer exec \
  --bind ~/timescaleDB:/pgdata \
  docker://timescale/timescaledb-ha:pg18 \
  psql -h 127.0.0.1 -U postgres -d gluroo_datasets
```


### 6. Enter interactive mode
```bash
apptainer shell \
  --bind ~/timescaleDB:/pgdata \
  --bind /data/home/t3chan/bgc/nocturnal-hypo-gly-prob-forecast:/workspace \
  --bind /data/shared/cache:/data/shared/cache \
  docker://timescale/timescaledb-ha:pg18
```


### 7. Import data
1. Run `insert_groups.sql` to import groups data
2. Run `insert_messages.sql` to import messages data
3. Run `insert_readings.sql` to import readings data
4. Run `add_patient_id.sql` to add patient id to the groups table

### 8. Import data
1. Run `schema.sql` to create the database schema
2. Run `reset_db.sql` to reset the database (CAN BE SKIPPED IF YOU DON'T WANT TO RESET THE DATABASE)
3. Run `insert_groups.sql` to import groups data
4. Run `insert_messages.sql` to import messages data
5. Run `insert_readings.sql` to import readings data (there might be 12 csv.gz files, so we need to run this 12 times.)
6. Run `add_patient_id.sql` to add patient id to the groups table


## Multi-User Access
Once running, **other users on the VM can connect** to the database using:
- Host: `127.0.0.1` or the server's IP address
- Port: `5432`
- Database: `gluroo_datasets`
- User: `postgres`
- Password: `password`