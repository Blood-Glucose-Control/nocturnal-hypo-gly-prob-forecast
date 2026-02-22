reference: https://www.tigerdata.com/docs/self-hosted/latest/install/installation-docker

## To symlink the cache directory to the shared cache directory
At the root of the project:
```bash
ln -s /data/shared/cache cache
```

## To remove the container
```bash
docker stop timescaledb
docker rm timescaledb
```


## To start a new container
```bash
docker run -d \
    --name timescaledb \
    -p 5432:5432 \
    -v </a/local/data/folder>:/pgdata \
    -e PGDATA=/pgdata \
    -e POSTGRES_PASSWORD=password \
    timescale/timescaledb-ha:pg18
```

For full example:
```bash
docker run -d \
  --name timescaledb \
  -p 5433:5432 \
  -v /Users/tonychan/GlucoseML/timescaleDB:/pgdata \
  -e PGDATA=/pgdata \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=gluroo_datasets \
  timescale/timescaledb-ha:pg18
```
- 5433:5432 is the port mapping (Note that the port is 5433 if 5432 is already in use)
- /Users/tonychan/GlucoseML/timescaleDB is the volume location
- password is the password for the database.
- gluroo_datasets is the database name.
- timescale/timescaledb-ha:pg18 is the image to use.

## Start the container:
```bash
docker start timescaledb
```

## Connection:
```bash
psql -d "postgresql://postgres:password@127.0.0.1:5433/gluroo_datasets"
```

To create a new database:
```bash
psql -d "postgres://postgres:password@127.0.0.1:5433/gluroo_datasets" -f src/data/diabetes_datasets/gluroo/db/schema.sql
```

To insert data:
```bash
psql -d "postgres://postgres:password@127.0.0.1:5433/gluroo_datasets" -f src/data/diabetes_datasets/gluroo/db/insert_data.sql
```

To add a p_num column to the groups table:
Reason we need to do this is because we need to partition the data by p_num for faster lookups.
```bash
psql -d "postgres://postgres:password@127.0.0.1:5433/gluroo_datasets" -f src/data/diabetes_datasets/gluroo/db/add_patient_id.sql
```
