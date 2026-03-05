#!/usr/bin/env bash

# Usage: ./src/data/diabetes_datasets/gluroo/db/setup.sh

DATABASE_URL=postgres://postgres:password@127.0.0.1:5433/gluroo_datasets

# Redact password in logs: postgres://user:pass@host/db -> postgres://user:***@host/db
SAFE_DATABASE_URL="$(printf '%s' "${DATABASE_URL}" | sed -E 's#(://[^:/]+:)[^@]+@#\\1***@#')"
echo "Creating new database (schema)... Schema will be created in: ${SAFE_DATABASE_URL}"

# # To create a new database (schema)
psql -d "${DATABASE_URL}" -f src/data/diabetes_datasets/gluroo/db/schema.sql

# To insert data
psql -d "${DATABASE_URL}" -f src/data/diabetes_datasets/gluroo/db/insert_data.sql

# To add a p_num column to the groups table (partitioning support)
psql -d "${DATABASE_URL}" -f src/data/diabetes_datasets/gluroo/db/add_patient_id.sql
