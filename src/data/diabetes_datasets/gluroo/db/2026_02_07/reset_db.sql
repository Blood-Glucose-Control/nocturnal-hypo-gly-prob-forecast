/*
=============================================================================
Reset Database - Clear All Data
=============================================================================

USAGE: Run this script to clear all data from the database tables
WARNING: This will DELETE all data from groups, readings, and messages tables!

Using Apptainer (inside the container):
    psql -h 127.0.0.1 -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/reset_db.sql

Using Docker:
  docker exec timescaledb psql -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/reset_db.sql

TYPICAL WORKFLOW:
  1. Run reset_db.sql (this file) to clear existing data
  2. Run insert_data.sql to import fresh data
  3. Run add_patient_id.sql to assign patient IDs

=============================================================================
*/

-- Fix source column to allow NULL values (in case schema wasn't re-run)
ALTER TABLE readings ALTER COLUMN source DROP NOT NULL;

-- Change id from SERIAL to INT if needed (in case schema wasn't re-run)
ALTER TABLE messages ALTER COLUMN id DROP DEFAULT;

DROP SEQUENCE IF EXISTS messages_id_seq CASCADE;

-- Clear existing data before importing
TRUNCATE TABLE messages CASCADE;

TRUNCATE TABLE readings;

TRUNCATE TABLE groups;
