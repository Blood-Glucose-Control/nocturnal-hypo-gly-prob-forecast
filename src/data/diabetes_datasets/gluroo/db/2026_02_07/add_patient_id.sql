/*
=============================================================================
Add Patient IDs (p_num) to Groups Table
=============================================================================

USAGE: Run this script AFTER importing data with insert_data.sql
This assigns sequential patient IDs (gluroo_0, gluroo_1, etc.) to each group

Using Apptainer (inside the container):
    psql -h 127.0.0.1 -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/add_patient_id.sql

Using Docker:
  docker exec timescaledb psql -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/add_patient_id.sql

=============================================================================
*/

-- Add string p_num column to groups table
-- Format: gluroo_{p_num} where p_num is 0-indexed
-- This makes partitioning much easier than using UUID-based indexing

-- Add p_num column (nullable first, then we'll populate it)
ALTER TABLE groups ADD COLUMN IF NOT EXISTS p_num VARCHAR(50);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS groups_p_num_idx ON groups (p_num);

-- Assign sequential string IDs to existing groups (sorted by gid for deterministic ordering)
-- Format: gluroo_0, gluroo_1, gluroo_2, etc.
-- This ensures consistent p_num assignment across runs
UPDATE groups
SET
    p_num = 'gluroo_' || (subquery.row_num - 1)::TEXT -- 0-indexed
FROM (
        SELECT gid, ROW_NUMBER() OVER (
                ORDER BY gid
            ) as row_num
        FROM groups
    ) AS subquery
WHERE
    groups.gid = subquery.gid;

-- Make p_num NOT NULL after populating
ALTER TABLE groups ALTER COLUMN p_num SET NOT NULL;

-- Create unique constraint to ensure one-to-one mapping
CREATE UNIQUE INDEX IF NOT EXISTS groups_p_num_unique_idx ON groups (p_num);
