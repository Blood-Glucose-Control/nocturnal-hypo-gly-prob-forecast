-- Add string patient_id column to groups table
-- Format: gluroo_{patient_id} where patient_id is 0-indexed
-- This makes partitioning much easier than using UUID-based indexing

-- Add patient_id column (nullable first, then we'll populate it)
ALTER TABLE groups ADD COLUMN IF NOT EXISTS patient_id VARCHAR(50);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS groups_patient_id_idx ON groups (patient_id);

-- Assign sequential string IDs to existing groups (sorted by gid for deterministic ordering)
-- Format: gluroo_0, gluroo_1, gluroo_2, etc.
-- This ensures consistent patient_id assignment across runs
UPDATE groups
SET
    patient_id = 'gluroo_' || (subquery.row_num - 1)::TEXT -- 0-indexed
FROM (
        SELECT gid, ROW_NUMBER() OVER (
                ORDER BY gid
            ) as row_num
        FROM groups
    ) AS subquery
WHERE
    groups.gid = subquery.gid;

-- Make patient_id NOT NULL after populating
ALTER TABLE groups ALTER COLUMN patient_id SET NOT NULL;

-- Create unique constraint to ensure one-to-one mapping
CREATE UNIQUE INDEX IF NOT EXISTS groups_patient_id_unique_idx ON groups (patient_id);
