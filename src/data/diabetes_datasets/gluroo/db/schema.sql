-- Active: 1768019085766@@127.0.0.1@5433@postgres

DROP TABLE IF EXISTS messages CASCADE;

DROP TABLE IF EXISTS readings;

DROP TABLE IF EXISTS groups;

CREATE TABLE IF NOT EXISTS readings (
    gid TEXT NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    bgl INT NOT NULL,
    trend trend_type NULL,
    source TEXT,
    PRIMARY KEY (gid, date)
);

CREATE TABLE IF NOT EXISTS groups (
    gid TEXT NOT NULL PRIMARY KEY, -- Obscured UUID
    p_num TEXT, -- Sequential string ID (0-indexed) for efficient partitioning gluroo_{p_num}
    num_members INT,
    date_created TIMESTAMPTZ NOT NULL DEFAULT now(),
    rapid_insulin TEXT,
    long_insulin TEXT,
    regular_insulin TEXT,
    insulin_delivery TEXT,
    bg_meter TEXT,
    group_timezone TEXT,
    date_onboarding TIMESTAMPTZ,
    who_help TEXT,
    who_help_other TEXT,
    age_at_onboarding INT,
    time_since_diagnosis_lower_days INT,
    time_since_diagnosis_upper_days INT,
    goals TEXT,
    use_cgm TEXT, -- This can be no_insulin, no_deciding, yes and ...
    use_pump TEXT, -- Same as use_cgm but for pump
    onboarding_bg_meter TEXT,
    onboarding_insulin_delivery TEXT,
    user_type TEXT,
    gender TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL NOT NULL,
    sender_id TEXT, -- Shorter obfuscated ID for sender
    gid TEXT NOT NULL, -- FK to groups table (TODO: Add FOREIGN KEY constraint)
    text TEXT,
    template TEXT,
    date TIMESTAMPTZ NOT NULL DEFAULT now(),
    original_date TIMESTAMPTZ,
    type TEXT NOT NULL DEFAULT 'TEXT',
    affects_fob TEXT NOT NULL DEFAULT 'f',
    affects_iob TEXT NOT NULL DEFAULT 'f',
    bgl INT,
    bgl_date TIMESTAMPTZ,
    trend TEXT,
    description TEXT,
    food_g FLOAT,
    food_glycemic_index FLOAT,
    exercise_mins INT,
    exercise_level TEXT,
    dose_units FLOAT,
    dose_type TEXT,
    fp_bgl INT,
    device_code TEXT,
    device_lot TEXT,
    device_location TEXT,
    response_message_id INT,
    to_name TEXT,
    from_name TEXT,
    planned_duration_hours FLOAT,
    fraction_change FLOAT,
    PRIMARY KEY (id, date, gid)
);

-- Convert time-series tables to hypertables (only if TimescaleDB is available)
-- Optimized for single-patient (gid) queries rather than temporal queries
-- Using larger time chunks (30 days) since temporal queries are less important
-- This reduces the number of time chunks to scan when querying a single patient
DO $$
BEGIN
    -- Convert readings table to hypertable
    PERFORM create_hypertable (
        'readings', 'date', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE
    );

    -- Add space partitioning by gid for fast single-patient queries
    -- Using 128 partitions to create smaller sub-chunks per gid for faster lookups
    -- With 100k+ gids, this means ~780 gids per partition, making single-patient queries very fast
    PERFORM add_dimension (
        'readings', 'gid', number_partitions => 128, if_not_exists => TRUE
    );

    -- Convert messages table to hypertable partitioned by date
    PERFORM create_hypertable (
        'messages', 'date', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE
    );

    -- Add space partitioning by gid for messages as well
    PERFORM add_dimension (
        'messages', 'gid', number_partitions => 128, if_not_exists => TRUE
    );

    -- Create indexes for better query performance without TimescaleDB
    CREATE INDEX IF NOT EXISTS readings_gid_date_idx ON readings (gid, date);
    CREATE INDEX IF NOT EXISTS messages_gid_date_idx ON messages (gid, date);
    CREATE INDEX IF NOT EXISTS messages_gid_idx ON messages (gid);

    -- Create index for p_num lookups
    CREATE INDEX IF NOT EXISTS groups_p_num_idx ON groups (p_num);
    CREATE UNIQUE INDEX IF NOT EXISTS groups_p_num_unique_idx ON groups (p_num);
END $$;
