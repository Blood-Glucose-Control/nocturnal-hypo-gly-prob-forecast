/*
=============================================================================
Import Groups Data Only
=============================================================================

USAGE: Run this script to import only groups data
NOTE: This script reads .csv.gz files directly using zcat

Inside the container:
psql -h 127.0.0.1 -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/insert_groups.sql

Using Docker:
docker exec timescaledb psql -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/insert_groups.sql

=============================================================================
*/

-- Insert groups data
\COPY groups(gid,num_members,date_created,rapid_insulin,long_insulin,regular_insulin,insulin_delivery,bg_meter,group_timezone,date_onboarding,who_help,who_help_other,age_at_onboarding,time_since_diagnosis_lower_days,time_since_diagnosis_upper_days,goals,use_cgm,use_pump,onboarding_bg_meter,onboarding_insulin_delivery,user_type,gender) FROM PROGRAM 'zcat /data/shared/cache/data/gluroo_2026/raw/groups.csv.gz' WITH (FORMAT csv, HEADER true, DELIMITER ',');