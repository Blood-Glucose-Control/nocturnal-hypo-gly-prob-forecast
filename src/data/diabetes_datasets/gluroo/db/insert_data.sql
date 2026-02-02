-- Fix source column to allow NULL values (in case schema wasn't re-run)
ALTER TABLE readings ALTER COLUMN source DROP NOT NULL;

-- Change id from SERIAL to INT if needed (in case schema wasn't re-run)
ALTER TABLE messages ALTER COLUMN id DROP DEFAULT;

DROP SEQUENCE IF EXISTS messages_id_seq CASCADE;

-- Clear existing data before importing
TRUNCATE TABLE messages CASCADE;

TRUNCATE TABLE readings;

TRUNCATE TABLE groups;

-- Import groups data
\COPY groups(gid,num_members,date_created,rapid_insulin,long_insulin,regular_insulin,insulin_delivery,bg_meter,group_timezone,date_onboarding,who_help,who_help_other,age_at_onboarding,time_since_diagnosis_lower_days,time_since_diagnosis_upper_days,goals,use_cgm,use_pump,onboarding_bg_meter,onboarding_insulin_delivery,user_type,gender) FROM '/Users/tonychan/GlucoseML/nocturnal-hypo-gly-prob-forecast/cache/data/gluroo/raw/groups.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Import messages data (specifying column order to match CSV: gid,id,sender_id,...)
\COPY messages(gid,id,sender_id,text,template,date,original_date,type,affects_fob,affects_iob,bgl,bgl_date,trend,description,food_g,food_glycemic_index,exercise_mins,exercise_level,dose_units,dose_type,fp_bgl,device_code,device_lot,device_location,response_message_id,to_name,from_name,planned_duration_hours,fraction_change) FROM '/Users/tonychan/GlucoseML/nocturnal-hypo-gly-prob-forecast/cache/data/gluroo/raw/messages.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Import readings data
\COPY readings FROM '/Users/tonychan/GlucoseML/nocturnal-hypo-gly-prob-forecast/cache/data/gluroo/raw/readings.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- After importing, assign integer p_num to each patient
-- Run: \i add_patient_id.sql
-- Or manually: Assign sequential integer IDs based on gid sort order
