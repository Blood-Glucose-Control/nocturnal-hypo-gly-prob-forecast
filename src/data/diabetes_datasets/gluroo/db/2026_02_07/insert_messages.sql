/*
=============================================================================
Import Messages Data Only
=============================================================================

USAGE: Run this script to import only messages data
NOTE: This script reads .csv.gz files directly using zcat

Using Apptainer (inside the container):
    nohup psql -h 127.0.0.1 -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/insert_messages.sql > insert_messages.log 2>&1 &

To check the log:
cat insert_messages.log

Using Docker:
  docker exec timescaledb psql -U postgres -d gluroo_datasets -f /workspace/src/data/diabetes_datasets/gluroo/db/2026_02_07/insert_messages.sql

=============================================================================
*/

-- Import messages data (specifying column order to match CSV: gid,id,sender_id,...)
\COPY messages(gid,id,sender_id,text,template,date,original_date,type,affects_fob,affects_iob,bgl,bgl_date,trend,description,food_g,food_glycemic_index,exercise_mins,exercise_level,dose_units,dose_type,fp_bgl,device_code,device_lot,device_location,response_message_id,to_name,from_name,planned_duration_hours,fraction_change) FROM PROGRAM 'zcat /data/shared/cache/data/gluroo_2026/raw/messages.csv.gz' WITH (FORMAT csv, HEADER true, DELIMITER ',');


/*
d
*/