-- PostgreSQL trigger for real-time driver assignment notifications
-- Run this script against the cab_logistics_test database once.
--
-- Usage:
--   psql -h 216.48.190.41 -U cab_db_user_rw -d cab_logistics_test -f trigger.sql

-- Create the notification function in ingest_db schema (where we have write access)
CREATE OR REPLACE FUNCTION ingest_db.notify_booking_assignment()
RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('booking_assigned', json_build_object(
        'assignment_id', NEW.assignment_id,
        'booking_id', NEW.booking_id,
        'driver_id', NEW.driver_id,
        'vehicle_id', NEW.vehicle_id
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop the trigger if it already exists (idempotent)
DROP TRIGGER IF EXISTS trg_booking_assignment ON transform_db.booking_assignments;

-- Create the trigger on the transform_db table, pointing to the ingest_db function
CREATE TRIGGER trg_booking_assignment
AFTER INSERT ON transform_db.booking_assignments
FOR EACH ROW EXECUTE FUNCTION ingest_db.notify_booking_assignment();
