import logging

from app.database import get_pool
from app.models.booking import BookingCreate

log = logging.getLogger(__name__)


async def insert_booking(data: BookingCreate) -> int:
    """Insert a booking and return the booking_id."""
    pool = get_pool()
    booking_id = await pool.fetchval(
        """
        INSERT INTO ingest_db.bookings (
            pickup_date, pickup_time, pickup_location, dropoff_location,
            dropoff_location_type, customer_id, customer_phone, booking_notes,
            est_ride_duration_mins, est_ride_distance_miles, est_ride_fare,
            currency, pickup_note, dropoff_note, wait_time_mins,
            base_location, base_to_src_distance_miles, dest_to_base_distance_miles,
            est_base_to_src_duration_mins, est_dest_to_base_duration_mins,
            journey_type, flight_number, flight_journey_type,
            number_of_luggages, booking_type
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
            $21, $22, $23, $24, $25
        )
        RETURNING booking_id
        """,
        data.pickup_date,
        data.pickup_time,
        data.pickup_location,
        data.dropoff_location,
        data.dropoff_location_type,
        data.customer_id,
        data.customer_phone,
        data.booking_notes,
        data.est_ride_duration_mins,
        data.est_ride_distance_miles,
        data.est_ride_fare,
        data.currency,
        data.pickup_note,
        data.dropoff_note,
        data.wait_time_mins,
        data.base_location,
        data.base_to_src_distance_miles,
        data.dest_to_base_distance_miles,
        data.est_base_to_src_duration_mins,
        data.est_dest_to_base_duration_mins,
        data.journey_type,
        data.flight_number,
        data.flight_journey_type,
        data.number_of_luggages,
        data.booking_type,
    )
    log.info("Booking created (id=%s) for customer %s", booking_id, data.customer_id)
    return booking_id


async def get_assignment_details(booking_id: int) -> dict | None:
    """Get driver + vehicle details for a booking assignment."""
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            d.driver_name, d.phone AS driver_phone,
            v.make, v.model, v.color, v.plate_number,
            ba.ride_status
        FROM transform_db.booking_assignments ba
        JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
        JOIN ingest_db.vehicles v ON v.vehicle_id = ba.vehicle_id
        WHERE ba.booking_id = $1
        ORDER BY ba.assignment_id DESC
        LIMIT 1
        """,
        booking_id,
    )
    if row:
        return dict(row)
    return None


async def get_latest_booking_id(customer_id: int) -> int | None:
    """Get the most recent booking_id for a customer."""
    pool = get_pool()
    return await pool.fetchval(
        """
        SELECT booking_id FROM ingest_db.bookings
        WHERE customer_id = $1
        ORDER BY booking_id DESC
        LIMIT 1
        """,
        customer_id,
    )


async def get_upcoming_bookings(customer_id: int) -> list[dict]:
    """Get all upcoming bookings for a customer."""
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT booking_id, pickup_date, pickup_time,
               pickup_location, dropoff_location, journey_type
        FROM ingest_db.bookings
        WHERE customer_id = $1
          AND pickup_date >= CURRENT_DATE
        ORDER BY pickup_date ASC, pickup_time ASC
        LIMIT 5
        """,
        customer_id,
    )
    return [dict(r) for r in rows]
