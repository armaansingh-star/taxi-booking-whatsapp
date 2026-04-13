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
            number_of_luggages, booking_type, booking_source
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
            $21, $22, $23, $24, $25, 'Whatsapp'
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


async def cancel_assigned_ride(booking_id: int) -> dict | None:
    """Cancel an assigned ride and return the assigned driver's phone."""
    pool = get_pool()
    row = await pool.fetchrow(
        """
        UPDATE transform_db.booking_assignments ba
        SET ride_status = 'Cancelled'
        FROM ingest_db.drivers d
        WHERE ba.booking_id = $1
          AND d.driver_id = ba.driver_id
        RETURNING d.phone AS driver_phone
        """,
        booking_id,
    )
    if row:
        return dict(row)
    return None


async def has_assignment(booking_id: int) -> bool:
    """Return True if a booking already has an assignment row."""
    pool = get_pool()
    exists = await pool.fetchval(
        """
        SELECT EXISTS(
            SELECT 1
            FROM transform_db.booking_assignments
            WHERE booking_id = $1
        )
        """,
        booking_id,
    )
    return bool(exists)


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


async def get_active_booking_statuses(customer_id: int, limit: int = 3) -> list[dict]:
    """Get upcoming bookings with latest assignment details when available."""
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT
            b.booking_id,
            b.pickup_date,
            b.pickup_time,
            b.pickup_location,
            b.dropoff_location,
            ba.ride_status,
            d.driver_name,
            d.phone AS driver_phone,
            v.make,
            v.model,
            v.color,
            v.plate_number
        FROM ingest_db.bookings b
        LEFT JOIN LATERAL (
            SELECT ba.driver_id, ba.vehicle_id, ba.ride_status, ba.assignment_id
            FROM transform_db.booking_assignments ba
            WHERE ba.booking_id = b.booking_id
            ORDER BY ba.assignment_id DESC
            LIMIT 1
        ) ba ON TRUE
        LEFT JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
        LEFT JOIN ingest_db.vehicles v ON v.vehicle_id = ba.vehicle_id
        WHERE b.customer_id = $1
          AND b.pickup_date >= CURRENT_DATE
        ORDER BY b.pickup_date ASC, b.pickup_time ASC
        LIMIT $2
        """,
        customer_id,
        limit,
    )
    return [dict(r) for r in rows]


async def get_customer_booking(customer_id: int, booking_id: int) -> dict | None:
    """Get a specific booking for a customer."""
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            booking_id, pickup_date, pickup_time, pickup_location, dropoff_location,
            dropoff_location_type, booking_notes, wait_time_mins, journey_type,
            flight_number, flight_journey_type, number_of_luggages, customer_phone
        FROM ingest_db.bookings
        WHERE customer_id = $1 AND booking_id = $2
        LIMIT 1
        """,
        customer_id,
        booking_id,
    )
    if row:
        return dict(row)
    return None


async def get_booking_status_for_customer(customer_id: int, booking_id: int | None = None) -> dict | None:
    """Get booking + latest assignment details for a specific customer booking.

    If booking_id is omitted, default to the most recent upcoming booking.
    """
    pool = get_pool()
    if booking_id is not None:
        row = await pool.fetchrow(
            """
            SELECT
                b.booking_id,
                b.pickup_date,
                b.pickup_time,
                b.pickup_location,
                b.dropoff_location,
                ba.ride_status,
                d.driver_name,
                d.phone AS driver_phone,
                v.make,
                v.model,
                v.color,
                v.plate_number
            FROM ingest_db.bookings b
            LEFT JOIN LATERAL (
                SELECT ba.driver_id, ba.vehicle_id, ba.ride_status, ba.assignment_id
                FROM transform_db.booking_assignments ba
                WHERE ba.booking_id = b.booking_id
                ORDER BY ba.assignment_id DESC
                LIMIT 1
            ) ba ON TRUE
            LEFT JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
            LEFT JOIN ingest_db.vehicles v ON v.vehicle_id = ba.vehicle_id
            WHERE b.customer_id = $1
              AND b.booking_id = $2
            LIMIT 1
            """,
            customer_id,
            booking_id,
        )
    else:
        row = await pool.fetchrow(
            """
            SELECT
                b.booking_id,
                b.pickup_date,
                b.pickup_time,
                b.pickup_location,
                b.dropoff_location,
                ba.ride_status,
                d.driver_name,
                d.phone AS driver_phone,
                v.make,
                v.model,
                v.color,
                v.plate_number
            FROM ingest_db.bookings b
            LEFT JOIN LATERAL (
                SELECT ba.driver_id, ba.vehicle_id, ba.ride_status, ba.assignment_id
                FROM transform_db.booking_assignments ba
                WHERE ba.booking_id = b.booking_id
                ORDER BY ba.assignment_id DESC
                LIMIT 1
            ) ba ON TRUE
            LEFT JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
            LEFT JOIN ingest_db.vehicles v ON v.vehicle_id = ba.vehicle_id
            WHERE b.customer_id = $1
              AND b.pickup_date >= CURRENT_DATE
            ORDER BY b.pickup_date ASC, b.pickup_time ASC
            LIMIT 1
            """,
            customer_id,
        )
    if row:
        return dict(row)
    return None


async def update_booking(
    booking_id: int,
    customer_phone: str,
    *,
    pickup_date,
    pickup_time,
    pickup_location: str,
    dropoff_location: str,
    dropoff_location_type: str | None,
    booking_notes: str | None,
    wait_time_mins: int | None,
    journey_type: str | None,
    flight_number: str | None,
    flight_journey_type: str | None,
    number_of_luggages: int | None,
) -> int | None:
    """Update an existing unassigned booking and return its id."""
    pool = get_pool()
    return await pool.fetchval(
        """
        UPDATE ingest_db.bookings
        SET
            pickup_date = $1,
            pickup_time = $2,
            pickup_location = $3,
            dropoff_location = $4,
            dropoff_location_type = $5,
            booking_notes = $6,
            wait_time_mins = $7,
            journey_type = $8,
            flight_number = $9,
            flight_journey_type = $10,
            number_of_luggages = $11
        WHERE booking_id = $12
          AND customer_phone = $13
        RETURNING booking_id
        """,
        pickup_date,
        pickup_time,
        pickup_location,
        dropoff_location,
        dropoff_location_type,
        booking_notes,
        wait_time_mins,
        journey_type,
        flight_number,
        flight_journey_type,
        number_of_luggages,
        booking_id,
        customer_phone,
    )
