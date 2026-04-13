import asyncio
import logging

from app.database import get_pool
from app.services import messaging_service

log = logging.getLogger(__name__)

# Tracks the highest assignment_id we've already processed
_last_seen_id: int | None = None

POLL_INTERVAL_SECONDS = 10


def _as_whatsapp_destination(phone: str | None) -> str | None:
    if not phone:
        return None
    normalized = phone.strip()
    if not normalized:
        return None
    if normalized.startswith("whatsapp:"):
        return normalized
    return f"whatsapp:{normalized}"


async def start_assignment_listener():
    """Poll transform_db.booking_assignments for new rows every N seconds.

    Fallback for when we don't have DB permissions to create
    LISTEN/NOTIFY triggers. Switch to LISTEN/NOTIFY once access is granted.
    """
    global _last_seen_id

    # Wait for DB pool to be ready
    await asyncio.sleep(2)

    pool = get_pool()

    # Seed with the current max assignment_id so we don't replay old rows
    try:
        _last_seen_id = await pool.fetchval(
            "SELECT COALESCE(MAX(assignment_id), 0) FROM transform_db.booking_assignments"
        )
        log.info("Assignment poller started (last_seen_id=%s)", _last_seen_id)
    except Exception:
        log.exception("Failed to seed last_seen_id, starting from 0")
        _last_seen_id = 0

    while True:
        try:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            await _poll_new_assignments()
        except asyncio.CancelledError:
            log.info("Assignment poller cancelled")
            break
        except Exception:
            log.exception("Assignment poller error, will retry next cycle")


async def _poll_new_assignments():
    """Check for any new assignments since last poll."""
    global _last_seen_id
    pool = get_pool()

    rows = await pool.fetch(
        """
        SELECT
            ba.assignment_id, ba.booking_id,
            d.driver_name, d.phone AS driver_phone,
            v.make, v.model, v.color, v.plate_number,
            b.customer_phone,
            b.pickup_date, b.pickup_time,
            b.pickup_location, b.dropoff_location, b.booking_notes
        FROM transform_db.booking_assignments ba
        JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
        JOIN ingest_db.vehicles v ON v.vehicle_id = ba.vehicle_id
        JOIN ingest_db.bookings b ON b.booking_id = ba.booking_id
        WHERE ba.assignment_id > $1
        ORDER BY ba.assignment_id ASC
        """,
        _last_seen_id,
    )

    for row in rows:
        customer_phone = f"whatsapp:{row['customer_phone']}"

        message = (
            "Your driver has been assigned!\n\n"
            f"Booking #{row['booking_id']}\n"
            f"Pickup: {row['pickup_location']}\n"
            f"Dropoff: {row['dropoff_location']}\n"
            f"Pickup Date/Time: {row['pickup_date']} at {row['pickup_time']}\n\n"
            f"Driver: {row['driver_name']}\n"
            f"Vehicle: {row['color']} {row['make']} {row['model']}\n"
            f"Plate: {row['plate_number']}\n"
            f"Driver Phone: {row['driver_phone']}"
        )

        await messaging_service.send(customer_phone, message)
        driver_phone = _as_whatsapp_destination(row["driver_phone"])
        if driver_phone:
            variables = {
                "1": str(row["booking_id"]),
                "2": str(row["pickup_date"]),
                "3": str(row["pickup_time"]),
                "4": row["pickup_location"],
                "5": row["dropoff_location"],
                "6": str(row["customer_phone"]),
                "7": row["booking_notes"] or "None",
            }
            await messaging_service.send_template(
                driver_phone,
                "HX60eb7e1b87a60b4df9bea37ad530f0cb",
                variables,
            )
        log.info(
            "Driver assignment notification sent to %s for booking %s",
            customer_phone,
            row["booking_id"],
        )

        _last_seen_id = row["assignment_id"]
