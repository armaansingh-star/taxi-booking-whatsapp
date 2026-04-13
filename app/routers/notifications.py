import logging
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings
from app.database import get_pool
from app.services import messaging_service

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API credentials"
        )


router = APIRouter(
    prefix="/api/notifications",
    tags=["Notifications"],
    dependencies=[Depends(verify_api_key)]
)

log = logging.getLogger(__name__)


def _as_whatsapp_destination(phone: str | None) -> str | None:
    if not phone:
        return None
    normalized = phone.strip()
    if not normalized:
        return None
    if normalized.startswith("whatsapp:"):
        return normalized
    return f"whatsapp:{normalized}"


@router.post("/dispatch-tomorrow")
async def dispatch_tomorrow():
    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT
            d.phone AS driver_phone,
            d.driver_name,
            b.booking_id,
            b.pickup_date,
            b.pickup_time,
            b.pickup_location,
            b.dropoff_location,
            b.customer_phone,
            b.booking_notes
        FROM ingest_db.bookings b
        JOIN LATERAL (
            SELECT ba.driver_id, ba.assignment_id, ba.ride_status
            FROM transform_db.booking_assignments ba
            WHERE ba.booking_id = b.booking_id
            ORDER BY ba.assignment_id DESC
            LIMIT 1
        ) ba ON TRUE
        JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
        WHERE b.pickup_date = CURRENT_DATE + 1
          AND ba.driver_id IS NOT NULL
          AND COALESCE(ba.ride_status, '') <> 'Cancelled'
        ORDER BY d.phone ASC, b.pickup_time ASC, b.booking_id ASC
        """
    )

    grouped: dict[str, dict] = defaultdict(lambda: {"driver_name": None, "items": []})
    for row in rows:
        phone = row["driver_phone"]
        if not phone:
            continue
        grouped[phone]["driver_name"] = row["driver_name"]
        grouped[phone]["items"].append(dict(row))

    notified = 0
    for driver_phone, payload in grouped.items():
        destination = _as_whatsapp_destination(driver_phone)
        if not destination:
            continue
        for item in payload["items"]:
            variables = {
                "1": str(item["booking_id"]),
                "2": str(item["pickup_date"]),
                "3": str(item["pickup_time"]),
                "4": item["pickup_location"],
                "5": item["dropoff_location"],
                "6": str(item["customer_phone"]),
                "7": item["booking_notes"] or "None",
            }
            await messaging_service.send_template(
                destination,
                "HX60eb7e1b87a60b4df9bea37ad530f0cb",
                variables,
            )
            notified += 1

    return {"status": "ok", "drivers_notified": notified}


@router.post("/booking/{booking_id}/customer")
async def notify_customer_for_booking(booking_id: int):
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            b.booking_id,
            b.customer_phone,
            b.pickup_date,
            b.pickup_time,
            b.pickup_location,
            b.dropoff_location,
            d.driver_name,
            d.phone AS driver_phone,
            v.make,
            v.model,
            v.plate_number
        FROM ingest_db.bookings b
        LEFT JOIN LATERAL (
            SELECT ba.driver_id, ba.vehicle_id, ba.assignment_id
            FROM transform_db.booking_assignments ba
            WHERE ba.booking_id = b.booking_id
            ORDER BY ba.assignment_id DESC
            LIMIT 1
        ) ba ON TRUE
        LEFT JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
        LEFT JOIN ingest_db.vehicles v ON v.vehicle_id = ba.vehicle_id
        WHERE b.booking_id = $1
        LIMIT 1
        """,
        booking_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")

    destination = _as_whatsapp_destination(row["customer_phone"])
    if not destination:
        raise HTTPException(status_code=400, detail="Customer phone unavailable")

    driver_info = "Not assigned yet"
    if row["driver_name"]:
        car = f"{row['make']} {row['model']} (Plate: {row['plate_number']})" if row['plate_number'] else "Vehicle TBD"
        driver_info = f"{row['driver_name']} | {row['driver_phone']} | {car}"

    variables = {
        "1": str(row["booking_id"]),
        "2": str(row["pickup_date"]),
        "3": str(row["pickup_time"]),
        "4": row["pickup_location"],
        "5": row["dropoff_location"],
        "6": driver_info,
    }
    await messaging_service.send_template(
        destination,
        "HX54da0b2b9ec531a7df325fba5fc81b99",
        variables,
    )
    return {"status": "ok", "booking_id": booking_id, "recipient": "customer"}


@router.post("/booking/{booking_id}/driver")
async def notify_driver_for_booking(booking_id: int):
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            b.booking_id,
            b.customer_phone,
            b.pickup_date,
            b.pickup_time,
            b.pickup_location,
            b.dropoff_location,
            b.booking_notes,
            d.driver_name,
            d.phone AS driver_phone
        FROM ingest_db.bookings b
        JOIN LATERAL (
            SELECT ba.driver_id, ba.assignment_id
            FROM transform_db.booking_assignments ba
            WHERE ba.booking_id = b.booking_id
            ORDER BY ba.assignment_id DESC
            LIMIT 1
        ) ba ON TRUE
        JOIN ingest_db.drivers d ON d.driver_id = ba.driver_id
        WHERE b.booking_id = $1
        LIMIT 1
        """,
        booking_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Assigned booking not found")

    destination = _as_whatsapp_destination(row["driver_phone"])
    if not destination:
        raise HTTPException(status_code=400, detail="Driver phone unavailable")

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
        destination,
        "HX60eb7e1b87a60b4df9bea37ad530f0cb",
        variables,
    )
    return {"status": "ok", "booking_id": booking_id, "recipient": "driver"}
