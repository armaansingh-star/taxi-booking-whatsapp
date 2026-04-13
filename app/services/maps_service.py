import logging
import math
import httpx

from app.config import settings
from app.models.booking import FareEstimate

log = logging.getLogger(__name__)

BASE_LAT = 51.067793
BASE_LNG = 0.686756
BASE_ORIGIN = f"{BASE_LAT},{BASE_LNG}"

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=15.0)
    return _client


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def reverse_geocode(lat: float, lng: float) -> str | None:
    """Convert lat/lng to a formatted address string."""
    try:
        r = await _get_client().get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"latlng": f"{lat},{lng}", "key": settings.GOOGLE_MAPS_API_KEY},
        )
        data = r.json()
        if data["status"] == "OK":
            return data["results"][0]["formatted_address"]
    except Exception:
        log.exception("Reverse geocode failed for (%s, %s)", lat, lng)
    return None


async def geocode_address(address: str) -> dict | None:
    """Convert an address string to structured location data.

    Returns dict with keys: formatted_address, lat, lng, town, postal_code
    """
    try:
        r = await _get_client().get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": settings.GOOGLE_MAPS_API_KEY},
        )
        data = r.json()
        if data["status"] != "OK":
            return None

        result = data["results"][0]
        location = result["geometry"]["location"]

        town = None
        postal_code = None
        for comp in result.get("address_components", []):
            types = comp.get("types", [])
            if "postal_town" in types or "locality" in types:
                town = comp["long_name"]
            if "postal_code" in types:
                postal_code = comp["long_name"]

        return {
            "formatted_address": result["formatted_address"],
            "lat": location["lat"],
            "lng": location["lng"],
            "town": town,
            "postal_code": postal_code,
        }
    except Exception:
        log.exception("Geocode failed for address: %s", address)
    return None


async def get_distance(origin: str, destination: str) -> tuple[float, int] | None:
    """Return (distance_miles, duration_mins) between two points.

    Uses the Google Maps Directions API (same as test_location.py).
    """
    try:
        r = await _get_client().get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params={
                "origin": origin,
                "destination": destination,
                "units": "imperial",
                "key": settings.GOOGLE_MAPS_API_KEY,
            },
        )
        data = r.json()
        if data["status"] != "OK":
            log.warning("Directions API status: %s for %s -> %s", data["status"], origin, destination)
            return None

        leg = data["routes"][0]["legs"][0]

        distance_meters = leg["distance"]["value"]
        duration_seconds = leg["duration"]["value"]

        distance_miles = round(distance_meters / 1609.344, 2)
        duration_mins = round(duration_seconds / 60)

        return (distance_miles, duration_mins)
    except Exception:
        log.exception("Directions failed: %s -> %s", origin, destination)
    return None


async def calculate_fare(
    pickup: str, dropoff: str, wait_time_mins: int = 0, journey_type: str | None = None
) -> FareEstimate | None:
    """Calculate fare using universal deadhead logic."""
    log.info("Calculating fare: base->%s->%s", pickup, dropoff)

    # A -> B (base to pickup)
    ab = await get_distance(BASE_ORIGIN, pickup)
    if ab is None:
        log.warning("Fare calc failed: base->pickup returned None")
        return None
    base_to_src_miles, base_to_src_mins = ab
    log.info("Base->Pickup: %.2f miles, %d mins", base_to_src_miles, base_to_src_mins)

    # B -> C (pickup to dropoff)
    bc = await get_distance(pickup, dropoff)
    if bc is None:
        log.warning("Fare calc failed: pickup->dropoff returned None")
        return None
    ride_miles, ride_mins = bc
    log.info("Pickup->Dropoff: %.2f miles, %d mins", ride_miles, ride_mins)

    # C -> A (dropoff back to base), except round trips end back at pickup
    if journey_type == "Round Trip":
        dest_to_base_miles = base_to_src_miles
        dest_to_base_mins = base_to_src_mins
    else:
        ca = await get_distance(dropoff, BASE_ORIGIN)
        dest_to_base_miles = ca[0] if ca else 0.0
        dest_to_base_mins = ca[1] if ca else 0

    is_inbound = base_to_src_miles > dest_to_base_miles
    charge_miles = ride_miles + (dest_to_base_miles if is_inbound else base_to_src_miles)
    display_distance_miles = ride_miles
    display_duration_mins = ride_mins

    if journey_type == "Round Trip":
        charge_miles += ride_miles
        display_distance_miles = round(ride_miles * 2, 2)
        display_duration_mins = ride_mins * 2

    if charge_miles <= 2:
        base_fare = 11.0
    else:
        base_fare = 11.0 + ((charge_miles - 2) * 2.40)

    wait_surcharge = 0.0
    if journey_type == "Round Trip":
        wait_surcharge = round((max(wait_time_mins, 0) / 30.0) * 5.0, 2)

    if journey_type == "Return":
        base_fare *= 2
        display_distance_miles = round(ride_miles * 2, 2)
        display_duration_mins = ride_mins * 2

    return FareEstimate(
        distance_miles=ride_miles,
        duration_mins=ride_mins,
        display_distance_miles=display_distance_miles,
        display_duration_mins=display_duration_mins,
        base_fare=round(base_fare, 2),
        wait_surcharge=round(wait_surcharge, 2),
        total_fare=round(base_fare + wait_surcharge, 2),
        base_to_src_miles=round(base_to_src_miles, 2),
        base_to_src_mins=base_to_src_mins,
        dest_to_base_miles=round(dest_to_base_miles, 2),
        dest_to_base_mins=dest_to_base_mins,
    )
