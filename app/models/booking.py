from datetime import date, time
from pydantic import BaseModel


class BookingExtraction(BaseModel):
    """Raw fields extracted by the LLM. All nullable."""
    journey_type: str | None = None          # One Way / Return / Round Trip
    pickup_location: str | None = None
    dropoff_location: str | None = None
    pickup_date: str | None = None           # raw string as spoken
    pickup_time: str | None = None           # raw string as spoken
    return_journey_date: str | None = None
    return_journey_time: str | None = None
    trip_type: str | None = None             # Airport / Local Trip
    flight_number: str | None = None
    flight_journey_type: str | None = None   # Arrival / Departure
    wait_time_mins: int | None = None
    number_of_luggages: int | None = None
    booking_notes: str | None = None


class BookingCreate(BaseModel):
    """Validated booking ready for DB insert."""
    pickup_date: date
    pickup_time: time
    pickup_location: str
    dropoff_location: str
    dropoff_location_type: str | None = None  # Airport / Local Trip
    customer_id: int
    customer_phone: str
    booking_notes: str | None = None
    est_ride_duration_mins: int | None = None
    est_ride_distance_miles: float | None = None
    est_ride_fare: float | None = None
    currency: str = "GBP"
    pickup_note: str | None = None
    dropoff_note: str | None = None
    wait_time_mins: int | None = None
    base_location: str = "Home"
    base_to_src_distance_miles: float | None = None
    dest_to_base_distance_miles: float | None = None
    est_base_to_src_duration_mins: int | None = None
    est_dest_to_base_duration_mins: int | None = None
    journey_type: str | None = None          # One Way / Return / Round Trip
    flight_number: str | None = None
    flight_journey_type: str | None = None   # Arrival / Departure
    number_of_luggages: int | None = None
    booking_type: str = "Regular"


class FareEstimate(BaseModel):
    distance_miles: float
    duration_mins: int
    display_distance_miles: float | None = None
    display_duration_mins: int | None = None
    base_fare: float
    wait_surcharge: float
    total_fare: float
    base_to_src_miles: float
    base_to_src_mins: int
    dest_to_base_miles: float
    dest_to_base_mins: int
