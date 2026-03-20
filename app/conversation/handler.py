import json
import logging
import os
import re
from datetime import datetime, timedelta

import pytz
from dateutil import parser as dateutil_parser

from app.conversation.session_store import get_session
from app.models.booking import BookingCreate, BookingExtraction
from app.models.customer import CustomerCreate
from app.models.session import ConversationSession, ConversationState
from app.services import (
    booking_service,
    customer_service,
    llm_service,
    maps_service,
    messaging_service,
    transcription_service,
)

log = logging.getLogger(__name__)

# ──────────────────────── Conversation Logger ────────────────────────

CONVO_LOG_DIR = "logs"
os.makedirs(CONVO_LOG_DIR, exist_ok=True)


def _log_conversation(phone: str, direction: str, message: str):
    """Append a line to the per-user conversation log file."""
    safe_phone = phone.replace("whatsapp:", "").replace("+", "")
    path = os.path.join(CONVO_LOG_DIR, f"convo_{safe_phone}.jsonl")
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "dir": direction,
        "msg": message,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ──────────────────────── Constants ────────────────────────

# Core required fields - must be filled before anything else
CORE_FIELDS = ["pickup_location", "dropoff_location", "pickup_date", "pickup_time"]

# Always asked after core fields
FOLLOWUP_FIELDS = ["journey_type"]

# Asked only for Airport trips (flight_journey_type is auto-detected, not asked)
AIRPORT_FIELDS = ["flight_number", "number_of_luggages"]

# Asked only for Round Trip
ROUND_TRIP_FIELDS = ["wait_time_mins"]

FIELD_QUESTIONS = {
    "pickup_location": "Where should I pick you up from?",
    "dropoff_location": "Where would you like to go?",
    "pickup_date": "What date would you like to travel?",
    "pickup_time": "What time would you like to be picked up?",
    "journey_type": "Is this a One Way, Return, or Round Trip?",
    "flight_number": "What is your flight number?",
    "number_of_luggages": "How many pieces of luggage will you have?",
    "wait_time_mins": "How long (in minutes) would you like the driver to wait?",
}

# Fields where direct text assignment is safe (user is answering a specific question)
DIRECT_ASSIGN_FIELDS = {
    "pickup_date", "pickup_time", "journey_type",
    "flight_number", "number_of_luggages", "wait_time_mins",
}

# Airport-related keywords for auto-detecting trip_type and flight_journey_type
_AIRPORT_KEYWORDS = {"airport", "heathrow", "gatwick", "stansted", "luton", "lhr", "lgw", "stn"}


async def _send(to: str, body: str):
    """Send message and log it."""
    _log_conversation(to, "bot", body)
    await messaging_service.send(to, body)


async def process_message(data: dict):
    """Main conversation orchestrator. Runs as a background task."""
    from_number = data.get("From", "")
    body = data.get("Body", "").strip()
    num_media = int(data.get("NumMedia", 0))
    latitude = data.get("Latitude")
    longitude = data.get("Longitude")

    # Log incoming message
    if latitude and longitude:
        _log_conversation(from_number, "user", f"[Location: {latitude},{longitude}]")
    elif body:
        _log_conversation(from_number, "user", body)
    elif num_media > 0:
        _log_conversation(from_number, "user", "[Voice/Media message]")

    session = get_session(from_number)
    log.info("State=%s | Phone=%s | Body=%s", session.state.value, from_number, body[:80] if body else "(none)")

    try:
        await _route(session, from_number, body, num_media, data, latitude, longitude)
    except Exception:
        log.exception("Error processing message from %s", from_number)
        await _send(
            from_number,
            "Sorry, something went wrong. Please try again.",
        )


async def _route(
    session: ConversationSession,
    from_number: str,
    body: str,
    num_media: int,
    data: dict,
    latitude: str | None,
    longitude: str | None,
):
    state = session.state

    if state == ConversationState.IDENTIFY:
        await _handle_identify(session, from_number, body, num_media, data, latitude, longitude)

    elif state == ConversationState.ONBOARD_NAME:
        await _handle_onboard_name(session, from_number, body)

    elif state == ConversationState.ONBOARD_EMAIL:
        await _handle_onboard_email(session, from_number, body)

    elif state == ConversationState.ONBOARD_ADDRESS:
        await _handle_onboard_address(session, from_number, body, latitude, longitude)

    elif state == ConversationState.COLLECTING:
        await _handle_collecting(
            session, from_number, body, num_media, data, latitude, longitude
        )

    elif state == ConversationState.AWAITING_CONFIRMATION:
        await _handle_confirmation(session, from_number, body)

    elif state == ConversationState.EDITING:
        await _handle_editing(session, from_number, body)

    elif state == ConversationState.POST_BOOKING:
        await _handle_post_booking(session, from_number, body)


# ──────────────────────── Phase A: Identification ────────────────────────


async def _handle_identify(
    session: ConversationSession,
    from_number: str,
    body: str,
    num_media: int,
    data: dict,
    latitude: str | None,
    longitude: str | None,
):
    customer = await customer_service.lookup_by_phone(from_number)
    if not customer:
        session.state = ConversationState.ONBOARD_NAME
        await _send(
            from_number,
            "Welcome to our taxi booking service! I don't have your details on file.\n\n"
            "May I have your full name please?",
        )
        return

    session.customer_id = customer.customer_id
    session.customer_name = customer.full_name

    # Check for upcoming bookings
    upcoming = await booking_service.get_upcoming_bookings(customer.customer_id)
    bookings_info = ""
    if upcoming:
        session.last_booking_id = upcoming[-1]["booking_id"]  # most recent by ID
        bookings_info = "\nYour upcoming bookings:\n"
        for b in upcoming:
            bookings_info += (
                f"  #{b['booking_id']}: {b['pickup_location']} -> {b['dropoff_location']}\n"
                f"    {b['pickup_date']} at {b['pickup_time']}\n"
            )
        bookings_info += (
            "\nYou can ask about driver status, or reply with a booking number "
            "to make changes (e.g. \"edit #2290\").\n"
        )

    # Check if the message has actual booking content (not just a greeting)
    has_content = (
        (latitude and longitude)
        or (num_media > 0)
        or (body and body.lower().strip() not in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"})
    )

    if has_content:
        if latitude and longitude or num_media > 0:
            await _start_new_booking_from_identify(
                session, from_number, customer.full_name, bookings_info, body, num_media, data, latitude, longitude
            )
            return

        route = await llm_service.classify_intent(body)
        intent = route.get("intent")
        booking_id = route.get("booking_id")

        if intent == "CHECK_STATUS":
            await _handle_identify_status(session, from_number, bookings_info, booking_id)
            return

        if intent == "EDIT_BOOKING":
            await _handle_identify_edit(session, from_number, booking_id)
            return

        if intent == "GENERAL_CHAT":
            reply = await llm_service.generate_conversational_reply(body)
            await _send(from_number, reply)
            return

        await _start_new_booking_from_identify(
            session, from_number, customer.full_name, bookings_info, body, num_media, data, latitude, longitude
        )
    else:
        # Just a greeting - show full welcome with options
        greeting = f"Welcome back, {customer.full_name}!"
        if bookings_info:
            greeting += bookings_info
        greeting += (
            "\nHow can I help you today?\n\n"
            "You can book a new ride via text, voice, or live location."
        )
        await _send(from_number, greeting)


async def _start_new_booking_from_identify(
    session: ConversationSession,
    from_number: str,
    customer_name: str,
    bookings_info: str,
    body: str,
    num_media: int,
    data: dict,
    latitude: str | None,
    longitude: str | None,
):
    session.state = ConversationState.COLLECTING
    greeting = f"Welcome back, {customer_name}!"
    if bookings_info:
        greeting += bookings_info + "\nProcessing your new request..."
    await _send(from_number, greeting)
    await _handle_collecting(
        session, from_number, body, num_media, data, latitude, longitude
    )


async def _handle_identify_status(
    session: ConversationSession, from_number: str, bookings_info: str, booking_id: int | None = None
):
    if not session.customer_id:
        await _send(from_number, "I couldn't find your customer profile yet.")
        return

    if booking_id is not None:
        item = await booking_service.get_booking_status_for_customer(session.customer_id, booking_id)
        if not item:
            await _send(from_number, f"I couldn't find booking #{booking_id} on your account.")
            return
        session.last_booking_id = booking_id
        await _send(from_number, _format_status_message(item))
        return

    statuses = await booking_service.get_active_booking_statuses(session.customer_id)
    if not statuses:
        message = "I couldn't find any active upcoming bookings for you."
        if bookings_info:
            message += bookings_info
        await _send(from_number, message)
        return

    session.last_booking_id = statuses[0]["booking_id"]
    lines = ["Here is the latest status for your active bookings:\n"]
    for item in statuses:
        lines.append(_format_status_message(item))
        lines.append("")
    await _send(from_number, "\n".join(lines).strip())


async def _handle_identify_edit(
    session: ConversationSession, from_number: str, booking_id: int | None
):
    if not session.customer_id:
        await _send(from_number, "I couldn't find your customer profile yet.")
        return

    if booking_id is None:
        await _send(
            from_number,
            "I can help route an edit request, but I need the booking number. Please reply with something like edit #2290.",
        )
        return

    booking = await booking_service.get_customer_booking(session.customer_id, booking_id)
    if not booking:
        await _send(from_number, f"I couldn't find booking #{booking_id} on your account.")
        return

    if await booking_service.has_assignment(booking_id):
        await _send(
            from_number,
            f"Booking #{booking_id} already has a driver assigned, so I can't edit it automatically. Please contact support for assistance.",
        )
        return

    session.last_booking_id = booking_id
    session.editing_booking_id = booking_id
    session.booking = _booking_row_to_extraction(booking)
    session.state = ConversationState.EDITING
    await _send(
        from_number,
        f"I see you want to edit booking #{booking_id}.\n"
        "What would you like to change? You can reply naturally, for example: change pickup to Oxford and time to 3:30 PM.",
    )


def _format_vehicle(item: dict) -> str:
    parts = [item.get("color"), item.get("make"), item.get("model")]
    vehicle = " ".join(part for part in parts if part)
    plate = item.get("plate_number")
    if vehicle and plate:
        return f"{vehicle} ({plate})"
    if vehicle:
        return vehicle
    if plate:
        return plate
    return "Not assigned yet"


def _format_status_message(item: dict) -> str:
    ride_status = item.get("ride_status") or "Awaiting driver assignment"
    eta = "ETA unavailable"
    lines = [
        f"Booking #{item['booking_id']}: {item['pickup_location']} -> {item['dropoff_location']}",
        f"Date/Time: {item['pickup_date']} at {item['pickup_time']}",
        f"Status: {ride_status}",
        f"ETA: {eta}",
        f"Vehicle: {_format_vehicle(item)}",
    ]
    if item.get("driver_name"):
        lines.append(f"Driver: {item['driver_name']}")
        lines.append(f"Driver Phone: {item.get('driver_phone') or 'phone unavailable'}")
    return "\n".join(lines)


def _booking_row_to_extraction(row: dict) -> BookingExtraction:
    passengers = None
    notes = row.get("booking_notes") or ""
    match = re.search(r"Passengers:\s*(\d+)", notes, re.IGNORECASE)
    if match:
        passengers = int(match.group(1))

    pickup_date = row.get("pickup_date")
    pickup_time = row.get("pickup_time")
    return BookingExtraction(
        journey_type=row.get("journey_type"),
        pickup_location=row.get("pickup_location"),
        dropoff_location=row.get("dropoff_location"),
        pickup_date=pickup_date.isoformat() if pickup_date else None,
        pickup_time=pickup_time.strftime("%H:%M") if pickup_time else None,
        trip_type=row.get("dropoff_location_type"),
        flight_number=row.get("flight_number"),
        flight_journey_type=row.get("flight_journey_type"),
        wait_time_mins=row.get("wait_time_mins"),
        number_of_luggages=row.get("number_of_luggages"),
        passengers=passengers,
    )


async def _handle_onboard_name(
    session: ConversationSession, from_number: str, body: str
):
    if len(body) < 2 or body.lower() in ("hi", "hello", "hey"):
        await _send(from_number, "Please enter your full name.")
        return

    session.customer_name = body.strip()
    session.state = ConversationState.ONBOARD_EMAIL
    await _send(
        from_number,
        f"Thanks, {session.customer_name}! What is your email address?",
    )


async def _handle_onboard_email(
    session: ConversationSession, from_number: str, body: str
):
    if "@" not in body or "." not in body:
        await _send(
            from_number, "That doesn't look like a valid email. Please try again."
        )
        return

    session.onboard_email = body.strip()
    session.state = ConversationState.ONBOARD_ADDRESS
    await _send(
        from_number,
        "What is your home address? You can type it or share your live location.\n\n"
        "If typing, please include street, town, and postcode if possible.",
    )


async def _handle_onboard_address(
    session: ConversationSession,
    from_number: str,
    body: str,
    latitude: str | None = None,
    longitude: str | None = None,
):
    # If user shared a live location, reverse geocode it
    if latitude and longitude:
        address = await maps_service.reverse_geocode(float(latitude), float(longitude))
        if address:
            geo = await maps_service.geocode_address(address)
        else:
            geo = None
    else:
        geo = await maps_service.geocode_address(body)

    if geo is None:
        await _send(
            from_number,
            "I couldn't validate that address. Please try again with a more complete address.",
        )
        return

    phone = customer_service.normalize_phone(from_number)
    new_customer = CustomerCreate(
        full_name=session.customer_name or "",
        primary_phone_no=phone,
        email=session.onboard_email,
        address=geo["formatted_address"],
        town=geo.get("town"),
        postal_code=geo.get("postal_code"),
    )
    customer_id = await customer_service.create_customer(new_customer)
    session.customer_id = customer_id
    session.state = ConversationState.COLLECTING

    await _send(
        from_number,
        f"You're all set, {session.customer_name}!\n\n"
        f"Address saved: {geo['formatted_address']}\n\n"
        "Now, how can I help you? Send a text or voice message describing your trip, "
        "or share a live location for pickup.",
    )


# ──────────────────────── Phase B: Collecting ────────────────────────


async def _handle_collecting(
    session: ConversationSession,
    from_number: str,
    body: str,
    num_media: int,
    data: dict,
    latitude: str | None,
    longitude: str | None,
):
    booking = session.booking

    # Live location shared
    if latitude and longitude:
        address = await maps_service.reverse_geocode(float(latitude), float(longitude))
        if address:
            booking.pickup_location = address
            await _ask_next_or_proceed(session, from_number,
                                       prefix=f"Pickup location set:\n{address}")
        else:
            await _send(
                from_number,
                "I couldn't resolve that location. Please try sharing again or type the address.",
            )
        return

    # Voice message
    if num_media > 0:
        media_type = data.get("MediaContentType0", "")
        media_url = data.get("MediaUrl0", "")
        if media_url and (not media_type or "audio" in media_type or "ogg" in media_type):
            try:
                audio_path = await transcription_service.download_media(media_url)
                text = await transcription_service.transcribe_audio(audio_path)
                await _extract_and_merge(session, from_number, text)
            except Exception:
                log.exception("Voice processing failed")
                await _send(
                    from_number, "Sorry, I couldn't process that voice message. Please try again."
                )
            return

    # Text message
    if body:
        next_field = _get_next_missing_field(session.booking)

        # If answering a specific direct-assign question, assign directly
        if next_field and next_field in DIRECT_ASSIGN_FIELDS:
            # Handle numeric fields
            if next_field in ("number_of_luggages", "wait_time_mins"):
                try:
                    val = int(body.strip())
                    setattr(session.booking, next_field, val)
                except ValueError:
                    await _send(from_number, "Please enter a number.")
                    return
            else:
                cleaned = _sanitize_field_value(next_field, body.strip())
                if cleaned is None and next_field == "journey_type":
                    await _send(from_number, "Please reply with One Way, Return, or Round Trip.")
                    return
                if cleaned is None and next_field == "pickup_time":
                    await _send(from_number, "Did you mean 3:00 AM or 3:00 PM? Please include a time like 3:00 PM.")
                    return
                setattr(session.booking, next_field, cleaned)

            log.info("Direct assign: %s = %s", next_field, getattr(session.booking, next_field))

            # After setting journey_type, auto-derive dependent fields
            if next_field == "journey_type":
                _auto_set_trip_metadata(session.booking)

            await _ask_next_or_proceed(session, from_number)
            return

        # If only one core field is missing, assign directly
        if next_field and next_field in CORE_FIELDS and _count_missing_core(session.booking) == 1:
            cleaned = _sanitize_field_value(next_field, body.strip())
            if cleaned is None and next_field == "pickup_time":
                await _send(from_number, "Did you mean 3:00 AM or 3:00 PM? Please include a time like 3:00 PM.")
                return
            setattr(session.booking, next_field, cleaned)
            log.info("Direct assign (last core): %s = %s", next_field, cleaned)
            await _ask_next_or_proceed(session, from_number)
            return

        # For multi-field extraction, use the LLM
        await _extract_and_merge(session, from_number, body)


async def _extract_and_merge(
    session: ConversationSession, from_number: str, text: str
):
    """Extract booking info from text, merge into session, and respond."""
    # Build context so LLM knows which fields are already filled
    context_parts = []
    b = session.booking
    if b.pickup_location:
        context_parts.append(f"Pickup is already set to: {b.pickup_location}")
    if b.dropoff_location:
        context_parts.append(f"Dropoff is already set to: {b.dropoff_location}")
    if b.pickup_date:
        context_parts.append(f"Date is already set to: {b.pickup_date}")
    if b.pickup_time:
        context_parts.append(f"Time is already set to: {b.pickup_time}")

    if context_parts:
        augmented = "Context: " + "; ".join(context_parts) + "\nUser message: " + text
    else:
        augmented = text

    extraction = await llm_service.extract_booking(augmented)
    _sanitize_extraction(extraction)
    _merge(session.booking, extraction)

    # Auto-set trip metadata after LLM extraction
    _auto_set_trip_metadata(session.booking)

    await _ask_next_or_proceed(session, from_number)


def _auto_set_trip_metadata(booking: BookingExtraction):
    """Auto-detect trip_type and flight_journey_type from locations.

    - If any location contains airport keywords -> trip_type = "Airport"
    - If dropoff is airport -> flight_journey_type = "Departure"
    - If pickup is airport -> flight_journey_type = "Arrival"
    - wait_time_mins defaults to 0 for One Way and Return
    """
    pickup = (booking.pickup_location or "").lower()
    dropoff = (booking.dropoff_location or "").lower()

    pickup_is_airport = any(kw in pickup for kw in _AIRPORT_KEYWORDS)
    dropoff_is_airport = any(kw in dropoff for kw in _AIRPORT_KEYWORDS)

    # Auto-set trip_type
    if pickup_is_airport or dropoff_is_airport:
        booking.trip_type = "Airport"
    elif booking.pickup_location and booking.dropoff_location and not booking.trip_type:
        booking.trip_type = "Local Trip"

    # Auto-set flight_journey_type (never ask the user)
    if booking.trip_type == "Airport":
        if dropoff_is_airport:
            booking.flight_journey_type = "Departure"
        elif pickup_is_airport:
            booking.flight_journey_type = "Arrival"

    # Default wait_time_mins to 0 for One Way and Return
    if booking.journey_type and booking.journey_type.lower() != "round trip":
        if booking.wait_time_mins is None:
            booking.wait_time_mins = 0


def _sanitize_extraction(extraction: BookingExtraction):
    extraction.journey_type = _sanitize_journey_type(extraction.journey_type)
    extraction.pickup_time = _sanitize_time_value(extraction.pickup_time)


def _sanitize_field_value(field: str, value: str):
    if field == "journey_type":
        return _sanitize_journey_type(value)
    if field == "pickup_time":
        return _sanitize_time_value(value)
    return value


def _sanitize_journey_type(value: str | None) -> str | None:
    if not value:
        return None

    normalized = value.strip().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).title()
    mapping = {
        "One Way": "One Way",
        "Oneway": "One Way",
        "Return": "Return",
        "Round Trip": "Round Trip",
        "Roundtrip": "Round Trip",
    }
    return mapping.get(normalized)


def _sanitize_time_value(value: str | None) -> str | None:
    if not value:
        return None

    stripped = value.strip()
    has_colon = ":" in stripped
    has_ampm = bool(re.search(r"\b(?:am|pm)\b", stripped, re.IGNORECASE))
    if not has_colon and not has_ampm:
        return None
    return stripped


async def _ask_next_or_proceed(
    session: ConversationSession, from_number: str, prefix: str = ""
):
    """Ask the next missing field, or proceed to fare calculation."""
    next_field = _get_next_missing_field(session.booking)
    if next_field:
        question = FIELD_QUESTIONS[next_field]
        if prefix:
            await _send(from_number, f"{prefix}\n\n{question}")
        else:
            await _send(from_number, question)
    else:
        await _try_fare_calculation(session, from_number)


async def _try_fare_calculation(session: ConversationSession, from_number: str):
    """Geocode locations, calculate fare, and show summary."""
    await _send(from_number, "Let me validate your locations and calculate the fare...")

    b = session.booking

    # Geocode pickup to get full address + coordinates
    pickup_geo = await maps_service.geocode_address(b.pickup_location or "")
    if pickup_geo:
        b.pickup_location = pickup_geo["formatted_address"]
        session.pickup_coords = f"{pickup_geo['lat']},{pickup_geo['lng']}"
        log.info("Geocoded pickup: %s (%s)", b.pickup_location, session.pickup_coords)
    else:
        log.warning("Could not geocode pickup: %s", b.pickup_location)
        session.pickup_coords = None

    # Geocode dropoff to get full address + coordinates
    dropoff_geo = await maps_service.geocode_address(b.dropoff_location or "")
    if dropoff_geo:
        b.dropoff_location = dropoff_geo["formatted_address"]
        session.dropoff_coords = f"{dropoff_geo['lat']},{dropoff_geo['lng']}"
        log.info("Geocoded dropoff: %s (%s)", b.dropoff_location, session.dropoff_coords)
    else:
        log.warning("Could not geocode dropoff: %s", b.dropoff_location)
        session.dropoff_coords = None

    # Re-run auto-detect on geocoded addresses (they may now contain "Airport")
    _auto_set_trip_metadata(b)

    # Use coordinates for fare calculation (more reliable than text addresses)
    pickup_for_fare = session.pickup_coords or b.pickup_location or ""
    dropoff_for_fare = session.dropoff_coords or b.dropoff_location or ""

    fare_summary = await _calculate_and_format_fare(session, pickup_for_fare, dropoff_for_fare)
    if fare_summary:
        session.state = ConversationState.AWAITING_CONFIRMATION
        await _send(from_number, fare_summary)
    else:
        # Fare calculation failed - still proceed but without fare
        log.warning("Fare calculation failed, proceeding without fare")
        session.state = ConversationState.AWAITING_CONFIRMATION
        await _send(from_number, _format_summary_without_fare(session))


def _format_summary_without_fare(session: ConversationSession) -> str:
    b = session.booking
    resolved_date = _normalize_date(b.pickup_date or "")
    lines = [
        "Here's your booking summary:\n",
        f"Pickup: {b.pickup_location}",
        f"Dropoff: {b.dropoff_location}",
        f"Date: {resolved_date or b.pickup_date}",
        f"Time: {b.pickup_time}",
        "\n(Fare could not be calculated for this route)",
        f"Journey: {b.journey_type}",
    ]
    if b.trip_type:
        lines.append(f"Trip Type: {b.trip_type}")
    if b.flight_number:
        lines.append(f"Flight: {b.flight_number} ({b.flight_journey_type or 'N/A'})")
    if b.number_of_luggages is not None:
        lines.append(f"Luggage: {b.number_of_luggages}")
    if b.wait_time_mins and b.wait_time_mins > 0:
        lines.append(f"Wait Time: {b.wait_time_mins} mins")
    if b.passengers:
        lines.append(f"Passengers: {b.passengers}")
    lines.append("\nReply YES to confirm or NO to make changes.")
    return "\n".join(lines)


# ──────────────────────── Field Logic ────────────────────────


def _get_next_missing_field(booking: BookingExtraction) -> str | None:
    """Determine the next field to ask about, in priority order:
    1. Core fields (pickup, dropoff, date, time)
    2. Journey type (always required)
    3. Airport-specific fields: flight_number, number_of_luggages
       (flight_journey_type is auto-detected from locations, never asked)
    4. Round Trip: wait_time_mins
    """
    # Core fields first
    for f in CORE_FIELDS:
        if getattr(booking, f) is None:
            return f

    # Journey type
    if booking.journey_type is None:
        return "journey_type"

    # Airport-specific fields
    if booking.trip_type and booking.trip_type.lower() == "airport":
        for f in AIRPORT_FIELDS:
            if getattr(booking, f) is None:
                return f

    # Round Trip: ask for wait time
    if booking.journey_type and booking.journey_type.lower() == "round trip":
        if booking.wait_time_mins is None:
            return "wait_time_mins"

    return None


def _merge(existing: BookingExtraction, new: BookingExtraction):
    """Merge new extraction into existing, only filling None fields."""
    for field in existing.model_fields:
        if getattr(existing, field) is None and getattr(new, field) is not None:
            setattr(existing, field, getattr(new, field))


def _count_missing_core(booking: BookingExtraction) -> int:
    return sum(1 for f in CORE_FIELDS if getattr(booking, f) is None)


# ──────────────────────── Fare Calculation ────────────────────────


async def _calculate_and_format_fare(
    session: ConversationSession, pickup_for_fare: str, dropoff_for_fare: str
) -> str | None:
    b = session.booking
    fare = await maps_service.calculate_fare(
        pickup_for_fare,
        dropoff_for_fare,
        b.wait_time_mins or 0,
        b.journey_type,
    )
    if fare is None:
        return None

    session.fare = fare

    resolved_date = _normalize_date(b.pickup_date or "")
    lines = [
        "Here's your booking summary:\n",
        f"Pickup: {b.pickup_location}",
        f"Dropoff: {b.dropoff_location}",
        f"Date: {resolved_date or b.pickup_date}",
        f"Time: {b.pickup_time}",
        f"Distance: {fare.distance_miles} miles",
        f"Est. Duration: {fare.duration_mins} mins",
        f"Fare: \u00a3{fare.total_fare:.2f}",
    ]
    if fare.wait_surcharge > 0:
        lines.append(f"  (includes \u00a3{fare.wait_surcharge:.2f} wait surcharge)")
    lines.append(f"Journey: {b.journey_type}")
    if b.trip_type:
        lines.append(f"Trip Type: {b.trip_type}")
    if b.flight_number:
        lines.append(f"Flight: {b.flight_number} ({b.flight_journey_type or 'N/A'})")
    if b.number_of_luggages is not None:
        lines.append(f"Luggage: {b.number_of_luggages}")
    if b.wait_time_mins and b.wait_time_mins > 0:
        lines.append(f"Wait Time: {b.wait_time_mins} mins")
    if b.passengers:
        lines.append(f"Passengers: {b.passengers}")

    lines.append("\nReply YES to confirm or NO to make changes.")
    return "\n".join(lines)


# ──────────────────────── Phase D: Confirmation ────────────────────────


async def _handle_confirmation(
    session: ConversationSession, from_number: str, body: str
):
    lower = body.lower().strip()

    if lower == "yes":
        booking_id = await _insert_booking(session, from_number)
        if booking_id:
            session.last_booking_id = booking_id
            session.state = ConversationState.POST_BOOKING
            await _send(
                from_number,
                f"Your booking is confirmed! (Booking #{booking_id})\n\n"
                "You'll receive a notification once a driver is assigned.\n"
                "You can ask me about your driver anytime, or book another ride.",
            )
        else:
            await _send(
                from_number,
                "Sorry, there was an error saving your booking. Please try again.",
            )

    elif lower == "no":
        session.editing_booking_id = None
        session.state = ConversationState.EDITING
        await _send(
            from_number,
            "No problem! What would you like to change?\n"
            "(pickup / dropoff / date / time / journey)",
        )

    else:
        await _send(
            from_number, "Please reply YES to confirm or NO to make changes."
        )


async def _insert_booking(
    session: ConversationSession, from_number: str
) -> int | None:
    b = session.booking
    fare = session.fare

    resolved_date = _normalize_date(b.pickup_date or "")
    resolved_time = _normalize_time(b.pickup_time or "")

    if not resolved_date or not resolved_time:
        log.error("Could not resolve date/time: %s / %s", b.pickup_date, b.pickup_time)
        return None

    notes = f"Passengers: {b.passengers}" if b.passengers else None

    phone = customer_service.normalize_phone(from_number)
    create = BookingCreate(
        pickup_date=resolved_date,
        pickup_time=resolved_time,
        pickup_location=b.pickup_location or "",
        dropoff_location=b.dropoff_location or "",
        dropoff_location_type=b.trip_type,
        customer_id=session.customer_id or 0,
        customer_phone=phone,
        booking_notes=notes,
        est_ride_duration_mins=fare.duration_mins if fare else None,
        est_ride_distance_miles=fare.distance_miles if fare else None,
        est_ride_fare=fare.total_fare if fare else None,
        currency="GBP",
        wait_time_mins=b.wait_time_mins or 0,
        base_location="Home",
        base_to_src_distance_miles=fare.base_to_src_miles if fare else None,
        dest_to_base_distance_miles=fare.dest_to_base_miles if fare else None,
        est_base_to_src_duration_mins=fare.base_to_src_mins if fare else None,
        est_dest_to_base_duration_mins=fare.dest_to_base_mins if fare else None,
        journey_type=b.journey_type,
        flight_number=b.flight_number,
        flight_journey_type=b.flight_journey_type,
        number_of_luggages=b.number_of_luggages,
        booking_type="Regular",
    )

    try:
        return await booking_service.insert_booking(create)
    except Exception:
        log.exception("Failed to insert booking")
        return None


# ──────────────────────── Phase: Editing ────────────────────────


EDIT_FIELD_MAP = {
    "pickup": "pickup_location",
    "dropoff": "dropoff_location",
    "drop": "dropoff_location",
    "date": "pickup_date",
    "time": "pickup_time",
    "journey": "journey_type",
}


async def _handle_editing(
    session: ConversationSession, from_number: str, body: str
):
    if session.editing_booking_id:
        await _handle_existing_booking_edit(session, from_number, body)
        return

    field_key = body.lower().strip()
    mapped = EDIT_FIELD_MAP.get(field_key)

    if mapped:
        setattr(session.booking, mapped, None)
        session.fare = None
        session.state = ConversationState.COLLECTING
        await _send(from_number, FIELD_QUESTIONS[mapped])
    else:
        await _send(
            from_number,
            "Please choose one of these to change:\n"
            "pickup / dropoff / date / time / journey",
        )


async def _handle_existing_booking_edit(
    session: ConversationSession, from_number: str, body: str
):
    booking_id = session.editing_booking_id
    if not booking_id:
        await _send(from_number, "I couldn't determine which booking you want to edit.")
        return

    if await booking_service.has_assignment(booking_id):
        session.editing_booking_id = None
        session.state = ConversationState.IDENTIFY
        await _send(
            from_number,
            f"Booking #{booking_id} has now been assigned to a driver, so automatic edits are blocked. Please contact support for assistance.",
        )
        return

    extraction = await llm_service.extract_booking(body)
    _sanitize_extraction(extraction)
    if not _has_updates(extraction):
        await _send(
            from_number,
            "I couldn't detect the changes yet. Please describe the new pickup, dropoff, date, time, or journey type.",
        )
        return

    _apply_booking_updates(session.booking, extraction)
    _auto_set_trip_metadata(session.booking)

    resolved_date = _normalize_date(session.booking.pickup_date or "")
    resolved_time = _normalize_time(session.booking.pickup_time or "")
    if not resolved_date or not resolved_time:
        await _send(
            from_number,
            "I still need a clear date and time before I can update this booking. Please include both, for example tomorrow at 3:30 PM.",
        )
        return

    notes = f"Passengers: {session.booking.passengers}" if session.booking.passengers else None
    phone = customer_service.normalize_phone(from_number)
    updated_id = await booking_service.update_booking(
        booking_id,
        phone,
        pickup_date=resolved_date,
        pickup_time=resolved_time,
        pickup_location=session.booking.pickup_location or "",
        dropoff_location=session.booking.dropoff_location or "",
        dropoff_location_type=session.booking.trip_type,
        booking_notes=notes,
        wait_time_mins=session.booking.wait_time_mins,
        journey_type=session.booking.journey_type,
        flight_number=session.booking.flight_number,
        flight_journey_type=session.booking.flight_journey_type,
        number_of_luggages=session.booking.number_of_luggages,
    )
    if not updated_id:
        await _send(
            from_number,
            f"I couldn't update booking #{booking_id}. Please try again or contact support.",
        )
        return

    session.last_booking_id = updated_id
    session.editing_booking_id = None
    session.state = ConversationState.POST_BOOKING
    await _send(
        from_number,
        f"Booking #{updated_id} has been updated.\n"
        f"Pickup: {session.booking.pickup_location}\n"
        f"Dropoff: {session.booking.dropoff_location}\n"
        f"Date: {session.booking.pickup_date}\n"
        f"Time: {session.booking.pickup_time}",
    )


def _has_updates(extraction: BookingExtraction) -> bool:
    return any(getattr(extraction, field) is not None for field in extraction.model_fields)


def _apply_booking_updates(existing: BookingExtraction, updates: BookingExtraction):
    for field in existing.model_fields:
        new_value = getattr(updates, field)
        if new_value is not None:
            setattr(existing, field, new_value)


# ──────────────────────── Phase F: Post-Booking ────────────────────────


async def _handle_post_booking(
    session: ConversationSession, from_number: str, body: str
):
    intent = llm_service.detect_intent(body)

    if intent == "driver_query":
        bid = session.last_booking_id
        if not bid and session.customer_id:
            bid = await booking_service.get_latest_booking_id(session.customer_id)

        if bid:
            details = await booking_service.get_assignment_details(bid)
            if details:
                await _send(
                    from_number,
                    f"Your driver is {details['driver_name']}.\n"
                    f"Vehicle: {details['color']} {details['make']} {details['model']}\n"
                    f"Plate: {details['plate_number']}\n"
                    f"Driver Phone: {details['driver_phone']}",
                )
            else:
                await _send(
                    from_number,
                    "A driver hasn't been assigned yet. You'll get a notification once one is assigned.",
                )
        else:
            await _send(
                from_number, "I don't have a recent booking on file for you."
            )

    elif intent == "greeting":
        await _send(
            from_number,
            f"Hello {session.customer_name or ''}! Would you like to book another ride?\n"
            "Send a text or voice message, or share your live location.",
        )

    else:
        # Treat as new booking request
        session.reset_booking()
        await _extract_and_merge(session, from_number, body)


# ──────────────────────── Date/Time Helpers ────────────────────────


def _normalize_date(date_str: str, timezone: str = "Europe/London"):
    """Resolve relative/natural dates to a date object."""
    if not date_str:
        return None

    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    lower = date_str.lower().strip()

    if lower == "today":
        return now.date()
    if lower == "tomorrow":
        return (now + timedelta(days=1)).date()
    if lower == "day after tomorrow":
        return (now + timedelta(days=2)).date()

    try:
        parsed = dateutil_parser.parse(date_str, fuzzy=True, default=now)
        return parsed.date()
    except Exception:
        log.warning("Could not parse date: %s", date_str)
        return None


def _normalize_time(time_str: str):
    """Parse a time string into a time object."""
    if not time_str:
        return None
    try:
        parsed = dateutil_parser.parse(time_str, fuzzy=True)
        return parsed.time()
    except Exception:
        log.warning("Could not parse time: %s", time_str)
        return None
