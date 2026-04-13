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


_AGENT_HISTORY: dict[str, list[dict]] = {}
MAX_AGENT_HISTORY = 12


# ──────────────────────── Constants ────────────────────────

# Core required fields - must be filled before anything else
CORE_FIELDS = ["pickup_location", "dropoff_location", "pickup_date", "pickup_time"]

# Always asked after core fields
FOLLOWUP_FIELDS = ["journey_type"]

# Asked only for Airport trips (flight_journey_type is auto-detected, not asked)
AIRPORT_FIELDS = ["flight_number", "number_of_luggages"]

# Asked only for Round Trip
ROUND_TRIP_FIELDS = ["wait_time_mins"]
RETURN_FIELDS = ["return_journey_date", "return_journey_time"]

FIELD_QUESTIONS = {
    "pickup_location": "Where should I pick you up from?",
    "dropoff_location": "Where would you like to go?",
    "pickup_date": "What date would you like to travel?",
    "pickup_time": "What time would you like to be picked up?",
    "journey_type": "Is this a One Way, Return, or Round Trip?",
    "flight_number": "What is your flight number?",
    "number_of_luggages": "How many pieces of luggage will you have?",
    "wait_time_mins": "How long (in minutes) would you like the driver to wait?",
    "return_journey_date": "What date would you like to return?",
    "return_journey_time": "What time would you like to return?",
    "booking_notes": "Do you have any special requests or notes for the driver?",
}

# Fields where direct text assignment is safe (user is answering a specific question)
DIRECT_ASSIGN_FIELDS = {
    "pickup_date", "pickup_time", "journey_type",
    "flight_number", "number_of_luggages", "wait_time_mins",
    "return_journey_date", "return_journey_time", "booking_notes",
}

# Airport-related keywords for auto-detecting trip_type and flight_journey_type
_AIRPORT_KEYWORDS = {"airport", "heathrow", "gatwick", "stansted", "luton", "lhr", "lgw", "stn"}
_AFFIRMATION_WORDS = {"yes", "yeah", "yep", "sure", "ok", "okay", "correct"}


async def _send(to: str, body: str):
    """Send message and log it."""
    _log_conversation(to, "bot", body)
    _append_agent_history(to, "assistant", body)
    await messaging_service.send(to, body)


async def _send_without_history(to: str, body: str):
    """Send and log a terminal message without reseeding agent memory."""
    _log_conversation(to, "bot", body)
    await messaging_service.send(to, body)


def _clear_agent_state(session: ConversationSession, phone: str, *, next_state: ConversationState = ConversationState.IDENTIFY):
    session.booking = BookingExtraction()
    session.fare = None
    session.pickup_coords = None
    session.dropoff_coords = None
    session.editing_booking_id = None
    session.state = next_state
    _AGENT_HISTORY.pop(phone, None)


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

    if state == ConversationState.ONBOARD_NAME:
        await _handle_onboard_name(session, from_number, body)

    elif state == ConversationState.ONBOARD_EMAIL:
        await _handle_onboard_email(session, from_number, body)

    elif state == ConversationState.ONBOARD_ADDRESS:
        await _handle_onboard_address(session, from_number, body, latitude, longitude)

    elif state == ConversationState.AWAITING_CONFIRMATION:
        await _handle_confirmation(session, from_number, body)

    else:
        await _handle_agent_turn(session, from_number, body, num_media, data, latitude, longitude)


# ──────────────────────── Agentic Dispatcher ────────────────────────


def _append_agent_history(phone: str, role: str, content: str):
    if not content:
        return
    history = _AGENT_HISTORY.setdefault(phone, [])
    history.append({"role": role, "content": content})
    if len(history) > MAX_AGENT_HISTORY:
        del history[:-MAX_AGENT_HISTORY]


def _get_agent_history(phone: str) -> list[dict]:
    return list(_AGENT_HISTORY.get(phone, []))


def _push_system_context(phone: str, content: str):
    if not content:
        return
    history = _AGENT_HISTORY.setdefault(phone, [])
    history.append({"role": "system", "content": content})
    if len(history) > MAX_AGENT_HISTORY:
        del history[:-MAX_AGENT_HISTORY]


def _has_active_draft(booking: BookingExtraction) -> bool:
    return any(getattr(booking, field) is not None for field in booking.model_fields)


def _get_missing_fields(booking: BookingExtraction) -> list[str]:
    missing: list[str] = []
    for field in CORE_FIELDS:
        if getattr(booking, field) is None:
            missing.append(field)
    if booking.journey_type is None:
        missing.append("journey_type")
    if booking.trip_type and booking.trip_type.lower() == "airport":
        for field in AIRPORT_FIELDS:
            if getattr(booking, field) is None:
                missing.append(field)
    if booking.journey_type and booking.journey_type.lower() == "round trip":
        if booking.wait_time_mins is None:
            missing.append("wait_time_mins")
    if booking.journey_type and booking.journey_type.lower() == "return":
        for field in RETURN_FIELDS:
            if getattr(booking, field) is None:
                missing.append(field)
    if not missing and _notes_missing(booking):
        missing.append("booking_notes")
    return missing


def _customer_context(session: ConversationSession) -> dict:
    full_name = (session.customer_name or "").strip()
    first_name = full_name.split()[0] if full_name else None
    return {
        "customer_id": session.customer_id,
        "customer_name": session.customer_name,
        "customer_first_name": first_name,
        "last_booking_id": session.last_booking_id,
    }


def _booking_context(session: ConversationSession, from_number: str) -> dict:
    draft = session.booking.model_dump()
    return {
        "draft_booking": draft,
        "draft_active": _has_active_draft(session.booking),
        "next_missing_field": _get_next_missing_field(session.booking),
        "missing_fields": _get_missing_fields(session.booking),
        "editing_booking_id": getattr(session, "editing_booking_id", None),
        "awaiting_confirmation": session.state == ConversationState.AWAITING_CONFIRMATION,
        "customer_phone": customer_service.normalize_phone(from_number),
    }


async def _handle_agent_turn(
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

    user_text = await _normalize_agent_input(from_number, body, num_media, data, latitude, longitude)
    if not user_text:
        await _send(from_number, "I'm sorry, I didn't quite catch that. Could you repeat?")
        return

    await _persist_user_details_into_draft(session, user_text)
    _append_agent_history(from_number, "user", user_text)
    agent_response = await llm_service.run_agent(
        _get_agent_history(from_number),
        _customer_context(session),
        _booking_context(session, from_number),
    )
    await _handle_agent_response(session, from_number, agent_response)


async def _normalize_agent_input(
    from_number: str,
    body: str,
    num_media: int,
    data: dict,
    latitude: str | None,
    longitude: str | None,
) -> str | None:
    if latitude and longitude:
        address = await maps_service.reverse_geocode(float(latitude), float(longitude))
        if not address:
            await _send(
                from_number,
                "I couldn't resolve that location. Please try sharing again or type the address.",
            )
            return None
        return f"My pickup location is {address}."

    if num_media > 0:
        media_type = data.get("MediaContentType0", "")
        media_url = data.get("MediaUrl0", "")
        if media_url and (not media_type or "audio" in media_type or "ogg" in media_type):
            try:
                audio_path = await transcription_service.download_media(media_url)
                text = await transcription_service.transcribe_audio(audio_path)
                return text.strip() or None
            except Exception:
                log.exception("Voice processing failed")
                await _send(
                    from_number, "Sorry, I couldn't process that voice message. Please try again."
                )
                return None
        return None

    return body.strip() or None


async def _handle_agent_response(
    session: ConversationSession,
    from_number: str,
    agent_response: dict,
):
    if agent_response.get("type") == "text":
        await _send(from_number, agent_response["message"])
        return

    if agent_response.get("type") != "tool_call":
        await _send(from_number, "I'm sorry, I didn't quite catch that. Could you repeat?")
        return

    tool = agent_response.get("tool")
    arguments = agent_response.get("arguments") or {}

    if tool == "check_driver_status":
        await _execute_check_driver_status(session, from_number, arguments)
        return

    if tool == "cancel_ride":
        await _execute_cancel_ride(session, from_number, arguments)
        return

    if tool == "edit_booking":
        await _execute_edit_booking(session, from_number, arguments)
        return

    if tool == "draft_new_booking":
        await _execute_draft_new_booking(session, from_number, arguments)
        return

    await _send(from_number, "I'm sorry, I didn't quite catch that. Could you repeat?")


async def _execute_check_driver_status(session: ConversationSession, from_number: str, arguments: dict):
    booking_id = _coerce_booking_id(arguments.get("booking_id"))
    await _handle_identify_status(session, from_number, "", booking_id)


def _as_whatsapp_destination(phone: str | None) -> str | None:
    if not phone:
        return None
    normalized = phone.strip()
    if not normalized:
        return None
    if normalized.startswith("whatsapp:"):
        return normalized
    return f"whatsapp:{normalized}"


async def _execute_cancel_ride(session: ConversationSession, from_number: str, arguments: dict):
    booking_id = _coerce_booking_id(arguments.get("booking_id"))
    if booking_id is None:
        await _send(from_number, "Please tell me which booking you want to cancel, for example: cancel #2290.")
        return

    result = await booking_service.cancel_assigned_ride(booking_id)
    if not result:
        await _send(from_number, "No driver is currently assigned to this booking, or the ID is invalid.")
        return

    driver_to = _as_whatsapp_destination(result.get("driver_phone"))
    if driver_to:
        await messaging_service.send(
            driver_to,
            f"\u274c TRIP CANCELLED \u274c\nBooking #{booking_id} has been cancelled by the customer. Please do not proceed to the pickup.",
        )

    await _send(from_number, "Your booking has been cancelled and the driver has been notified.")


async def _execute_edit_booking(session: ConversationSession, from_number: str, arguments: dict):
    booking_id = _coerce_booking_id(arguments.get("booking_id"))
    if booking_id is None:
        await _send(from_number, "Please tell me which booking you want to edit, for example: edit #2290.")
        return

    booking = await booking_service.get_customer_booking(session.customer_id or 0, booking_id)
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
    session.state = ConversationState.IDENTIFY
    await _send(
        from_number,
        f"I've opened booking #{booking_id} for editing. Tell me what you'd like to change.",
    )


async def _execute_draft_new_booking(session: ConversationSession, from_number: str, arguments: dict):
    extraction = _tool_args_to_extraction(arguments)
    _sanitize_extraction(extraction)
    _apply_slot_guardrails(extraction)
    _apply_booking_updates(session.booking, extraction)
    _auto_set_trip_metadata(session.booking)

    if session.editing_booking_id:
        await _maybe_apply_existing_booking_update(session, from_number)
        return

    if _get_next_missing_field(session.booking) is None:
        await _try_fare_calculation(session, from_number)
        return

    followup = await _agent_followup_for_draft(session, from_number, arguments)
    await _send(from_number, followup)


async def _agent_followup_for_draft(session: ConversationSession, from_number: str, arguments: dict) -> str:
    followup = await llm_service.run_agent(
        _get_agent_history(from_number),
        _customer_context(session),
        _booking_context(session, from_number),
    )
    if followup.get("type") == "text":
        return followup["message"]
    return "I'm sorry, I didn't quite catch that. Could you repeat?"


async def _persist_user_details_into_draft(session: ConversationSession, user_text: str):
    normalized = user_text.strip()
    if not normalized:
        return

    next_missing_field = _get_next_missing_field(session.booking)
    lower = normalized.lower()
    if next_missing_field in {"wait_time_mins", "number_of_luggages"} and lower in {"no", "none", "skip", "n/a", "na"}:
        setattr(session.booking, next_missing_field, 0)
        return
    if next_missing_field == "flight_number" and lower in {"no", "none", "skip", "n/a", "na"}:
        session.booking.flight_number = ""
        return
    if next_missing_field == "booking_notes" and lower in {"no", "none", "skip", "n/a", "na"}:
        session.booking.booking_notes = ""
        return
    if next_missing_field == "flight_number" and lower in _AFFIRMATION_WORDS:
        return
    if next_missing_field == "booking_notes" and lower in _AFFIRMATION_WORDS:
        return
    if next_missing_field in DIRECT_ASSIGN_FIELDS:
        if next_missing_field in {"wait_time_mins", "number_of_luggages"}:
            try:
                setattr(session.booking, next_missing_field, int(normalized))
                return
            except ValueError:
                return
        cleaned = _sanitize_field_value(next_missing_field, normalized)
        if cleaned is not None:
            setattr(session.booking, next_missing_field, cleaned)
            _auto_set_trip_metadata(session.booking)
            return

    extraction = await llm_service.extract_booking(user_text)
    _sanitize_extraction(extraction)
    _apply_slot_guardrails(extraction)
    _apply_booking_updates(session.booking, extraction)
    _auto_set_trip_metadata(session.booking)


def _coerce_booking_id(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tool_args_to_extraction(arguments: dict) -> BookingExtraction:
    wait_time = arguments.get("wait_time_mins")
    number_of_luggages = arguments.get("number_of_luggages")
    try:
        wait_time_value = int(wait_time) if wait_time is not None else None
    except (TypeError, ValueError):
        wait_time_value = None
    try:
        number_of_luggages_value = int(number_of_luggages) if number_of_luggages is not None else None
    except (TypeError, ValueError):
        number_of_luggages_value = None

    return BookingExtraction(
        pickup_location=arguments.get("pickup_location"),
        dropoff_location=arguments.get("dropoff_location"),
        pickup_date=arguments.get("date"),
        pickup_time=arguments.get("time"),
        return_journey_date=arguments.get("return_journey_date"),
        return_journey_time=arguments.get("return_journey_time"),
        journey_type=arguments.get("journey_type"),
        wait_time_mins=wait_time_value,
        number_of_luggages=number_of_luggages_value,
        flight_number=arguments.get("flight_number"),
        booking_notes=arguments.get("booking_notes"),
    )


async def _handle_identify(
    session: ConversationSession,
    from_number: str,
    body: str,
    num_media: int,
    data: dict,
    latitude: str | None,
    longitude: str | None,
):
    await _handle_agent_turn(session, from_number, body, num_media, data, latitude, longitude)


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

    item = await booking_service.get_booking_status_for_customer(session.customer_id, None)
    if not item:
        message = "I couldn't find any active upcoming bookings for you."
        if bookings_info:
            message += bookings_info
        await _send(from_number, message)
        return

    session.last_booking_id = item["booking_id"]
    await _send(from_number, _format_status_message(item))


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
        booking_notes=row.get("booking_notes"),
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
    session.state = ConversationState.IDENTIFY

    await _send(
        from_number,
        f"You're all set, {session.customer_name}!\n\n"
        f"Address saved: {geo['formatted_address']}\n\n"
        "Now, how can I help you? Send a text or voice message describing your trip, "
        "or share a live location for pickup.",
    )


# ──────────────────────── Legacy Helpers ────────────────────────


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
    if b.return_journey_date:
        context_parts.append(f"Return date is already set to: {b.return_journey_date}")
    if b.return_journey_time:
        context_parts.append(f"Return time is already set to: {b.return_journey_time}")
    if b.booking_notes is not None:
        context_parts.append(f"Booking notes are already set to: {b.booking_notes}")

    if context_parts:
        augmented = "Context: " + "; ".join(context_parts) + "\nUser message: " + text
    else:
        augmented = text

    extraction = await llm_service.extract_booking(augmented)
    _sanitize_extraction(extraction)
    _apply_slot_guardrails(extraction)
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
    else:
        booking.flight_number = None
        booking.number_of_luggages = None
        booking.flight_journey_type = None

    if booking.journey_type == "Round Trip":
        booking.return_journey_date = None
        booking.return_journey_time = None
    elif booking.journey_type == "Return":
        booking.wait_time_mins = None
    elif booking.journey_type == "One Way":
        booking.return_journey_date = None
        booking.return_journey_time = None
        if booking.wait_time_mins is None:
            booking.wait_time_mins = 0


def _sanitize_extraction(extraction: BookingExtraction):
    extraction.journey_type = _sanitize_journey_type(extraction.journey_type)
    extraction.pickup_time = _sanitize_time_value(extraction.pickup_time)
    extraction.return_journey_time = _sanitize_time_value(extraction.return_journey_time)
    _enforce_journey_field_consistency(extraction)


def _sanitize_field_value(field: str, value: str):
    if field == "journey_type":
        return _sanitize_journey_type(value)
    if field in {"pickup_time", "return_journey_time"}:
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


def _enforce_journey_field_consistency(extraction: BookingExtraction):
    journey_type = extraction.journey_type
    if journey_type == "Round Trip":
        extraction.return_journey_date = None
        extraction.return_journey_time = None
    elif journey_type == "Return":
        extraction.wait_time_mins = None
    elif journey_type == "One Way":
        extraction.wait_time_mins = None
        extraction.return_journey_date = None
        extraction.return_journey_time = None


def _apply_slot_guardrails(extraction: BookingExtraction):
    for field in ("pickup_location", "dropoff_location", "flight_number", "booking_notes"):
        value = getattr(extraction, field)
        if isinstance(value, str) and value.strip().lower() in _AFFIRMATION_WORDS:
            setattr(extraction, field, None)


def _sanitize_time_value(value: str | None) -> str | None:
    if not value:
        return None

    stripped = value.strip()
    compact = re.sub(r"\s+", "", stripped.lower())

    match = re.fullmatch(r"(\d{1,2})(am|pm)", compact)
    if match:
        hour, suffix = match.groups()
        return f"{int(hour)}:00 {suffix.upper()}"

    match = re.fullmatch(r"(\d{1,2})([ap])", compact)
    if match:
        hour, suffix = match.groups()
        return f"{int(hour)}:00 {suffix.upper()}M"

    if re.fullmatch(r"\d{3,4}", compact):
        padded = compact.zfill(4)
        hour = int(padded[:2])
        minute = int(padded[2:])
        if hour <= 23 and minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    has_colon = ":" in stripped
    has_ampm = bool(re.search(r"\b(?:am|pm)\b", stripped, re.IGNORECASE))
    if not has_colon and not has_ampm:
        return None
    return stripped


def _notes_missing(booking: BookingExtraction) -> bool:
    return booking.booking_notes is None


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
    resolved_return_date = _normalize_date(b.return_journey_date or "")
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
    if b.journey_type == "Return":
        lines.append(f"Return Date: {resolved_return_date or b.return_journey_date}")
        lines.append(f"Return Time: {b.return_journey_time}")
    if b.flight_number:
        lines.append(f"Flight: {b.flight_number} ({b.flight_journey_type or 'N/A'})")
    if b.number_of_luggages is not None:
        lines.append(f"Luggage: {b.number_of_luggages}")
    if b.wait_time_mins and b.wait_time_mins > 0:
        lines.append(f"Wait Time: {b.wait_time_mins} mins")
    if b.booking_notes:
        lines.append(f"Booking Notes: {b.booking_notes}")
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
    5. Return: return_journey_date, return_journey_time
    6. Final optional booking_notes
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

    if booking.journey_type and booking.journey_type.lower() == "return":
        for field in RETURN_FIELDS:
            if getattr(booking, field) is None:
                return field

    if _notes_missing(booking):
        return "booking_notes"

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
    resolved_return_date = _normalize_date(b.return_journey_date or "")
    lines = [
        "Here's your booking summary:\n",
        f"Pickup: {b.pickup_location}",
        f"Dropoff: {b.dropoff_location}",
        f"Date: {resolved_date or b.pickup_date}",
        f"Time: {b.pickup_time}",
        f"Distance: {fare.display_distance_miles or fare.distance_miles} miles",
        f"Est. Duration: {fare.display_duration_mins or fare.duration_mins} mins",
        f"Fare: \u00a3{fare.total_fare:.2f}",
    ]
    if fare.wait_surcharge > 0:
        lines.append(f"  (includes \u00a3{fare.wait_surcharge:.2f} wait surcharge)")
    lines.append(f"Journey: {b.journey_type}")
    if b.trip_type:
        lines.append(f"Trip Type: {b.trip_type}")
    if b.journey_type == "Return":
        lines.append(f"Return Date: {resolved_return_date or b.return_journey_date}")
        lines.append(f"Return Time: {b.return_journey_time}")
    if b.flight_number:
        lines.append(f"Flight: {b.flight_number} ({b.flight_journey_type or 'N/A'})")
    if b.number_of_luggages is not None:
        lines.append(f"Luggage: {b.number_of_luggages}")
    if b.wait_time_mins and b.wait_time_mins > 0:
        lines.append(f"Wait Time: {b.wait_time_mins} mins")
    if b.booking_notes:
        lines.append(f"Booking Notes: {b.booking_notes}")

    lines.append("\nReply YES to confirm or NO to make changes.")
    return "\n".join(lines)


# ──────────────────────── Phase D: Confirmation ────────────────────────


async def _handle_confirmation(
    session: ConversationSession, from_number: str, body: str
):
    text_lower = body.strip().lower()
    yes_words = ["yes", "yeah", "yep", "okay", "ok", "sure", "proceed", "confirm"]

    if any(word in text_lower for word in yes_words):
        booking_id = await _insert_booking(session, from_number)
        if booking_id:
            session.last_booking_id = booking_id
            _clear_agent_state(session, from_number)
            await _send_without_history(
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

    elif "no" in text_lower:
        session.state = ConversationState.IDENTIFY
        session.fare = None
        _push_system_context(
            from_number,
            "The user wants to make changes to the current draft. Ask them what they want to change.",
        )
        followup = await llm_service.run_agent(
            _get_agent_history(from_number),
            _customer_context(session),
            _booking_context(session, from_number),
        )
        if followup.get("type") == "text":
            await _send(from_number, followup["message"])
        else:
            await _send(from_number, "Tell me what you'd like to change in the current draft.")

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
    resolved_return_date = _normalize_date(b.return_journey_date or "")
    resolved_return_time = _normalize_time(b.return_journey_time or "")

    if not resolved_date or not resolved_time:
        log.error("Could not resolve date/time: %s / %s", b.pickup_date, b.pickup_time)
        return None
    if b.journey_type == "Return" and (not resolved_return_date or not resolved_return_time):
        log.error(
            "Could not resolve return date/time: %s / %s",
            b.return_journey_date,
            b.return_journey_time,
        )
        return None

    phone = customer_service.normalize_phone(from_number)
    try:
        if b.journey_type != "Return":
            create_dropoff_location = b.dropoff_location or ""
            create_distance_miles = (fare.display_distance_miles or fare.distance_miles) if fare else None
            create_duration_mins = (fare.display_duration_mins or fare.duration_mins) if fare else None
            create_booking_notes = b.booking_notes or None

            if b.journey_type == "Round Trip":
                create_dropoff_location = b.pickup_location or ""
                create_distance_miles = fare.display_distance_miles if fare else None
                create_duration_mins = (
                    (fare.display_duration_mins or 0) + (b.wait_time_mins or 0)
                    if fare
                    else None
                )
                create_booking_notes = _format_round_trip_booking_notes(
                    b.booking_notes,
                    b.dropoff_location,
                    b.wait_time_mins or 0,
                )

            create = _build_booking_create(
                session,
                phone,
                pickup_date=resolved_date,
                pickup_time=resolved_time,
                pickup_location=b.pickup_location or "",
                dropoff_location=create_dropoff_location,
                booking_notes=create_booking_notes,
                est_ride_duration_mins=create_duration_mins,
                est_ride_distance_miles=create_distance_miles,
                est_ride_fare=fare.total_fare if fare else None,
                wait_time_mins=b.wait_time_mins or 0,
                base_to_src_distance_miles=fare.base_to_src_miles if fare else None,
                dest_to_base_distance_miles=fare.dest_to_base_miles if fare else None,
                est_base_to_src_duration_mins=fare.base_to_src_mins if fare else None,
                est_dest_to_base_duration_mins=fare.dest_to_base_mins if fare else None,
                journey_type=b.journey_type,
                flight_number=b.flight_number,
                flight_journey_type=b.flight_journey_type,
                number_of_luggages=b.number_of_luggages,
            )
            booking_id = await booking_service.insert_booking(create)
            if not booking_id:
                log.error(
                    "Booking insert returned no booking_id for non-return trip. customer_id=%s pickup=%s dropoff=%s journey_type=%s",
                    session.customer_id,
                    b.pickup_location,
                    b.dropoff_location,
                    b.journey_type,
                )
            return booking_id

        outbound_create = _build_booking_create(
            session,
            phone,
            pickup_date=resolved_date,
            pickup_time=resolved_time,
            pickup_location=b.pickup_location or "",
            dropoff_location=b.dropoff_location or "",
            booking_notes=b.booking_notes or None,
            est_ride_duration_mins=fare.duration_mins if fare else None,
            est_ride_distance_miles=fare.distance_miles if fare else None,
            est_ride_fare=round((fare.total_fare / 2), 2) if fare else None,
            wait_time_mins=0,
            base_to_src_distance_miles=fare.base_to_src_miles if fare else None,
            dest_to_base_distance_miles=fare.dest_to_base_miles if fare else None,
            est_base_to_src_duration_mins=fare.base_to_src_mins if fare else None,
            est_dest_to_base_duration_mins=fare.dest_to_base_mins if fare else None,
            journey_type="Return",
            flight_number=b.flight_number,
            flight_journey_type=_derive_flight_journey_type(b.pickup_location, b.dropoff_location),
            number_of_luggages=b.number_of_luggages,
        )
        outbound_id = await booking_service.insert_booking(outbound_create)
        if not outbound_id:
            log.error(
                "Outbound return-leg insert returned no booking_id. customer_id=%s pickup=%s dropoff=%s pickup_date=%s pickup_time=%s",
                session.customer_id,
                b.pickup_location,
                b.dropoff_location,
                resolved_date,
                resolved_time,
            )
            return None

        inbound_create = _build_booking_create(
            session,
            phone,
            pickup_date=resolved_return_date,
            pickup_time=resolved_return_time,
            pickup_location=b.dropoff_location or "",
            dropoff_location=b.pickup_location or "",
            booking_notes=_prepend_return_journey_notes(b.booking_notes),
            est_ride_duration_mins=fare.duration_mins if fare else None,
            est_ride_distance_miles=fare.distance_miles if fare else None,
            est_ride_fare=round((fare.total_fare / 2), 2) if fare else None,
            wait_time_mins=0,
            base_to_src_distance_miles=fare.dest_to_base_miles if fare else None,
            dest_to_base_distance_miles=fare.base_to_src_miles if fare else None,
            est_base_to_src_duration_mins=fare.dest_to_base_mins if fare else None,
            est_dest_to_base_duration_mins=fare.base_to_src_mins if fare else None,
            journey_type="Return",
            flight_number=b.flight_number,
            flight_journey_type=_derive_flight_journey_type(b.dropoff_location, b.pickup_location),
            number_of_luggages=b.number_of_luggages,
        )
        inbound_id = await booking_service.insert_booking(inbound_create)
        if not inbound_id:
            log.error(
                "Inbound return-leg insert returned no booking_id. outbound_id=%s pickup=%s dropoff=%s pickup_date=%s pickup_time=%s",
                outbound_id,
                b.dropoff_location,
                b.pickup_location,
                resolved_return_date,
                resolved_return_time,
            )
            return None
        return outbound_id
    except Exception:
        log.exception(
            "Failed to insert booking. journey_type=%s customer_id=%s pickup=%s dropoff=%s pickup_date=%s pickup_time=%s return_date=%s return_time=%s",
            b.journey_type,
            session.customer_id,
            b.pickup_location,
            b.dropoff_location,
            b.pickup_date,
            b.pickup_time,
            b.return_journey_date,
            b.return_journey_time,
        )
        return None


def _build_booking_create(
    session: ConversationSession,
    customer_phone: str,
    *,
    pickup_date,
    pickup_time,
    pickup_location: str,
    dropoff_location: str,
    booking_notes: str | None,
    est_ride_duration_mins: int | None,
    est_ride_distance_miles: float | None,
    est_ride_fare: float | None,
    wait_time_mins: int | None,
    base_to_src_distance_miles: float | None,
    dest_to_base_distance_miles: float | None,
    est_base_to_src_duration_mins: int | None,
    est_dest_to_base_duration_mins: int | None,
    journey_type: str | None,
    flight_number: str | None,
    flight_journey_type: str | None,
    number_of_luggages: int | None,
) -> BookingCreate:
    return BookingCreate(
        pickup_date=pickup_date,
        pickup_time=pickup_time,
        pickup_location=pickup_location,
        dropoff_location=dropoff_location,
        dropoff_location_type=_derive_trip_type(pickup_location, dropoff_location),
        customer_id=session.customer_id or 0,
        customer_phone=customer_phone,
        booking_notes=booking_notes,
        est_ride_duration_mins=est_ride_duration_mins,
        est_ride_distance_miles=est_ride_distance_miles,
        est_ride_fare=est_ride_fare,
        currency="GBP",
        wait_time_mins=wait_time_mins,
        base_location="Home",
        base_to_src_distance_miles=base_to_src_distance_miles,
        dest_to_base_distance_miles=dest_to_base_distance_miles,
        est_base_to_src_duration_mins=est_base_to_src_duration_mins,
        est_dest_to_base_duration_mins=est_dest_to_base_duration_mins,
        journey_type=journey_type,
        flight_number=flight_number,
        flight_journey_type=flight_journey_type,
        number_of_luggages=number_of_luggages,
        booking_type="Regular",
    )


def _prepend_return_journey_notes(notes: str | None) -> str:
    clean = (notes or "").strip()
    if clean:
        return f"RETURN JOURNEY. {clean}"
    return "RETURN JOURNEY."


def _format_round_trip_booking_notes(notes: str | None, destination: str | None, wait_time_mins: int) -> str:
    parts = ["ROUND TRIP."]
    clean_notes = (notes or "").strip()
    if clean_notes:
        parts.append(clean_notes)
    parts.append(f"Destination: {destination or ''}".strip())
    parts.append(f"Wait: {wait_time_mins} mins")
    return " | ".join(part for part in parts if part).replace(". |", ".")


def _derive_trip_type(pickup_location: str | None, dropoff_location: str | None) -> str | None:
    pickup = (pickup_location or "").lower()
    dropoff = (dropoff_location or "").lower()
    if any(kw in pickup for kw in _AIRPORT_KEYWORDS) or any(kw in dropoff for kw in _AIRPORT_KEYWORDS):
        return "Airport"
    if pickup_location and dropoff_location:
        return "Local Trip"
    return None


def _derive_flight_journey_type(pickup_location: str | None, dropoff_location: str | None) -> str | None:
    pickup = (pickup_location or "").lower()
    dropoff = (dropoff_location or "").lower()
    pickup_is_airport = any(kw in pickup for kw in _AIRPORT_KEYWORDS)
    dropoff_is_airport = any(kw in dropoff for kw in _AIRPORT_KEYWORDS)
    if dropoff_is_airport:
        return "Departure"
    if pickup_is_airport:
        return "Arrival"
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
    await _handle_agent_turn(session, from_number, body, 0, {}, None, None)


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
            "I still need the missing trip timing details before I can update this booking. Please include the date and time clearly.",
        )
        return

    phone = customer_service.normalize_phone(from_number)
    updated_id = await booking_service.update_booking(
        booking_id,
        phone,
        pickup_date=resolved_date,
        pickup_time=resolved_time,
        pickup_location=session.booking.pickup_location or "",
        dropoff_location=session.booking.dropoff_location or "",
        dropoff_location_type=session.booking.trip_type,
        booking_notes=session.booking.booking_notes or None,
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


async def _maybe_apply_existing_booking_update(session: ConversationSession, from_number: str):
    booking_id = session.editing_booking_id
    if not booking_id:
        followup = await _agent_followup_for_draft(session, from_number, {})
        await _send(from_number, followup)
        return

    if await booking_service.has_assignment(booking_id):
        session.editing_booking_id = None
        session.state = ConversationState.IDENTIFY
        await _send(
            from_number,
            f"Booking #{booking_id} has now been assigned to a driver, so automatic edits are blocked. Please contact support for assistance.",
        )
        return

    resolved_date = _normalize_date(session.booking.pickup_date or "")
    resolved_time = _normalize_time(session.booking.pickup_time or "")
    if _get_next_missing_field(session.booking) is not None:
        followup = await _agent_followup_for_draft(session, from_number, {})
        await _send(from_number, followup)
        return
    if not resolved_date or not resolved_time:
        followup = await _agent_followup_for_draft(session, from_number, {})
        await _send(from_number, followup)
        return

    phone = customer_service.normalize_phone(from_number)
    updated_id = await booking_service.update_booking(
        booking_id,
        phone,
        pickup_date=resolved_date,
        pickup_time=resolved_time,
        pickup_location=session.booking.pickup_location or "",
        dropoff_location=session.booking.dropoff_location or "",
        dropoff_location_type=session.booking.trip_type,
        booking_notes=session.booking.booking_notes or None,
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
    await _try_fare_calculation(session, from_number)


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
