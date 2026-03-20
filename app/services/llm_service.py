import json
import logging
import re

from openai import AsyncOpenAI

from app.config import settings
from app.models.booking import BookingExtraction

log = logging.getLogger(__name__)

_llm_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            base_url=settings.VLLM_BASE_URL,
            api_key="not-needed",
        )
    return _llm_client


EXTRACTION_SYSTEM_PROMPT = (
    "Output raw JSON only. No markdown. No explanation. No code blocks.\n"
    "Extract taxi booking fields from the user message.\n"
    "JSON schema: "
    '{"journey_type":string|null,"pickup_location":string|null,'
    '"dropoff_location":string|null,"pickup_date":string|null,'
    '"pickup_time":string|null,"trip_type":string|null,'
    '"flight_number":string|null,"flight_journey_type":string|null,'
    '"wait_time_mins":int|null,"number_of_luggages":int|null,'
    '"passengers":int|null}\n'
    "Rules:\n"
    '- journey_type: ONLY set if user explicitly says "return", "round trip", or "one way". null otherwise.\n'
    '- Keep dates as spoken (e.g. "tomorrow", "next Friday")\n'
    '- trip_type: "Airport" if any location is an airport, else "Local Trip"\n'
    '- flight_journey_type: "Arrival" or "Departure"\n'
    "- null for anything not explicitly mentioned by the user"
)

INTENT_SYSTEM_PROMPT = (
    "Output raw JSON only. No markdown. No explanation. No code blocks.\n"
    'Classify the user message into exactly one intent from: "NEW_BOOKING", "CHECK_STATUS", "EDIT_BOOKING", "GENERAL_CHAT".\n'
    'JSON schema: {"intent":"NEW_BOOKING|CHECK_STATUS|EDIT_BOOKING|GENERAL_CHAT","booking_id":int|null}\n'
    "Rules:\n"
    '- Use "CHECK_STATUS" for driver status, assigned driver, car, ETA, vehicle, plate, or booking progress questions.\n'
    '- Use "EDIT_BOOKING" when the user asks to edit, change, update, reschedule, or modify an existing booking.\n'
    '- Use "NEW_BOOKING" for any request to make a new ride, taxi, pickup, dropoff, or trip.\n'
    '- Use "GENERAL_CHAT" for greetings, farewells, thanks, no-thanks, casual conversation, or off-topic questions.\n'
    "- Extract booking_id only if explicitly present in the message, otherwise null."
)

CHAT_REPLY_SYSTEM_PROMPT = (
    "You are a polite, concise taxi booking assistant on WhatsApp.\n"
    "Reply in 1-2 short sentences.\n"
    "Handle greetings, thanks, farewells, and small talk warmly.\n"
    "Do not start a booking flow unless the user asks for a ride.\n"
    "If appropriate, gently mention that you can help book a taxi."
)


async def extract_booking(text: str) -> BookingExtraction:
    """Extract booking fields from user text via local vLLM."""
    try:
        response = await _get_client().chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        log.info("LLM raw output: %s", raw)
        parsed = _parse_json(raw)
        return BookingExtraction(**parsed)
    except Exception:
        log.exception("LLM extraction failed for text: %s", text)
        return BookingExtraction()


async def classify_intent(text: str) -> dict:
    """Classify identify-state user text into a routing intent."""
    booking_id = _extract_booking_id(text)
    lower = text.lower().strip()
    if lower in _GREETING_KEYWORDS:
        return {"intent": "GENERAL_CHAT", "booking_id": booking_id}

    try:
        response = await _get_client().chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=64,
        )
        raw = response.choices[0].message.content.strip()
        log.info("Intent raw output: %s", raw)
        parsed = _parse_json(raw)
        intent = parsed.get("intent")
        if intent not in {"NEW_BOOKING", "CHECK_STATUS", "EDIT_BOOKING", "GENERAL_CHAT"}:
            raise ValueError(f"Invalid intent: {intent}")
        parsed["booking_id"] = parsed.get("booking_id") or booking_id
        return parsed
    except Exception:
        log.exception("LLM intent classification failed for text: %s", text)
        if any(kw in lower for kw in _DRIVER_KEYWORDS):
            return {"intent": "CHECK_STATUS", "booking_id": booking_id}
        if any(kw in lower for kw in {"edit", "change", "update", "reschedule", "modify"}):
            return {"intent": "EDIT_BOOKING", "booking_id": booking_id}
        if lower in _GREETING_KEYWORDS or any(kw in lower for kw in _CHAT_KEYWORDS):
            return {"intent": "GENERAL_CHAT", "booking_id": booking_id}
        return {"intent": "NEW_BOOKING", "booking_id": booking_id}


async def generate_conversational_reply(text: str) -> str:
    """Generate a short conversational reply without entering booking flow."""
    lower = text.lower().strip()
    if lower in _GREETING_KEYWORDS:
        return "Hello! I can help whenever you need a ride."
    if any(kw in lower for kw in {"thanks", "thank you", "thx"}):
        return "You're welcome! Let me know if you need a ride later."
    if any(kw in lower for kw in {"bye", "goodbye", "see you", "no thanks", "no thank you"}):
        return "Goodbye! If you need a taxi later, just message me."

    try:
        response = await _get_client().chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=[
                {"role": "system", "content": CHAT_REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=60,
        )
        reply = response.choices[0].message.content.strip()
        return reply or "I’m here if you need a taxi."
    except Exception:
        log.exception("Conversational reply generation failed for text: %s", text)
        return "I’m here if you need a taxi."


def _parse_json(raw: str) -> dict:
    """Multi-step fallback chain for parsing LLM JSON output."""
    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Extract JSON object via regex
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 3. Try each line
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    log.warning("Could not parse LLM JSON output: %s", raw)
    return {}


def _extract_booking_id(text: str) -> int | None:
    match = re.search(r"#?(\d{3,})", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


_DRIVER_KEYWORDS = {"driver", "car", "vehicle", "who", "assigned", "plate", "registration"}
_GREETING_KEYWORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
_CHAT_KEYWORDS = {"thanks", "thank you", "thx", "bye", "goodbye", "see you", "no thanks", "no thank you"}


def detect_intent(text: str) -> str:
    """Classify user intent via keyword matching.

    Returns: "driver_query", "greeting", or "booking"
    """
    lower = text.lower().strip()
    if any(kw in lower for kw in _DRIVER_KEYWORDS):
        return "driver_query"
    if lower in _GREETING_KEYWORDS:
        return "greeting"
    return "booking"
