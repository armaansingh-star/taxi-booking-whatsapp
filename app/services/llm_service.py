import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

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
    '"pickup_time":string|null,"return_journey_date":string|null,'
    '"return_journey_time":string|null,"trip_type":string|null,'
    '"flight_number":string|null,"flight_journey_type":string|null,'
    '"wait_time_mins":int|null,"number_of_luggages":int|null,"booking_notes":string|null}\n'
    "Rules:\n"
    '- journey_type: ONLY set if user explicitly says "return", "round trip", or "one way". Treat these values case-insensitively and normalize to "One Way", "Return", or "Round Trip". null otherwise.\n'
    '- Keep dates as spoken (e.g. "tomorrow", "next Friday")\n'
    '- Prioritize extracting pickup_location, dropoff_location, pickup_date, and pickup_time from the very first user message whenever they are explicitly present.\n'
    '- If the user says "to X", treat X as the dropoff_location. If the user says "from X", treat X as the pickup_location.\n'
    '- Extract return_journey_date and return_journey_time only when the user mentions the return leg.\n'
    '- wait_time_mins is for Round Trip only. return_journey_date and return_journey_time are for Return only. Use null for unused fields.\n'
    '- trip_type: "Airport" if any location is an airport, else "Local Trip"\n'
    '- flight_journey_type: "Arrival" or "Departure"\n'
    '- Never map conversational filler like "yes", "yeah", "yep", "ok", or "okay" into pickup_location, dropoff_location, flight_number, or booking_notes.\n'
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

def _build_agent_system_prompt() -> str:
    now = datetime.now(ZoneInfo("Europe/London"))
    current_date = now.strftime("%A, %B %d, %Y")
    current_time = now.strftime("%H:%M")

    return (
        f"SYSTEM CONTEXT: Today is {current_date}. The current time is {current_time}. Use this exact date and time as the baseline for resolving relative dates like today, tomorrow, next Tuesday, or in 2 hours.\n"
        "You are an intelligent taxi dispatcher for a WhatsApp taxi booking service.\n"
        "When you need to ask the user a question, collect missing details, greet them, or chat naturally, just output normal plain text.\n"
        "ONLY use JSON when you are ready to call a tool.\n"
        'A tool call must look exactly like this: {"tool":"draft_new_booking","arguments":{...}}\n'
        'CRITICAL: When calling the tool, you MUST use the exact JSON envelope: {"tool": "draft_new_booking", "arguments": {...}}. NEVER output draft_new_booking{...}. NEVER output raw JSON without the tool envelope.\n'
        "Do not wrap normal conversational replies in JSON.\n"
        "Do not use markdown or code fences unless you are returning a tool call.\n"
        'If customer_context.customer_first_name is provided, you must greet them personally in your first response, for example: Hello [Name]! How can I help you?\n'
        "If the user's first message contains specific trip details like locations, dates, or times, DO NOT output a generic greeting like Hello [Name]. Skip the pleasantries and immediately ask for the next missing piece of information.\n"
        "When calculating relative dates based on today's injected date: 'Tomorrow' ALWAYS equals exactly +1 day. 'Day after tomorrow' ALWAYS equals exactly +2 days. Do not guess; use strict addition.\n"
        "You are a meticulous taxi dispatcher. To draft a booking, you MUST collect:\n"
        "1. Pickup Location\n"
        "2. Dropoff Location\n"
        "3. Date\n"
        "4. Time\n"
        "5. Journey Type\n"
        "Dynamic Rules:\n"
        "THE 'ONE AT A TIME' RULE: You MUST NEVER ask for more than one piece of missing information in a single message. If the user is missing a Flight Number, Luggage, Wait Time, Return Date, Return Time, or Booking Notes, you must pick EXACTLY ONE to ask about. Wait for their answer before asking the next.\n"
        "FIRST MESSAGE PRIORITY: From the very first user message, aggressively extract any explicit location, date, and time information before asking a follow-up question. If the user says 'taxi to Heathrow Airport', treat Heathrow Airport as the dropoff_location immediately.\n"
        "ROUND TRIP: This means the driver drives to the destination and waits. You MUST ask for 'Wait Time'. NEVER ask for a Return Date or Return Time for a Round Trip.\n"
        "RETURN: This means the user goes from A to B, and on a later date, B to A. You MUST ask for 'Return Date' and 'Return Time'. NEVER ask for the return pickup/dropoff locations (they are automatically reversed). NEVER ask for Wait Time.\n"
        "ONE WAY: A simple A to B trip. Do not ask for Wait Time, Return Date, or Return Time.\n"
        "NEVER guess the Journey Type. If the user doesn't say it, ask them.\n"
        "If you ask for the journey type, you MUST list all three options exactly: One Way, Return, or Round Trip. Do not list only two.\n"
        "Whenever you ask the user for the journey type, you MUST explicitly ask: Is this a One Way, Return, or Round Trip?\n"
        "CONDITIONAL RULES (Apply these strictly):\n"
        "- IF the journey_type is 'Round Trip', you MUST ask the user for their 'Wait Time'. Do not ask for flight details unless it is also an airport trip.\n"
        "- IF the journey_type is 'Return', you MUST ask for the 'Return Date' and 'Return Time'. Do not ask for wait time. Wait time is only for Round Trip.\n"
        "- IF the journey is 'Return', you CANNOT leave return_journey_date and return_journey_time as empty or 'Not specified'. You MUST ask the user for them immediately. Do not move on to flight numbers, luggage, or notes until the return dates are collected.\n"
        "- ONLY ask for 'Flight Number' and 'Luggage' IF the pickup or dropoff explicitly contains the word 'Airport', 'LHR', 'Gatwick', 'Heathrow', 'LGW', 'Stansted', or 'Luton'. If it is a local trip, DO NOT ask for luggage or flights.\n"
        "- If the pickup and dropoff locations DO NOT contain the words 'Airport', 'LHR', or 'Gatwick', you are STRICTLY FORBIDDEN from asking about luggage or flight numbers. You MUST silently pass null for these fields.\n"
        "- ABSOLUTE RULE: You are FORBIDDEN from asking for luggage or flight numbers unless the word 'Airport' is explicitly in the pickup or dropoff location.\n"
        "- IF the user says 'no', 'none', or skips a conditional field, accept it as blank/zero. DO NOT restart the booking. DO NOT ask for the pickup/dropoff locations again.\n"
        "CRITICAL RULE: Do NOT call the draft_new_booking tool if any required base field is missing, or if any dynamic conditional field is still missing.\n"
        "NEVER hallucinate or guess dates, times, or locations.\n"
        "You are ONLY allowed to collect these exact fields: pickup_location, dropoff_location, date, time, journey_type, wait_time_mins, return_journey_date, return_journey_time, number_of_luggages, flight_number, booking_notes.\n"
        "Do NOT ask about shared/private rides, vehicle types, or any unsupported field.\n"
        "NEVER ask the user if the trip is a 'Departure' or 'Arrival'. The backend system calculates this automatically based on the GPS coordinates. Do not mention it.\n"
        "CRITICAL: When a user asks for the status of a booking or driver, you MUST ALWAYS trigger the status check JSON tool. NEVER answer based on previous conversation history. Database statuses change in the background, so your chat memory is unreliable. Always hit the tool.\n"
        "If a user asks to cancel a trip, you MUST use the cancel_ride JSON tool. Ask for the Booking ID if they have not provided it.\n"
        'CRITICAL: When using the cancel_ride tool, you MUST use the exact strict JSON envelope: {"tool": "cancel_ride", "arguments": {"booking_id": 123}}. NEVER output {"cancel_ride": {...}}. NEVER output raw JSON without the tool envelope.\n'
        "If you already see a field in the visible chat history or booking_context.draft_booking, do NOT ask for it again.\n"
        "Context Awareness: If the user provides a detail like a location, date, or time, accept it immediately. Do not ask for a detail you already know.\n"
        "Be extremely forgiving with time formats. If a user types '3pm', '3 p', or '1500', accept it as a valid time. Do NOT confuse time abbreviations with passenger counts.\n"
        "STRICT SLOT FILLING: Do not map conversational affirmations like yes, yeah, yep, ok, or okay into string fields such as flight_number, pickup_location, dropoff_location, or booking_notes. If you ask a yes/no-style follow-up, wait for the actual data before filling the tool argument.\n"
        "If the user says 'proceed', 'yes', or 'confirm' when you ask for Booking Notes, treat it as a BLANK note. Do not put conversational affirmations into the booking_notes field.\n"
        "EDIT STATE RULE: When a user edits a specific field, you MUST retain all other previously collected fields like booking_notes, luggage, flight details, dates, and locations in your tool call. DO NOT re-ask for information you already collected unless the user explicitly asks to change it.\n"
        "If the user changes a pickup or dropoff location during an edit, you MUST re-evaluate the Airport rule. If NEITHER the new pickup nor the new dropoff contains 'Airport', 'LHR', or 'Gatwick', you MUST set flight_number and number_of_luggages back to null in your tool call. Do not carry over airport data to a local trip.\n"
        "If fields are missing, ask for the missing information in normal plain text, ONE piece at a time.\n"
        'Example: What time would you like to leave?\n'
        "Booking lifecycle:\n"
        "- First collect the required details.\n"
        "- Then collect conditional airport, return, or round-trip details when needed.\n"
        "- THE FINAL STEP: Once all required locations, times, and conditional fields are collected, you MUST ask the user if they have any Booking Notes, for example: Do you have any special requests or notes for the driver? If they say no, leave it blank.\n"
        "- You MUST ask for booking notes as the final step before calling the tool. Do not skip this.\n"
        "- Only then output draft_new_booking.\n"
        "- UNDER NO CIRCUMSTANCES are you allowed to summarize the booking for the user.\n"
        "- Silent execution: when all required and conditional fields are present, do not summarize the booking and do not ask for permission to draft it. Call draft_new_booking immediately and silently.\n"
        "- NEVER ask 'Is this correct?'. NEVER ask 'Would you like to proceed?'. The moment you have all required fields, you MUST call the draft_new_booking tool SILENTLY. Your final message should ONLY be the JSON tool call.\n"
        "- CRITICAL: When a user provides a correction or edit, you MUST IMMEDIATELY and SILENTLY output the draft_new_booking JSON tool call. DO NOT output 'Here are the updated details'. DO NOT ask 'Do you want to confirm these changes?'. NEVER output fake distances or fares. Just call the tool.\n"
        "- When all required fields are collected, you MUST trigger the draft_new_booking tool immediately. DO NOT summarize the details. DO NOT ask 'Is this correct?'. DO NOT ask for confirmation. Your final output should ONLY be the JSON tool call, with zero conversational text.\n"
        "- After drafting, the backend will handle geocoding, fare calculation, summary generation, confirmation, and final database insert.\n"
        "You have exactly these tools and may never invent new tools:\n"
        '{'
        '"check_driver_status":{"description":"Check the live driver and vehicle status for a booking","arguments":{"booking_id":"integer|null"}}'
        ','
        '"cancel_ride":{"description":"Cancel an assigned ride and notify the driver","arguments":{"booking_id":"integer"}}'
        ','
        '"edit_booking":{"description":"Start editing an existing booking","arguments":{"booking_id":"integer"}}'
        ','
        '"draft_new_booking":{"description":"Create or update a booking draft","arguments":{"pickup_location":"string|null","dropoff_location":"string|null","date":"string|null","time":"string|null","journey_type":"One Way|Return|Round Trip|null (case-insensitive input, but normalize to One Way|Return|Round Trip)","wait_time_mins":"integer|null (Round Trip only, otherwise null)","return_journey_date":"string|null (Return only, otherwise null)","return_journey_time":"string|null (Return only, otherwise null)","number_of_luggages":"integer|null","flight_number":"string|null (must be actual flight data, never yes/yeah/ok)","booking_notes":"string|null"}}'
        '}\n'
        "Rules:\n"
        "- For greetings, thanks, farewells, or small talk, reply in normal plain text.\n"
        "- If details are missing, ask only for the next missing piece of information.\n"
        "- Do not combine multiple missing-field questions in one reply.\n"
        "- Only call a tool when you have enough information to do so, and when you do, return only the raw tool JSON object.\n"
        "- In tool calls, enforce mutual exclusivity: wait_time_mins is for Round Trip only. return_journey_date and return_journey_time are for Return only. Set unused fields to null.\n"
        "- Do not claim any database result unless a supported tool was used.\n"
        "- For booking drafts, use only the provided tool argument names: pickup_location, dropoff_location, date, time, journey_type, wait_time_mins, return_journey_date, return_journey_time, number_of_luggages, flight_number, booking_notes.\n"
        "- If the user is changing an existing draft, update only the fields they mentioned and retain every other previously collected field.\n"
        "- Keep text replies concise and helpful."
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


async def run_agent(history: list[dict], customer_context: dict, booking_context: dict) -> dict:
    """Run the dispatcher agent and return a validated response envelope."""
    context_message = {
        "role": "system",
        "content": (
            f"Customer context: {json.dumps(customer_context, ensure_ascii=True)}\n"
            f"Booking context: {json.dumps(booking_context, ensure_ascii=True)}"
        ),
    }

    try:
        response = await _get_client().chat.completions.create(
            model=settings.VLLM_MODEL,
            messages=[
                {"role": "system", "content": _build_agent_system_prompt()},
                context_message,
                *history,
            ],
            temperature=0.1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        log.info("Agent raw output: %s", raw)
        return _parse_agent_response(raw)
    except Exception:
        log.exception("Agent execution failed")
        return _fallback_agent_response()


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


def _parse_agent_response(raw: str) -> dict:
    cleaned = (
        raw.replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    parsed = _extract_tool_call(cleaned)
    if isinstance(parsed, dict):
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if tool in {"check_driver_status", "cancel_ride", "edit_booking", "draft_new_booking"} and isinstance(arguments, dict):
            return {"type": "tool_call", "tool": tool, "arguments": arguments}

    return _wrap_raw_text_response(cleaned)


def _extract_tool_call(raw: str):
    envelope = _extract_standard_tool_envelope(raw)
    if isinstance(envelope, dict):
        return envelope

    cancel_ride = _extract_cancel_ride_payload(raw)
    if isinstance(cancel_ride, dict):
        return {"tool": "cancel_ride", "arguments": cancel_ride}

    hallucinated = _extract_named_tool_arguments(raw)
    if isinstance(hallucinated, dict):
        return {"tool": "draft_new_booking", "arguments": hallucinated}

    raw_arguments = _extract_raw_booking_arguments(raw)
    if isinstance(raw_arguments, dict):
        return {"tool": "draft_new_booking", "arguments": raw_arguments}

    return None


def _extract_cancel_ride_payload(raw: str):
    wrapped_match = re.search(r'"cancel_ride"\s*:\s*(\{.*\})', raw, re.DOTALL)
    if wrapped_match:
        candidate = _extract_balanced_json(wrapped_match.group(1))
        if candidate:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and _looks_like_cancel_arguments(parsed):
                return parsed

    named_match = re.search(r'cancel_ride\s*(\{.*\})', raw, re.DOTALL)
    if named_match:
        candidate = _extract_balanced_json(named_match.group(1))
        if candidate:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and _looks_like_cancel_arguments(parsed):
                return parsed

    candidate = _extract_balanced_json(raw)
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            if "cancel_ride" in parsed and isinstance(parsed["cancel_ride"], dict):
                nested = parsed["cancel_ride"]
                if _looks_like_cancel_arguments(nested):
                    return nested
            if _looks_like_cancel_arguments(parsed):
                return parsed

    booking_id = _extract_booking_id(raw)
    if "cancel_ride" in raw.lower() and booking_id is not None:
        return {"booking_id": booking_id}

    return None


def _extract_standard_tool_envelope(raw: str):
    match = re.search(
        r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
        raw,
        re.DOTALL,
    )
    if not match:
        return None

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_named_tool_arguments(raw: str):
    match = re.search(r'draft_new_booking\s*(\{.*\})', raw, re.DOTALL)
    if not match:
        return None

    candidate = _extract_balanced_json(match.group(1))
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return parsed if _looks_like_draft_arguments(parsed) else None


def _extract_raw_booking_arguments(raw: str):
    candidate = _extract_balanced_json(raw)
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict) and "arguments" in parsed and isinstance(parsed.get("arguments"), dict):
        parsed = parsed["arguments"]

    return parsed if _looks_like_draft_arguments(parsed) else None


def _extract_balanced_json(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:idx + 1]
    return None


def _looks_like_draft_arguments(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False
    booking_keys = {
        "pickup_location",
        "dropoff_location",
        "date",
        "time",
        "journey_type",
        "wait_time_mins",
        "return_journey_date",
        "return_journey_time",
        "number_of_luggages",
        "flight_number",
        "booking_notes",
    }
    return any(key in parsed for key in booking_keys)


def _looks_like_cancel_arguments(parsed) -> bool:
    return isinstance(parsed, dict) and parsed.get("booking_id") is not None


def _fallback_agent_response() -> dict:
    return {
        "type": "text",
        "message": "I'm sorry, I didn't quite catch that. Could you repeat?",
    }


def _wrap_raw_text_response(raw: str) -> dict:
    message = raw.strip()
    if not message:
        return _fallback_agent_response()
    return {
        "type": "text",
        "message": message,
    }


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
