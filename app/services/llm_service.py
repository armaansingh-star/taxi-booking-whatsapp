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


_DRIVER_KEYWORDS = {"driver", "car", "vehicle", "who", "assigned", "plate", "registration"}
_GREETING_KEYWORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}


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
