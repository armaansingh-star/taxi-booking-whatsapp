import asyncio
import json
import logging

from twilio.rest import Client

from app.config import settings

log = logging.getLogger(__name__)

_twilio_client: Client | None = None


def _get_client() -> Client:
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    return _twilio_client


async def send(to: str, body: str):
    """Send a WhatsApp message via Twilio REST API (non-blocking)."""
    try:
        await asyncio.to_thread(
            _get_client().messages.create,
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=to,
            body=body,
        )
        log.info("Message sent to %s", to)
    except Exception:
        log.exception("Failed to send message to %s", to)


async def send_template(destination: str, content_sid: str, variables: dict):
    """Send a WhatsApp template message via Twilio Content API."""
    try:
        # Force all variables to strings to prevent JSON nulls (Error 21656)
        safe_variables = {str(k): str(v) if v is not None else "None" for k, v in variables.items()}

        # Deep inspection log so we can see exactly what Twilio is receiving
        log.info("Attempting to send template %s to %s. Payload: %s", content_sid, destination, json.dumps(safe_variables))

        await asyncio.to_thread(
            _get_client().messages.create,
            from_=settings.TWILIO_WHATSAPP_NUMBER,
            to=destination,
            content_sid=content_sid,
            content_variables=json.dumps(safe_variables),
        )
        log.info("Successfully dispatched template %s to %s", content_sid, destination)
    except Exception as e:
        log.error("Twilio API Exception when sending template %s to %s: %s", content_sid, destination, str(e))
