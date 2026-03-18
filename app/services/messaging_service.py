import asyncio
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
