import asyncio
import logging
from datetime import datetime, timedelta

from app.models.session import ConversationSession

log = logging.getLogger(__name__)

_sessions: dict[str, ConversationSession] = {}

SESSION_TTL_HOURS = 24


def get_session(phone: str) -> ConversationSession:
    """Get or create a conversation session for a phone number."""
    if phone not in _sessions:
        _sessions[phone] = ConversationSession(customer_phone=phone)
    session = _sessions[phone]
    session.touch()
    return session


def clear_session(phone: str):
    """Remove a session entirely."""
    _sessions.pop(phone, None)


async def cleanup_stale_sessions():
    """Background task: remove sessions older than SESSION_TTL_HOURS every hour."""
    while True:
        await asyncio.sleep(3600)
        cutoff = datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)
        stale = [k for k, v in _sessions.items() if v.updated_at < cutoff]
        for k in stale:
            del _sessions[k]
        if stale:
            log.info("Cleaned up %d stale sessions", len(stale))
