import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import httpx

from app.config import settings

log = logging.getLogger(__name__)

_whisper_model = None
_executor = ThreadPoolExecutor(max_workers=2)


def load_whisper_model():
    """Load faster-whisper model (call once at startup)."""
    global _whisper_model
    from faster_whisper import WhisperModel

    log.info("Loading Whisper model: %s", settings.WHISPER_MODEL)
    _whisper_model = WhisperModel(
        settings.WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
    )
    log.info("Whisper model loaded successfully")


def _sync_transcribe(file_path: str) -> str:
    """Synchronous transcription (runs in thread pool)."""
    segments, _ = _whisper_model.transcribe(file_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments)


async def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file using faster-whisper on GPU.

    Runs in a ThreadPoolExecutor to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(_executor, _sync_transcribe, file_path)
    log.info("Transcription result: %s", text)
    return text


async def download_media(media_url: str, save_dir: str = "audio") -> str:
    """Download media from Twilio and save to disk.

    Returns the saved file path.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"voice_{timestamp}.ogg")

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            media_url,
            auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            follow_redirects=True,
        )
        r.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(r.content)

    log.info("Media downloaded to %s", file_path)
    return file_path
