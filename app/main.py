import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response

from app.config import settings
from app.conversation.handler import process_message
from app.conversation.session_store import cleanup_stale_sessions
from app.database import close_pool, init_pool
from app.listeners.assignment_listener import start_assignment_listener
from app.routers.notifications import router as notifications_router
from app.services import maps_service, transcription_service

import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log"),
    ],
)
log = logging.getLogger(__name__)

EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'

_background_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    log.info("Starting up...")
    await init_pool()
    transcription_service.load_whisper_model()

    _background_tasks.append(asyncio.create_task(start_assignment_listener()))
    _background_tasks.append(asyncio.create_task(cleanup_stale_sessions()))
    log.info("Background tasks started")

    yield

    # ── Shutdown ──
    log.info("Shutting down...")
    for task in _background_tasks:
        task.cancel()
    await maps_service.close_client()
    await close_pool()
    log.info("Shutdown complete")


app = FastAPI(title="Taxi Booking Bot", lifespan=lifespan)
app.include_router(notifications_router)


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """Twilio WhatsApp webhook.

    Returns empty TwiML immediately and processes the message
    in a background asyncio task via Twilio REST API.
    """
    form = await request.form()
    data = dict(form)
    asyncio.create_task(process_message(data))
    return Response(content=EMPTY_TWIML, media_type="application/xml")


@app.get("/health")
async def health():
    return {"status": "ok"}
