# Taxi Booking WhatsApp Backend

## 1. Project Overview
This repository contains a FastAPI backend for a WhatsApp Taxi Booking Bot. It receives inbound WhatsApp messages from Twilio, uses a local vLLM-served Qwen model plus faster-whisper for text and voice understanding, calculates fares with Google Maps, stores bookings directly in PostgreSQL, and sends operational notifications to customers and drivers.

The backend is designed to support both conversational booking intake and operational workflows such as driver assignment alerts, booking update notifications, and secured dispatch triggers for frontend/admin tooling.

## 2. Architecture & Key Features
### High-level architecture
- FastAPI provides the HTTP surface area for Twilio webhooks, health checks, and secured notification endpoints.
- `asyncpg` provides a shared PostgreSQL connection pool for direct reads/writes into the raw operational schemas.
- A local vLLM endpoint serves `Qwen/Qwen2.5-7B-Instruct` for booking extraction, intent routing, and agent-style follow-up logic.
- `faster-whisper` runs locally for WhatsApp voice note transcription.
- Google Maps Directions/Geocoding APIs provide route validation, distance, duration, and fare inputs.
- A background asyncio poller watches `transform_db.booking_assignments` for newly assigned rides and triggers automated notifications.

### Key backend features
- Async request handling with immediate webhook acknowledgment and background message processing.
- In-memory conversational session state with draft booking retention and cleanup of stale sessions.
- Agent-style LLM orchestration for collecting booking fields one step at a time.
- Direct PostgreSQL writes into `ingest_db.customers` and `ingest_db.bookings`.
- Polling-based assignment detection instead of DB triggers or LISTEN/NOTIFY.
- Twilio Content API template sending for notification workflows that need to bypass the 24-hour WhatsApp session window.
- Protected operational notification endpoints secured with `X-API-Key`.

### Booking logic highlights
- Supports `One Way`, `Round Trip`, and `Return`.
- `Return` journeys are stored as two booking rows in the handler layer.
- `Round Trip` uses wait-time aware fare billing.
- Airport trips can collect flight number and luggage details.
- Booking notes are collected as the final optional field before draft execution.

## 3. Project Structure (Directory Map)
```text
app/
├── __init__.py
├── config.py
├── database.py
├── main.py
├── conversation/
│   ├── __init__.py
│   ├── handler.py
│   └── session_store.py
├── listeners/
│   ├── __init__.py
│   └── assignment_listener.py
├── models/
│   ├── __init__.py
│   ├── booking.py
│   ├── customer.py
│   └── session.py
├── routers/
│   ├── __init__.py
│   └── notifications.py
└── services/
    ├── __init__.py
    ├── booking_service.py
    ├── customer_service.py
    ├── llm_service.py
    ├── maps_service.py
    ├── messaging_service.py
    └── transcription_service.py
```

### Major files and responsibilities
#### `app/main.py`
FastAPI entry point. It initializes the DB pool and Whisper model on startup, starts background tasks, mounts the secured notifications router, exposes `/health`, and accepts Twilio WhatsApp webhook traffic on `/webhook`.

#### `app/config.py`
Pydantic settings layer. It validates required environment variables at process startup and provides a single `settings` object used across Twilio, database, Google Maps, LLM, Whisper, and API security configuration.

#### `app/database.py`
Central async PostgreSQL pooling module. It creates and closes a shared `asyncpg` pool and exposes `get_pool()` for all services and background workers.

#### `app/conversation/handler.py`
Core booking orchestration engine. It manages onboarding, session state, LLM interactions, Twilio input normalization, booking draft evolution, fare calculation handoff, confirmation handling, editing, status checks, cancellations, and final DB inserts.

#### `app/conversation/session_store.py`
In-memory session registry keyed by phone number. It creates session objects on demand and runs a periodic cleanup loop for stale sessions.

#### `app/models/session.py`
Conversation state definitions and the session data model. It stores booking draft state, fare state, customer metadata, active edit references, and timestamps.

#### `app/models/booking.py`
Booking schemas used throughout the backend. `BookingExtraction` represents LLM-collected raw fields, `BookingCreate` represents DB-ready rows, and `FareEstimate` represents route/fare outputs.

#### `app/models/customer.py`
Pydantic models for customer lookup and creation payloads used when reading/writing `ingest_db.customers`.

#### `app/services/customer_service.py`
Customer lookup and customer creation logic. It normalizes phone numbers, finds existing customers, and inserts new customer records with default active status.

#### `app/services/booking_service.py`
Booking persistence and booking-query module. It inserts bookings, updates editable bookings, checks assignment state, fetches booking status, and performs cancellation on assigned rides.

#### `app/services/llm_service.py`
LLM integration layer. It handles raw field extraction, agent prompt construction, tolerant tool-call parsing, status/edit/cancel tool routing, and short conversational responses.

#### `app/services/maps_service.py`
Google Maps integration and fare engine. It performs geocoding, reverse geocoding, route distance lookups, and applies the project’s deadhead-based fare rules for One Way, Round Trip, and Return flows.

#### `app/services/transcription_service.py`
Voice note ingestion pipeline. It downloads Twilio media, loads the local faster-whisper model, and transcribes WhatsApp audio without blocking the event loop.

#### `app/services/messaging_service.py`
Twilio messaging wrapper. It sends normal WhatsApp text messages and Twilio Content API templates, and it string-sanitizes template variables to avoid JSON null/type errors in Twilio payloads.

#### `app/routers/notifications.py`
Secured operational notification API for the frontend/admin layer. It exposes manual dispatch, driver notification, and customer notification endpoints behind an `X-API-Key` guard.

#### `app/listeners/assignment_listener.py`
Async polling worker watching `transform_db.booking_assignments` for newly assigned rides. It pushes automated assignment notifications to customers and drivers.

## 4. Environment Setup (`.env` Variables)
Create a `.env` file in the repository root. Do not commit real secrets.

### Twilio
- `TWILIO_ACCOUNT_SID`
  Twilio account SID used by the Python SDK.
- `TWILIO_AUTH_TOKEN`
  Twilio auth token used to send WhatsApp messages and download media.
- `TWILIO_WHATSAPP_NUMBER`
  The WhatsApp-enabled Twilio sender number, typically in `whatsapp:+1415...` format.

### API Security
- `API_SECRET_KEY`
  Shared secret required by all `/api/notifications/*` endpoints through the `X-API-Key` header.

### Google Maps
- `GOOGLE_MAPS_API_KEY`
  Used for geocoding, reverse geocoding, and route distance calculations.

### PostgreSQL
- `DB_HOST`
  PostgreSQL host.
- `DB_PORT`
  PostgreSQL port, usually `5432`.
- `DB_NAME`
  Operational database name.
- `DB_USER`
  Database username.
- `DB_PASSWORD`
  Database password.

### Local LLM / vLLM
- `VLLM_BASE_URL`
  OpenAI-compatible base URL for the local vLLM server, for example `http://localhost:8015/v1`.
- `VLLM_MODEL`
  Model identifier served by vLLM, currently `Qwen/Qwen2.5-7B-Instruct`.

### Local Whisper
- `WHISPER_MODEL`
  Faster-whisper model name, currently expected to be something like `large-v3-turbo`.

### Example `.env` skeleton
```env
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
API_SECRET_KEY=replace_with_a_long_random_secret

GOOGLE_MAPS_API_KEY=your_google_maps_key

DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password

VLLM_BASE_URL=http://localhost:8015/v1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

WHISPER_MODEL=large-v3-turbo
```

## 5. Running the Pipeline Locally
### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the local vLLM server
The repository includes a helper script:

```bash
chmod +x start_vllm.sh
./start_vllm.sh
```

Current script behavior:
- serves `Qwen/Qwen2.5-7B-Instruct`
- binds to `0.0.0.0:8015`
- uses `float16`
- reserves GPU headroom for faster-whisper

### 4. Run the FastAPI server
```bash
python run.py
```

This starts Uvicorn on:
- `http://0.0.0.0:8014`

### 5. Health check
```bash
curl http://localhost:8014/health
```

Expected response:
```json
{"status":"ok"}
```

### 6. Twilio webhook configuration
Point your Twilio WhatsApp webhook at:

```text
POST /webhook
```

The backend immediately returns empty TwiML and processes the message in a background task.

## 6. API Security & Usage
All notification endpoints under `/api/notifications/*` are protected by an API key.

### Required header
```text
X-API-Key: <your API_SECRET_KEY>
```

If the header is missing or incorrect, the API returns:
- `403 Forbidden`
- `{"detail":"Could not validate API credentials"}`

### Example: dispatch tomorrow
```bash
curl -X POST "http://localhost:8014/api/notifications/dispatch-tomorrow" \
  -H "X-API-Key: your_api_secret_key"
```

### Example: notify a customer
```bash
curl -X POST "http://localhost:8014/api/notifications/booking/2383/customer" \
  -H "X-API-Key: your_api_secret_key"
```

### Example: notify a driver
```bash
curl -X POST "http://localhost:8014/api/notifications/booking/2383/driver" \
  -H "X-API-Key: your_api_secret_key"
```

## Booking and Notification Data Flow
### Inbound booking flow
1. Twilio sends a WhatsApp webhook to `/webhook`.
2. FastAPI returns empty TwiML immediately.
3. `process_message()` loads or creates the session.
4. Voice notes are transcribed if media is present.
5. The LLM extracts fields or returns a tool call.
6. `handler.py` merges the draft, geocodes the route, calculates fare, and asks for confirmation.
7. On confirmation, the booking is inserted into `ingest_db.bookings`.
8. If the journey type is `Return`, the handler inserts two booking rows.

### Assignment flow
1. `assignment_listener.py` polls `transform_db.booking_assignments`.
2. New assignment rows are joined with driver, vehicle, and booking data.
3. The customer receives a direct assignment message.
4. The driver receives a Twilio template notification.

### Manual notification flow
1. Frontend/admin calls `/api/notifications/*` with `X-API-Key`.
2. The router queries live DB state.
3. The backend dispatches Twilio Content API templates to customers or drivers.

## Notification Endpoints
### `POST /api/notifications/dispatch-tomorrow`
Sends one driver-facing template per assigned ride scheduled for tomorrow.

### `POST /api/notifications/booking/{booking_id}/customer`
Sends a customer update for a specific booking, including current assigned-driver details if available.

### `POST /api/notifications/booking/{booking_id}/driver`
Sends a driver assignment/update template for a specific booking.

## Logging and Debugging
Logs are written to:
- `logs/app.log`
- `logs/convo_<phone>.jsonl`

### What to look for
- startup failures: DB pool, Whisper load, or missing env vars
- Twilio delivery/template errors
- LLM raw output and parsing behavior
- fare calculation warnings
- assignment poller activity

## Operational Notes
- The backend reads and writes directly to raw operational schemas, not abstracted views.
- Driver assignment detection is polling-based, not trigger-based.
- Notification APIs are secured, but Twilio webhook ingress is intentionally public.
- The agent layer is stateful in memory. If the process restarts, active chat memory resets.
- Twilio template payload variables are coerced to strings before dispatch to avoid invalid JSON/null issues.

## Recommended Team Handoff Notes
### Frontend team
- Use `/api/notifications/*` only with `X-API-Key`.
- Expect booking confirmation, fare calculation, and booking insertion to remain backend-owned.
- Notification endpoints are operational triggers, not the conversational booking API.

### Data team
- Core write targets are `ingest_db.customers` and `ingest_db.bookings`.
- Assignment state is read from `transform_db.booking_assignments`.
- Return journeys are persisted as two booking rows.
- Notification and status lookups use latest assignment rows joined via `booking_id`.
