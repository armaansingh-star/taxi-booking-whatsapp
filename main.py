# In-memory booking store (keyed by WhatsApp number)
booking_sessions = {}


from fastapi import FastAPI, Request
from twilio.rest import Client
from dotenv import load_dotenv
import os


load_dotenv()
app = FastAPI()

# Twilio credentials (sandbox works too)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

print("DEBUG SID:", TWILIO_ACCOUNT_SID)
print("DEBUG TOKEN LOADED:", TWILIO_AUTH_TOKEN is not None)

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)



import requests
from datetime import datetime

def download_twilio_audio(media_url: str, save_dir="audio"):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"{save_dir}/voice_{timestamp}.ogg"

    response = requests.get(
        media_url,
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    )

    if response.status_code != 200:
        raise Exception(f"Failed to download audio: {response.status_code}")

    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path





#email parser function


from email import policy
from email.parser import BytesParser

def parse_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    subject = msg['subject']
    sender = msg['from']

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()

    return {
        "subject": subject,
        "from": sender,
        "body": body
    }




def clean_email_text(text):
    stop_phrases = [
        "Sent from",
        "Disclaimer",
        "This e-mail message is confidential",
        "Registered Office Address",
        "Get Outlook for iOS"
    ]

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        if any(p.lower() in line.lower() for p in stop_phrases):
            break
        cleaned.append(line)

    return "\n".join(cleaned).strip()














from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-mini-transcribe"
        )
    return transcription.text


def extract_booking_info(text: str) -> dict:
    prompt = f"""
Extract taxi booking information from the text below.

IMPORTANT RULES:
- DO NOT convert relative dates into calendar dates.
- If the user says "tomorrow", return exactly "tomorrow".
- If the user says "next Monday", return exactly "next Monday".
- Return ONLY what the user explicitly said.

Return ONLY valid JSON.
Do NOT include explanations, markdown, or extra text.

The JSON must have exactly these keys:
- pickup
- drop
- date
- time
- passengers

If a field is not mentioned, use null.

Text:
{text}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()
    print("RAW LLM OUTPUT:", raw_output)

    import json
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # hard fallback (never crash the bot)
        return {
            "pickup": None,
            "drop": None,
            "date_raw": None,
            "date_resolved": None,
            "date_confirmed": False,
            "time": None,
            "passengers": None
        }




# ================= EMAIL BOOKING EXTRACTION =================

def extract_email_bookings(email_text: str) -> dict:
    prompt = f"""
You are extracting taxi booking requests from an email.

IMPORTANT:
- One email may contain MULTIPLE taxi trips.
- Extract ALL trips mentioned.
- DO NOT guess missing information.
- Ignore email signatures and disclaimers.

For each trip extract:
- trip_type (outward / return / single)
- pickup
- drop
- date
- pickup_time
- flight_number
- passengers
- notes

Return ONLY valid JSON in this format:

{{
  "customer_name": "",
  "contact_email": "",
  "trips": []
}}

Email:
<<<
{email_text}
>>>
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    import json
    return json.loads(response.choices[0].message.content)





REQUIRED_FIELDS = ["pickup", "drop", "date_raw", "time", "passengers"]

def merge_booking(old: dict, new: dict) -> dict:
    for k in REQUIRED_FIELDS:
        if old.get(k) is None and new.get(k) is not None:
            old[k] = new[k]
    return old

def get_missing_fields(booking: dict):
    return [k for k in REQUIRED_FIELDS if booking.get(k) is None]

def get_next_missing_field(session: dict):
    if session.get("pickup") is None:
        return "pickup"
    if session.get("drop") is None:
        return "drop"
    if session.get("date_raw") is None:
        return "date"
    if session.get("time") is None:
        return "time"
    if session.get("passengers") is None:
        return "passengers"
    return None

def question_for_field(field: str) -> str:
    questions = {
        "pickup": "Where should I pick you up from?",
        "drop": "Where would you like to go?",
        "date": "What date would you like to travel?",
        "time": "What time would you like to travel?",
        "passengers": "How many passengers will be travelling?"
    }
    return questions[field]



# ================= EMAIL → SESSION CONVERTER =================

def email_trips_to_sessions(email_data: dict):
    sessions = []

    for trip in email_data.get("trips", []):
        session = {
            "name": email_data.get("customer_name"),
            "pickup": trip.get("pickup"),
            "drop": trip.get("drop"),
            "date_raw": trip.get("date"),
            "date_resolved": normalize_date(trip.get("date")),
            "date_confirmed": False,
            "time": trip.get("pickup_time"),
            "passengers": trip.get("passengers"),
            "status": "awaiting_confirmation",
            "notes": trip.get("notes"),
            "flight_number": trip.get("flight_number")
        }
        sessions.append(session)

    return sessions


def reset_field(session: dict, field: str):
    if field in session:
        session[field] = None







from datetime import datetime, timedelta
from dateutil import parser
import pytz

def normalize_date(date_str: str, timezone="Europe/London"):
    if not date_str:
        return None

    tz = pytz.timezone(timezone)
    now = datetime.now(tz)

    date_str = date_str.lower().strip()

    if date_str == "today":
        return now.date().isoformat()

    if date_str == "tomorrow":
        return (now + timedelta(days=1)).date().isoformat()

    if date_str == "day after tomorrow":
        return (now + timedelta(days=2)).date().isoformat()

    try:
        parsed = parser.parse(date_str, fuzzy=True, default=now)
        return parsed.date().isoformat()
    except Exception:
        return None







@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    data = dict(form)

    from_number = data.get("From")
    num_media = int(data.get("NumMedia", 0))

    # ---- SESSION INIT (INSIDE WEBHOOK) ----
    user_id = from_number

    if user_id not in booking_sessions:
        booking_sessions[user_id] = {
            "name": None,
            "pickup": None,
            "drop": None,
            "date_raw": None,
            "date_resolved": None,
            "date_confirmed": False,
            "time": None,
            "passengers": None,
            "status": "collecting"
        }

    session = booking_sessions[user_id]
    # ---- BACKWARD SAFE SESSION PATCH ----
    session.setdefault("name", None)
    session.setdefault("pickup", None)
    session.setdefault("drop", None)
    session.setdefault("date_raw", None)
    session.setdefault("date_resolved", None)
    session.setdefault("date_confirmed", False)
    session.setdefault("time", None)
    session.setdefault("passengers", None)
    session.setdefault("status", "collecting")

    booking_sessions[user_id] = session
    reply_text = "Got it 👍 Please continue."


    
    # ---- ASK FOR NAME FIRST (RIGHT AFTER SESSION INIT) ----
    if session["name"] is None:
        # If user sent text, treat it as their name
        if num_media == 0:
            name_text = data.get("Body", "").strip()

            if len(name_text)>2 and name_text.lower() not in ["hi", "hello", "hey"]:
                session["name"] = name_text
                booking_sessions[user_id] = session
                reply_text = f"Thanks {session['name']}! Let’s book your ride 🚕"
            else:
                reply_text = "May I have your name please?"

        # If user sent voice, ask explicitly
        else:
            reply_text = "May I have your name please?"

        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            to=from_number,
            body=reply_text
        )
        return {"status": "ok"}





    # ---- VOICE PATH ----
    if num_media > 0:
        media_url = data.get("MediaUrl0")
        media_type = data.get("MediaContentType0", "")

        if "message/rfc822" in media_type or "application/octet-stream" in media_type:
            try:
                eml_path = download_twilio_audio(media_url, save_dir="emails")
                email = parse_eml(eml_path)
                cleaned_text = clean_email_text(email["body"])

                email_data = extract_email_bookings(cleaned_text)
                email_sessions = email_trips_to_sessions(email_data)

                session["email_queue"] = email_sessions[1:]
                session.update(email_sessions[0])
                session["status"] = "awaiting_confirmation"
                booking_sessions[user_id] = session

                reply_text = (
                    "📧 I’ve received a booking request from the email.\n\n"
                    f"Pickup: {session['pickup']}\n"
                    f"Drop: {session['drop']}\n"
                    f"Date: {session['date_resolved']}\n"
                    f"Time: {session['time']}\n"
                    f"Passengers: {session['passengers']}\n\n"
                    "Reply YES to confirm or NO to make changes."
                )
                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=from_number,
                    body=reply_text
                )
                return {"status": "ok"}

            except Exception as e:
                print("EMAIL ERROR:", e)
                reply_text = "Sorry, I couldn't process that email."


        elif "audio" in media_type:
            try:
                audio_path = download_twilio_audio(media_url)
                print("AUDIO SAVED AT:", audio_path)

                text = transcribe_audio(audio_path)
                print("TRANSCRIPTION:", text)

                booking_update = extract_booking_info(text)
                print("BOOKING UPDATE:", booking_update)

                session = merge_booking(session, booking_update)
                booking_sessions[user_id] = session

            # ---- DATE NORMALIZATION (VOICE PATH) ----
                if booking_update.get("date") and session["date_raw"] is None:
                    session["date_raw"] = booking_update["date"]
                    session["date_resolved"] = normalize_date(booking_update["date"])
                    session["date_confirmed"] = False
                    booking_sessions[user_id] = session
                
                
                missing = get_missing_fields(session)
                print("CURRENT SESSION:", session)

                if missing:
                    reply_text = question_for_field(missing[0])

                    client.messages.create(
                        from_=TWILIO_WHATSAPP_NUMBER,
                        to=from_number,
                        body=reply_text
                    )
                    return {"status": "ok"}
                

                session["status"] = "awaiting_confirmation"  
                booking_sessions[user_id] = session
                reply_text = (
                    "Just confirming your booking 🚕\n"
                    f"Pickup: {session['pickup']}\n"
                    f"Drop: {session['drop']}\n"
                    f"Date: {session['date_resolved']}\n"
                    f"Time: {session['time']}\n"
                    f"Passengers: {session['passengers']}\n\n"
                    "Shall I book this?\n"
                    "Reply YES to confirm or NO to make changes."
                )

            except Exception as e:
                print("ERROR:", e)
                reply_text = "Sorry, I couldn’t process that voice message."



    # ---- TEXT PATH (for replies like 'tomorrow', 'yes', etc.) ----
    else:
        text = data.get("Body", "").strip()
        text_lower = text.lower()

        #Confirmation handler----
        if session["status"] == "awaiting_confirmation":
            if text_lower == "yes":
                reply_text = (
                    f"Thanks {session['name']}! 🚕\n\n"
                    "Your ride has been successfully booked."
                )

                if session.get("email_queue"):
                    next_trip = session["email_queue"].pop(0)
                    session.update(next_trip)
                    session["status"] = "awaiting_confirmation"

                    reply_text += (
                        "\n\n📧 Next trip from the email:\n\n"
                        f"Pickup: {session['pickup']}\n"
                        f"Drop: {session['drop']}\n"
                        f"Date: {session['date_resolved']}\n"
                        f"Time: {session['time']}\n"
                        f"Passengers: {session['passengers']}\n\n"
                        "Reply YES to confirm or NO to make changes."
                    )
                else:
                    session["status"] = "collecting"
                    session["pickup"] = None
                    session["drop"] = None
                    session["date_raw"] = None
                    session["date_resolved"] = None
                    session["time"] = None
                    session["passengers"] = None
                     
                booking_sessions[user_id] = session
                
                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=from_number,
                    body=reply_text
                )
                return {"status": "ok"}

            elif text_lower == "no":
                session["status"] = "editing"
                booking_sessions[user_id] = session

                reply_text = (
                    "No problem 👍\n"
                    "What would you like to change? "
                    "(pickup / drop / date / time / passengers)"
                )

                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=from_number,
                    body=reply_text
                )
                return {"status": "ok"}
                
            else:
                reply_text = "Please reply YES to confirm or NO to make changes."

                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=from_number,
                    body=reply_text
                )
                return {"status": "ok"}



        if (
            session["name"] is not None
            and session["pickup"] is None
            and text.lower() in ["hi", "hello", "hey"]
        ):
            reply_text = f"Hey {session['name']}! Let’s book another ride 🚕"
            client.messages.create(
                from_=TWILIO_WHATSAPP_NUMBER,
                to=from_number,
                body=reply_text
            )
            return {"status": "ok"}
        

        # --- EDIT MODE HANDLER ---
        if session["status"] == "editing":
            field = text.lower().strip()

            if field in ["pickup", "drop", "date", "time", "passengers"]:
                if field == "date":
                    session["date_raw"] = None
                    session["date_resolved"] = None
                    session["date_confirmed"] = False
                else:
                    session[field] = None
         
                session["status"] = "collecting"
                booking_sessions[user_id] = session

                reply_text = question_for_field(field)

                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=from_number,
                    body=reply_text
                )
                return {"status": "ok"}

            else:
                reply_text = (
                    "Please choose one of these to change:\n"
                    "pickup / drop / date / time / passengers"
                )

                client.messages.create(
                    from_=TWILIO_WHATSAPP_NUMBER,
                    to=from_number,
                    body=reply_text
                )
                return {"status": "ok"}






        booking_update = extract_booking_info(text)
        session = merge_booking(session, booking_update)
        booking_sessions[user_id] = session
        
        if booking_update.get("date") and session["date_raw"] is None:
            session["date_raw"] = booking_update["date"]
            session["date_resolved"] = normalize_date(booking_update["date"])
            session["date_confirmed"] = False
            booking_sessions[user_id] = session



        # ask next missing field
        next_field = get_next_missing_field(session)

        if next_field:
            reply_text = question_for_field(next_field)
        else:

            session["status"] = "awaiting_confirmation"  
            booking_sessions[user_id] = session
            reply_text = (
                "Just confirming your booking 🚕\n"
                f"Pickup: {session['pickup']}\n"
                f"Drop: {session['drop']}\n"
                f"Date: {session['date_resolved']}\n"
                f"Time: {session['time']}\n"
                f"Passengers: {session['passengers']}\n\n"
                "Shall I book this?\n"
                "Reply YES to confirm or NO to make changes."
            )



    # ---- SEND WHATSAPP REPLY ----
    client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        to=from_number,
        body=reply_text
    )

    return {"status": "ok"}
