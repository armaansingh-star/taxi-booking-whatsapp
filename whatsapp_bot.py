from fastapi import FastAPI, Request
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = "my_verify_token"


# ---------------- VERIFY WEBHOOK ----------------
@app.get("/webhook")
async def verify(request: Request):

    params = dict(request.query_params)

    if (
        params.get("hub.mode") == "subscribe" and
        params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        return int(params.get("hub.challenge"))

    return "Verification failed"


# ---------------- RECEIVE MESSAGE ----------------
@app.post("/webhook")
async def receive_message(request: Request):

    data = await request.json()
    print("INCOMING:", data)

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender = message["from"]

        # ---- TEXT ----
        if message["type"] == "text":
            user_text = message["text"]["body"]

            reply = f"Bot reply: {user_text}"

        # ---- LOCATION ----
        elif message["type"] == "location":
            lat = message["location"]["latitude"]
            lng = message["location"]["longitude"]

            reply = f"📍 Got location: {lat}, {lng}"

        # ---- AUDIO ----
        elif message["type"] == "audio":
            reply = "🎤 Got your voice note!"

        else:
            reply = "Unsupported message"

        send_reply(sender, reply)

    except Exception as e:
        print("ERROR:", e)

    return {"status": "ok"}


# ---------------- SEND REPLY ----------------
def send_reply(to, text):

    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    response = requests.post(url, headers=headers, json=payload)

    print("REPLY STATUS:", response.status_code)
    print("REPLY RESPONSE:", response.json())