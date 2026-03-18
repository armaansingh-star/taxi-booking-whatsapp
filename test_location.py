from fastapi import FastAPI, Request
from twilio.rest import Client
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = FastAPI()

# ---------------- TWILIO CONFIG ----------------

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---------------- GOOGLE MAPS CONFIG ----------------

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# store sessions
sessions = {}

# ---------------- REVERSE GEOCODE ----------------

def reverse_geocode(lat, lng):

    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {
        "latlng": f"{lat},{lng}",
        "key": GOOGLE_MAPS_API_KEY
    }

    r = requests.get(url, params=params)
    data = r.json()

    if data["status"] == "OK":
        return data["results"][0]["formatted_address"]

    return None


# ---------------- VALIDATE DESTINATION ----------------

def geocode_address(address):

    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {
        "address": address,
        "key": GOOGLE_MAPS_API_KEY
    }

    r = requests.get(url, params=params)
    data = r.json()

    if data["status"] == "OK":

        location = data["results"][0]["geometry"]["location"]

        return {
            "address": data["results"][0]["formatted_address"],
            "lat": location["lat"],
            "lng": location["lng"]
        }

    return None


# ---------------- DISTANCE + TIME ----------------

def calculate_distance(origin, destination):

    url = "https://maps.googleapis.com/maps/api/directions/json"

    params = {
        "origin": origin,
        "destination": destination,
        "units": "imperial",   # miles
        "key": GOOGLE_MAPS_API_KEY
    }

    r = requests.get(url, params=params)
    data = r.json()

    if data["status"] == "OK":

        leg = data["routes"][0]["legs"][0]

        distance = leg["distance"]["text"]   # miles
        duration = leg["duration"]["text"]

        return distance, duration

    return None, None


# ---------------- WEBHOOK ----------------

@app.post("/webhook")
async def whatsapp_webhook(request: Request):

    form = await request.form()
    data = dict(form)

    print("\nTWILIO PAYLOAD:", data)

    user = data.get("From")

    if user not in sessions:
        sessions[user] = {
            "pickup_coords": None,
            "pickup_address": None
        }

    session = sessions[user]

    latitude = data.get("Latitude")
    longitude = data.get("Longitude")

    text = data.get("Body", "").strip()

    # ---------------- LOCATION MESSAGE ----------------

    if latitude and longitude:

        print("LOCATION RECEIVED:", latitude, longitude)

        address = reverse_geocode(latitude, longitude)

        session["pickup_coords"] = f"{latitude},{longitude}"
        session["pickup_address"] = address

        reply = (
            f"📍 Pickup location received\n\n"
            f"{address}\n\n"
            "Where would you like to go?"
        )

    # ---------------- DESTINATION TEXT ----------------

    elif session["pickup_coords"] and text:

        dest = geocode_address(text)

        if not dest:

            reply = "❌ I couldn't find that destination. Please try again."

        else:

            drop_coords = f"{dest['lat']},{dest['lng']}"
            drop_address = dest["address"]

            distance, duration = calculate_distance(
                session["pickup_coords"],
                drop_coords
            )

            map_link = f"https://www.google.com/maps/dir/{session['pickup_coords']}/{drop_coords}"

            reply = (
                "🚕 Trip Estimate\n\n"
                f"Pickup:\n{session['pickup_address']}\n\n"
                f"Drop:\n{drop_address}\n\n"
                f"Distance: {distance}\n"
                f"Estimated Time: {duration}\n\n"
                f"Route:\n{map_link}"
            )

            # reset for next test
            sessions[user] = {
                "pickup_coords": None,
                "pickup_address": None
            }

    else:

        reply = "Please share your pickup location 📍 first."

    # ---------------- SEND REPLY ----------------

    client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        to=user,
        body=reply
    )

    return {"status": "ok"}