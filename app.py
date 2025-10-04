
import os
import re
from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

# Optional: Gemini for NLU (fallback to keyword rules if not configured)
USE_GEMINI = False
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
except Exception:
    USE_GEMINI = False

app = Flask(__name__)

# ------- Basic Config (set via ENV or defaults for demo) -------
CAFE_NAME = os.getenv("CAFE_NAME", "Your Café")
STAFF_NUMBER = os.getenv("STAFF_NUMBER")  # e.g., "+9198XXXXXXXX"
HOURS = os.getenv("CAFE_HOURS", "Mon–Sun, 8 AM to 9 PM")
ADDRESS = os.getenv("CAFE_ADDRESS", "123, Main Street, Your City")
WIFI_INFO = os.getenv("WIFI_INFO", "Network: YOUR_CAFE_WIFI, Password: latte123")
MENU_LINK = os.getenv("MENU_LINK", "https://example.com/menu")

# How many misunderstandings before forwarding to a human
MAX_MISUNDERSTANDINGS = int(os.getenv("MAX_MISUNDERSTANDINGS", "2"))


def say_text(resp: VoiceResponse, text: str):
    # Voice + language tuned for India English; adjust as you like
    resp.say(text, voice="alice", language="en-IN")


def prompt_main_menu() -> VoiceResponse:
    resp = VoiceResponse()
    g = Gather(
        input="speech dtmf",
        action="/handle",
        method="POST",
        timeout=5,
        num_digits=1,
        language="en-IN",
        hints="reservation,book,booking,table,menu,vegan,hours,time,location,address,directions,wifi,order,speak to staff,agent,manager"
    )
    say_text(g, f"Hi, welcome to {CAFE_NAME}! "
                 "You can say things like 'book a table for two at 7 p.m. today', "
                 "or ask for 'today's menu' or 'opening hours'. "
                 "Or press 1 for reservations, 2 for menu, 3 for hours, 4 for location, 5 for Wi-Fi, 0 to talk to staff.")
    resp.append(g)
    # If nothing captured, loop back
    resp.redirect("/voice")
    return resp


@app.route("/voice", methods=["GET", "POST"])
def voice():
    return str(prompt_main_menu())


@app.route("/handle", methods=["POST"])
def handle():
    speech = request.values.get("SpeechResult", "") or ""
    digits = request.values.get("Digits", "") or ""
    misunderstandings = int(request.values.get("mis", "0"))

    # DTMF quick routes
    if digits == "1":
        return str(prompt_reservation())
    elif digits == "2":
        return say_and_back(f"You can see our menu here: {MENU_LINK}. Would you like me to send the link by SMS?", misunderstandings)
    elif digits == "3":
        return say_and_back(f"Our opening hours are {HOURS}. Anything else I can help with?", misunderstandings)
    elif digits == "4":
        return say_and_back(f"We are at {ADDRESS}. Shall I send a Google Maps link by SMS?", misunderstandings)
    elif digits == "5":
        return say_and_back(f"Here is the Wi-Fi information. {WIFI_INFO}. Anything else?", misunderstandings)
    elif digits == "0":
        return transfer_to_staff()

    # Speech routes
    if speech:
        intent = detect_intent(speech)
        if intent == "reservation":
            return str(prompt_reservation())
        if intent == "menu":
            return say_and_back(f"You can see our menu here: {MENU_LINK}. Want me to text you the link?", misunderstandings)
        if intent == "hours":
            return say_and_back(f"Our opening hours are {HOURS}.", misunderstandings)
        if intent == "location":
            return say_and_back(f"We are at {ADDRESS}.", misunderstandings)
        if intent == "wifi":
            return say_and_back(f"{WIFI_INFO}.", misunderstandings)
        if intent == "human":
            return transfer_to_staff()

    # fallback
    misunderstandings += 1
    if misunderstandings > MAX_MISUNDERSTANDINGS and STAFF_NUMBER:
        return transfer_to_staff()
    resp = VoiceResponse()
    say_text(resp, "Sorry, I didn't get that.")
    resp.redirect(f"/voice?mis={misunderstandings}")
    return str(resp)


def say_and_back(text: str, misunderstandings: int = 0):
    resp = VoiceResponse()
    say_text(resp, text)
    # after answering, offer main menu again
    resp.redirect(f"/voice?mis={misunderstandings}")
    return str(resp)


def prompt_reservation() -> VoiceResponse:
    resp = VoiceResponse()
    g = Gather(
        input="speech",
        action="/reserve",
        method="POST",
        timeout=7,
        language="en-IN"
    )
    say_text(g, "Great. Please say your booking like this: "
                "'Book a table for two, tomorrow at 7 p.m., under the name Priya'.")
    resp.append(g)
    resp.redirect("/voice")
    return resp


@app.route("/reserve", methods=["POST"])
def reserve():
    speech = request.values.get("SpeechResult", "") or ""
    data = extract_reservation(speech)

    if not data:
        resp = VoiceResponse()
        say_text(resp, "Sorry, I could not get the reservation details. Let's try again.")
        resp.redirect("/voice")
        return str(resp)

    # Confirmation
    resp = VoiceResponse()
    summary = (
        f"Let me confirm: {data.get('party_size', 'unknown')} people, "
        f"on {data.get('date_text', 'the selected date')} at {data.get('time_text','the selected time')} "
        f"under the name {data.get('name', 'unknown')}."
    )
    say_text(resp, summary + " If this is correct, say 'confirm'. To change, say 'change'.")
    g = Gather(input="speech", action="/confirm_booking", method="POST", timeout=5, language="en-IN")
    resp.append(g)
    # Store pending in session-like memory via Twilio Memory (not available) → include in URL
    # Minimal approach: encode details in query string
    state = encode_state(data)
    resp.redirect(f"/confirm_booking?state={state}")
    return str(resp)


@app.route("/confirm_booking", methods=["POST", "GET"])
def confirm_booking():
    speech = request.values.get("SpeechResult", "") or ""
    state = request.values.get("state", "")
    data = decode_state(state)

    resp = VoiceResponse()
    if "confirm" in (speech or "").lower():
        # TODO: store to your DB/Sheets here
        say_text(resp, "Awesome. Your table is booked. We look forward to seeing you!")
        # Optional: SMS confirmation, email, etc.
        return str(resp)
    else:
        say_text(resp, "Okay, let's restart.")
        resp.redirect("/voice")
        return str(resp)


def transfer_to_staff():
    resp = VoiceResponse()
    if STAFF_NUMBER:
        say_text(resp, "Connecting you to our staff. Please hold.")
        d = Dial(caller_id=os.getenv("TWILIO_CALLER_ID"))
        d.number(STAFF_NUMBER)
        resp.append(d)
    else:
        say_text(resp, "Sorry, no staff number is configured. Please try again later.")
    return str(resp)


# ---------- NLU Helpers ----------

def detect_intent(utterance: str) -> str:
    text = utterance.lower()
    # quick keyword rules as a fallback
    rules = [
        ("reservation", ["book", "reservation", "table", "reserve"]),
        ("menu", ["menu", "food", "special", "dish", "vegan", "gluten"]),
        ("hours", ["hour", "open", "close", "timing", "time do you open"]),
        ("location", ["location", "address", "where are you", "directions"]),
        ("wifi", ["wifi", "wi fi", "internet", "password"]),
        ("human", ["human", "staff", "agent", "manager", "speak to"]),        
    ]
    for intent, kws in rules:
        if any(k in text for k in kws):
            return intent

    # Optional: use Gemini for better detection
    if USE_GEMINI:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")  # fast + cheap
            prompt = (
                "Classify this café caller request into one of: reservation, menu, hours, location, wifi, human.\n"
                f"Utterance: {utterance}\n"
                "Return only the label."
            )
            out = model.generate_content(prompt)
            cand = (out.text or "").strip().lower()
            for label in ["reservation","menu","hours","location","wifi","human"]:
                if label in cand:
                    return label
        except Exception:
            pass

    return "unknown"


def extract_reservation(utterance: str) -> Optional[Dict]:
    # Very lightweight parser + optional Gemini enhancement
    data = {"raw": utterance}

    if USE_GEMINI:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Extract a café reservation from the text.\n"
                "Return JSON with keys: name, party_size (int), date_text, time_text, iso_datetime if you can.\n"
                "Text: " + utterance
            )
            out = model.generate_content(prompt)
            import json
            j = json.loads(out.text.strip("`\n "))
            return j
        except Exception:
            pass

    # Heuristic fallback
    m = re.search(r"(?:for|party of)\s*(\d+)", utterance, re.I)
    if m:
        data["party_size"] = int(m.group(1))
    name_m = re.search(r"(?:name (?:is|under)\s*|under the name\s*)([A-Za-z ]{2,})", utterance, re.I)
    if name_m:
        data["name"] = name_m.group(1).strip()

    # naive time/date capture
    time_m = re.search(r"(\d{1,2}\s*(?:am|pm))", utterance, re.I)
    if time_m:
        data["time_text"] = time_m.group(1)
    date_m = re.search(r"(today|tomorrow|on \w+\s*\d{0,2})", utterance, re.I)
    if date_m:
        data["date_text"] = date_m.group(1).strip()

    # minimal viability: need at least party_size and some time/date cue
    if "party_size" not in data or ("time_text" not in data and "date_text" not in data):
        return None
    return data


# ---- tiny state encoding (unsafe but OK for demo) ----
import base64
import json as _json

def encode_state(d: dict) -> str:
    try:
        s = _json.dumps(d)
        return base64.urlsafe_b64encode(s.encode()).decode()
    except Exception:
        return ""

def decode_state(s: str) -> dict:
    try:
        return _json.loads(base64.urlsafe_b64decode(s.encode()).decode())
    except Exception:
        return {}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
