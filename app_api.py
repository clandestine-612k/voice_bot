"""
Real-time receptionist:
- /voice (Flask) -> returns TwiML to start Media Stream to wss://PUBLIC_HOST/media
- /media (websocket server) -> receives Twilio media events, forwards audio to Deepgram WS,
  receives transcripts, sends to Gemini, synthesizes reply (gTTS -> mp3), and plays into call
  by updating Twilio call TwiML to <Play> the mp3 URL.

Environment variables required:
- PUBLIC_HOST (e.g. https://voice.yourdomain.com)  (no trailing slash)
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- DEEPGRAM_API_KEY
- GEMINI_API_KEY
- CAFE_NAME (optional)
- STAFF_NUMBER (optional)
"""

import os
import asyncio
import base64
import json
import time
import tempfile
from pathlib import Path
from threading import Thread

from flask import Flask, request, Response, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client as TwilioRestClient
from gtts import gTTS
import websockets
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import aiofiles

load_dotenv()

# ---------- Config ----------
PUBLIC_HOST = os.getenv("PUBLIC_HOST")  # e.g. https://voice.yourcafe.com (required)
if not PUBLIC_HOST:
    raise SystemExit("Set PUBLIC_HOST env var (e.g. https://voice.yourcafe.com)")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
    raise SystemExit("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise SystemExit("Set DEEPGRAM_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # allow non-Gemini fallback, but best to set it
    print("WARNING: GEMINI_API_KEY not set — replies will be basic.")

CAFE_NAME = os.getenv("CAFE_NAME", "Your Café")
STAFF_NUMBER = os.getenv("STAFF_NUMBER", None)

# deepgram realtime websocket url (Twilio streams μ-law 8k by default)
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000"

# Ensure static dir exists
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# twilio client
twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# configure Gemini (if provided)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-1.5-flash"  # adjust as desired

# ---------- Flask app (for /voice TwiML and serving static audio) ----------
app = Flask(__name__, static_folder="static")


@app.route("/voice", methods=["GET","POST"])
def voice():
    """
    Twilio will POST here when a call arrives.
    Return TwiML that starts a Media Stream to wss://PUBLIC_HOST/media
    """
    resp = VoiceResponse()
    resp.say(f"Hi — welcome to {CAFE_NAME}. Connecting you to our receptionist.", voice="alice", language="en-IN")
    # Twilio expects wss URL (no trailing slash)
    stream_url = f"wss://{PUBLIC_HOST.replace('https://','').replace('http://','')}/media"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">Connecting you now.</Say>
    <Connect>
        <Stream url="{stream_url}"/>
    </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@app.route("/static/<path:filename>")
def static_file(filename):
    # serve generated audio files
    return send_from_directory(str(STATIC_DIR), filename)


# ---------- WebSocket media server (Twilio Media Streams) ----------
# We'll run a separate asyncio WebSocket server at /media and port 8765
# Twilio will connect to wss://PUBLIC_HOST/media

async def handle_twilio_media(ws, path):
    """
    Each connection is one call's media stream.
    Twilio sends events: 'start', 'media', 'stop'. For 'start' event we capture callSid.
    For 'media' events we get base64 mu-law audio payload. We forward this to Deepgram WS.
    When Deepgram returns a final transcript, we call Gemini for a reply, synthesize to mp3,
    store it at static/<filename>, then instruct Twilio to play it into the call using REST API.
    """
    dg_ws = None
    call_sid = None
    stream_sid = None

    try:
        # Open a Deepgram WS connection
        dg_ws = await websockets.connect(
            DEEPGRAM_WS_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        )

        # create an async task to listen to Deepgram messages
        async def read_deepgram():
            nonlocal dg_ws, call_sid
            async for dg_msg in dg_ws:
                try:
                    obj = json.loads(dg_msg)
                except Exception:
                    continue
                # Deepgram final transcription arrives as {"type":"Transcription", ... } or channel object.
                # check for transcript in obj
                transcript = None
                if "channel" in obj:
                    alt = obj["channel"].get("alternatives")
                    if alt and len(alt) > 0:
                        transcript = alt[0].get("transcript", "").strip()
                # Some Deepgram messages are partials; we only act on non-empty final transcripts
                if transcript:
                    print("[Deepgram] transcript:", transcript)
                    # Ask Gemini (if available) to generate reply
                    if GEMINI_API_KEY:
                        try:
                            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                            prompt = (
                                "You are a polite cafe receptionist. Keep replies short and actionable.\n"
                                f"Customer: {transcript}\nAssistant:"
                            )
                            out = model.generate_content(prompt)
                            reply_text = (out.text or "").strip()
                        except Exception as e:
                            print("Gemini error:", e)
                            reply_text = "Sorry, I couldn't process that. Could you repeat?"
                    else:
                        # fallback simple rule-based reply
                        if "hour" in transcript.lower() or "open" in transcript.lower():
                            reply_text = f"We are open {os.getenv('CAFE_HOURS','Mon–Sun, 8 AM to 9 PM')}."
                        elif "menu" in transcript.lower():
                            reply_text = f"Our menu is at {os.getenv('MENU_LINK','https://example.com/menu')}."
                        elif "reserve" in transcript.lower() or "book" in transcript.lower():
                            reply_text = "Sure — please tell me the number of people and time."
                        else:
                            reply_text = "Sorry, I didn't get that. Can you repeat?"

                    print("[Bot] reply:", reply_text)

                    # Synthesize reply to mp3 using gTTS (works on Render)
                    timestamp = int(time.time() * 1000)
                    filename = f"reply_{call_sid or 'call'}_{timestamp}.mp3"
                    out_path = STATIC_DIR / filename
                    try:
                        tts = gTTS(reply_text, lang="en")
                        tts.save(str(out_path))
                    except Exception as e:
                        print("gTTS error:", e)
                        continue

                    # Now instruct Twilio to play the file into the live call using REST API
                    if call_sid:
                        try:
                            public_url = f"{PUBLIC_HOST}/static/{filename}"
                            twiml_play = f"<Response><Play>{public_url}</Play></Response>"
                            twilio_client.calls(call_sid).update(twiml=twiml_play)
                            print(f"Injected audio into call {call_sid}: {public_url}")
                        except Exception as e:
                            print("Twilio play error:", e)
                    else:
                        print("No call_sid found — cannot inject audio.")

        read_task = asyncio.create_task(read_deepgram())

        # Read messages from Twilio and forward binary audio to Deepgram
        async for raw in ws:
            # Twilio sends text JSON messages
            data = json.loads(raw)
            event = data.get("event")
            if event == "start":
                start = data.get("start", {})
                stream_sid = start.get("streamSid") or start.get("streamId")
                call_sid = start.get("callSid") or start.get("CallSid") or call_sid
                print("Media start:", stream_sid, "call:", call_sid)
            elif event == "media":
                # media payload is base64 mu-law bytes
                media = data.get("media", {})
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue
                # Forward to Deepgram: send JSON with type Binary and audio (base64)
                await dg_ws.send(json.dumps({
                    "type": "Binary",
                    "audio": payload_b64
                }))
            elif event == "stop":
                print("Media stopped")
                break

        # Clean up
        if read_task:
            read_task.cancel()
        if dg_ws:
            await dg_ws.close()
    except Exception as e:
        print("WS handler error:", e)
        try:
            if dg_ws:
                await dg_ws.close()
        except:
            pass


# ---------- Run both Flask (HTTP) and the websocket server concurrently ----------
def start_ws_server():
    # websockets server listens on port 8765 and path /media
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_server = websockets.serve(handle_twilio_media, "0.0.0.0", 8765, ping_interval=20, ping_timeout=10)
    print("Starting websocket server on port 8765 (path /media)")
    loop.run_until_complete(ws_server)
    loop.run_forever()


if __name__ == "__main__":
    # Start WS server in background thread
    Thread(target=start_ws_server, daemon=True).start()

    # Start Flask (HTTP) — Twilio webhook uses /voice on port 5000
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
