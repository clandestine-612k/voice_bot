import os
import json
import base64
import asyncio
import websockets
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv
import pyttsx3
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)

# ------------------ Gemini ------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ pyttsx3 TTS ------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speaking speed
engine.setProperty('voice', 'english')  # choose voice

def synthesize_tts(text):
    """
    Returns base64-encoded WAV audio from text using pyttsx3
    """
    import tempfile, soundfile as sf
    import numpy as np
    # Save to temp WAV
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    engine.save_to_file(text, tmpfile.name)
    engine.runAndWait()
    # Read WAV and encode base64
    data, samplerate = sf.read(tmpfile.name)
    audio_bytes = data.tobytes()
    encoded = base64.b64encode(audio_bytes).decode()
    return encoded

# ------------------ Cafe Config ------------------
CAFE_NAME = os.getenv("CAFE_NAME", "Your Cafe")
STAFF_NUMBER = os.getenv("STAFF_NUMBER")
HOURS = os.getenv("CAFE_HOURS", "Mon–Sun, 8 AM to 9 PM")
ADDRESS = os.getenv("CAFE_ADDRESS", "123 Main Street")
WIFI_INFO = os.getenv("WIFI_INFO", "Network: YOUR_WIFI, Password: latte123")
MENU_LINK = os.getenv("MENU_LINK", "https://example.com/menu")
TWILIO_CALLER_ID = os.getenv("TWILIO_CALLER_ID")
MAX_MISUNDERSTANDINGS = 2

# ------------------ Twilio Voice webhook ------------------
@app.route("/voice", methods=["POST"])
def voice():
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">Welcome to {CAFE_NAME}. Connecting to our voice bot now.</Say>
    <Connect>
        <Stream url="wss://your-server.com/media"/>
    </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")

# ------------------ WebSocket Media Stream Handler ------------------
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_WS = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000"

async def media_ws(twilio_ws):
    async with websockets.connect(
        DEEPGRAM_WS,
        extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    ) as dg_ws:

        async def forward_audio():
            async for msg in twilio_ws:
                data = json.loads(msg)
                if data.get("event") == "media":
                    audio_chunk = base64.b64decode(data["media"]["payload"])
                    await dg_ws.send(json.dumps({
                        "type": "Binary",
                        "audio": base64.b64encode(audio_chunk).decode()
                    }))

        async def handle_transcripts():
            async for msg in dg_ws:
                res = json.loads(msg)
                if "channel" in res:
                    transcript = res["channel"]["alternatives"][0].get("transcript")
                    if transcript:
                        # Gemini generates response
                        gemini_resp = GEMINI_MODEL.generate_content(
                            f"Cafe bot reply to: {transcript}"
                        ).text
                        # Convert to TTS via pyttsx3
                        audio_b64 = synthesize_tts(gemini_resp)
                        # Send back to Twilio
                        reply = json.dumps({
                            "event": "media",
                            "media": {"payload": audio_b64}
                        })
                        await twilio_ws.send(reply)

        await asyncio.gather(forward_audio(), handle_transcripts())

# ------------------ Run Flask ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
