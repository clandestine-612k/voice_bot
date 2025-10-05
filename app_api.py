"""
Real-time receptionist:
- /voice (Flask) -> returns TwiML to start Media Stream to wss://PUBLIC_HOST/media
- /media (websocket server) -> receives Twilio media events, forwards audio to Deepgram WS,
  receives transcripts, sends to Gemini, synthesizes reply (gTTS -> mp3), and plays into call

Environment variables required:
- PUBLIC_HOST (e.g. https://voice.yourdomain.com)  (no trailing slash)
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- DEEPGRAM_API_KEY
- GEMINI_API_KEY
- CAFE_NAME (optional)
- STAFF_NUMBER (optional)
- CAFE_HOURS (optional)
- MENU_LINK (optional)
"""

import os
import asyncio
import json
import time
from pathlib import Path
from threading import Thread

from flask import Flask, request, Response, send_from_directory
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioRestClient
from gtts import gTTS
import websockets
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ---------- Config ----------
PUBLIC_HOST = os.getenv("PUBLIC_HOST")  # e.g. https://voice.yourcafe.com (required)
if not PUBLIC_HOST:
    raise SystemExit("Set PUBLIC_HOST env var (e.g. https://voice.yourcafe.com)")

# Remove trailing slash if present
PUBLIC_HOST = PUBLIC_HOST.rstrip('/')

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN):
    raise SystemExit("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise SystemExit("Set DEEPGRAM_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set — replies will be basic.")

CAFE_NAME = os.getenv("CAFE_NAME", "Your Café")
STAFF_NUMBER = os.getenv("STAFF_NUMBER", None)
CAFE_HOURS = os.getenv("CAFE_HOURS", "Monday to Sunday, 8 AM to 9 PM")
MENU_LINK = os.getenv("MENU_LINK", "https://example.com/menu")

# Deepgram realtime websocket url (Twilio streams μ-law 8k by default)
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&punctuate=true"

# Ensure static dir exists
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Twilio client
twilio_client = TwilioRestClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Configure Gemini (if provided)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Track greeting state per call
call_states = {}

# ---------- Flask app ----------
app = Flask(__name__, static_folder="static")


@app.route("/voice", methods=["POST", "GET"])
def voice():
    """
    Twilio will POST here when a call arrives.
    Return TwiML that starts a Media Stream to wss://PUBLIC_HOST/media
    """
    resp = VoiceResponse()
    
    # Extract hostname from PUBLIC_HOST for WebSocket URL
    hostname = PUBLIC_HOST.replace('https://', '').replace('http://', '')
    stream_url = f"wss://{hostname}/media"
    
    # Use TwiML objects - Start stream immediately without initial Say
    connect = Connect()
    connect.stream(url=stream_url)
    resp.append(connect)
    
    return Response(str(resp), mimetype="text/xml")


@app.route("/static/<path:filename>")
def static_file(filename):
    """Serve generated audio files"""
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "cafe": CAFE_NAME}


# ---------- WebSocket media server ----------

async def handle_twilio_media(ws, path):
    """
    Handles Twilio Media Stream WebSocket connection.
    Processes audio, transcribes with Deepgram, generates responses with Gemini,
    and plays them back into the call.
    """
    dg_ws = None
    call_sid = None
    stream_sid = None
    read_task = None
    has_greeted = False

    try:
        # Open Deepgram WebSocket connection
        dg_ws = await websockets.connect(
            DEEPGRAM_WS_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        )
        print("[Deepgram] Connected")

        async def send_greeting(call_sid):
            """Send initial greeting when call starts"""
            greeting = f"Hello! Thank you for calling {CAFE_NAME}. I'm your virtual receptionist. How may I help you today?"
            print(f"[Bot] Greeting: {greeting}")
            await play_response(call_sid, greeting)

        async def read_deepgram():
            """Listen to Deepgram transcription results"""
            nonlocal call_sid, has_greeted
            
            async for dg_msg in dg_ws:
                try:
                    obj = json.loads(dg_msg)
                except json.JSONDecodeError:
                    continue

                # Check if this is a final transcript
                is_final = obj.get("is_final", False)
                if not is_final:
                    continue

                # Extract transcript
                transcript = None
                if "channel" in obj:
                    alternatives = obj["channel"].get("alternatives", [])
                    if alternatives and len(alternatives) > 0:
                        transcript = alternatives[0].get("transcript", "").strip()

                if not transcript:
                    continue

                print(f"[Deepgram] Final transcript: {transcript}")

                # Generate response
                reply_text = await generate_reply(transcript, call_sid)
                print(f"[Bot] Reply: {reply_text}")

                # Synthesize and play response
                await play_response(call_sid, reply_text)

        async def generate_reply(transcript, call_sid):
            """Generate reply using Gemini or fallback logic"""
            # Track conversation state
            if call_sid not in call_states:
                call_states[call_sid] = {"message_count": 0}
            
            call_states[call_sid]["message_count"] += 1
            
            if GEMINI_API_KEY:
                try:
                    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                    
                    # Enhanced prompt with context and conversation awareness
                    prompt = f"""You are a polite and helpful receptionist for {CAFE_NAME}.

Context:
- Operating hours: {CAFE_HOURS}
- Menu available at: {MENU_LINK}
{f"- Staff contact: {STAFF_NUMBER}" if STAFF_NUMBER else ""}

This is message #{call_states[call_sid]["message_count"]} in the conversation.

Customer said: "{transcript}"

Provide a brief, natural, and helpful response (1-2 sentences max). Be conversational and friendly.
If the customer is just greeting back or saying hello, acknowledge it briefly and ask how you can help them.

Response:"""
                    
                    response = model.generate_content(prompt)
                    reply_text = response.text.strip()
                    
                    # Ensure reply isn't too long
                    if len(reply_text) > 200:
                        reply_text = reply_text[:197] + "..."
                    
                    return reply_text
                    
                except Exception as e:
                    print(f"[Gemini] Error: {e}")
                    return "Sorry, I'm having trouble processing that. Could you please repeat?"
            else:
                # Fallback rule-based responses
                transcript_lower = transcript.lower()
                
                # Handle greetings
                if any(word in transcript_lower for word in ["hello", "hi", "hey"]):
                    return "Nice to hear from you! What can I help you with today?"
                elif "hour" in transcript_lower or "open" in transcript_lower:
                    return f"We are open {CAFE_HOURS}."
                elif "menu" in transcript_lower:
                    return f"You can view our menu at {MENU_LINK}."
                elif "reserve" in transcript_lower or "book" in transcript_lower:
                    return "Sure! How many people and what time would you like?"
                elif "location" in transcript_lower or "address" in transcript_lower:
                    return f"We are located at {CAFE_NAME}. Would you like directions?"
                else:
                    return "I didn't quite catch that. Could you please repeat?"

        async def play_response(call_sid, text):
            """Synthesize speech and play it into the call"""
            if not call_sid:
                print("[Error] No call_sid available")
                return

            # Generate unique filename
            timestamp = int(time.time() * 1000)
            filename = f"reply_{call_sid}_{timestamp}.mp3"
            out_path = STATIC_DIR / filename

            try:
                # Synthesize speech
                tts = gTTS(text, lang="en", slow=False)
                tts.save(str(out_path))
                print(f"[TTS] Saved audio to {filename}")

                # Play audio into call
                public_url = f"{PUBLIC_HOST}/static/{filename}"
                twiml_play = f'<?xml version="1.0" encoding="UTF-8"?><Response><Play>{public_url}</Play><Pause length="1"/></Response>'
                
                twilio_client.calls(call_sid).update(twiml=twiml_play)
                print(f"[Twilio] Playing audio: {public_url}")

            except Exception as e:
                print(f"[Error] Failed to play response: {e}")

        # Start Deepgram listener task
        read_task = asyncio.create_task(read_deepgram())

        # Process Twilio media stream
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = data.get("event")

            if event == "start":
                start = data.get("start", {})
                stream_sid = start.get("streamSid")
                call_sid = start.get("callSid")
                print(f"[Twilio] Media stream started - Call: {call_sid}, Stream: {stream_sid}")
                
                # Send greeting as soon as stream starts
                if not has_greeted:
                    # Small delay to ensure connection is stable
                    await asyncio.sleep(0.5)
                    await send_greeting(call_sid)
                    has_greeted = True

            elif event == "media":
                # Forward audio to Deepgram only after greeting
                if has_greeted:
                    media = data.get("media", {})
                    payload_b64 = media.get("payload")
                    
                    if payload_b64 and dg_ws:
                        await dg_ws.send(json.dumps({
                            "type": "Binary",
                            "audio": payload_b64
                        }))

            elif event == "stop":
                print("[Twilio] Media stream stopped")
                # Clean up call state
                if call_sid and call_sid in call_states:
                    del call_states[call_sid]
                break

    except websockets.exceptions.ConnectionClosed:
        print("[WebSocket] Connection closed")
    except Exception as e:
        print(f"[Error] WebSocket handler: {e}")
    finally:
        # Cleanup
        if read_task and not read_task.done():
            read_task.cancel()
            try:
                await read_task
            except asyncio.CancelledError:
                pass
        
        if dg_ws:
            try:
                await dg_ws.close()
            except:
                pass


# ---------- Run servers ----------

def start_ws_server():
    """Start WebSocket server for Twilio media streams"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    ws_server = websockets.serve(
        handle_twilio_media, 
        "0.0.0.0", 
        8765,
        ping_interval=20,
        ping_timeout=10
    )
    
    print("[WebSocket] Starting server on port 8765 (path /media)")
    loop.run_until_complete(ws_server)
    loop.run_forever()


if __name__ == "__main__":
    # Start WebSocket server in background thread
    ws_thread = Thread(target=start_ws_server, daemon=True)
    ws_thread.start()
    
    # Start Flask HTTP server
    port = int(os.getenv("PORT", 8080)) #port changed
    print(f"[Flask] Starting HTTP server on port {port}")
    print(f"[Config] PUBLIC_HOST: {PUBLIC_HOST}")
    print(f"[Config] Cafe: {CAFE_NAME}")
    
    # Use waitress or gunicorn in production instead of Flask debug
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)