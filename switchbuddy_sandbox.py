from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
from upstash_redis import Redis

# ---- Upstash via env (no secrets in code) ----
UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")

r = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/redis_diag", methods=["GET"])
def redis_diag():
    # Show helpful diagnostics without leaking secrets
    has_url = bool(UPSTASH_URL)
    has_token = bool(UPSTASH_TOKEN)
    url_prefix = None
    if has_url:
        # show just the scheme/host prefix for sanity (no creds)
        # Upstash REST URLs look like "https://<host>.upstash.io"
        try:
            url_prefix = UPSTASH_URL.split("://", 1)[0] + "://" + UPSTASH_URL.split("://", 1)[1].split("/", 1)[0]
        except Exception:
            url_prefix = "unparsed"

    try:
        ping_ok = r.ping()
        return {
            "has_url": has_url,
            "has_token": has_token,
            "url_prefix": url_prefix,
            "connected": True,
            "ping": ping_ok,
            "error": None,
        }, 200
    except Exception as e:
        return {
            "has_url": has_url,
            "has_token": has_token,
            "url_prefix": url_prefix,
            "connected": False,
            "ping": False,
            "error": str(e),
        }, 500

@app.route("/set_session", methods=["GET"])
def set_session():
    r.set("test_key", "Hello from Redis!")
    return "Session set!", 200

@app.route("/get_session", methods=["GET"])
def get_session():
    value = r.get("test_key")
    return f"Value from Redis: {value}", 200

@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = (request.values.get("Body") or "").lower()
    resp = MessagingResponse()
    msg = resp.message()

    if "hi" in incoming_msg:
        msg.body("Hi! I'm SwitchBuddy, please send me a photo of your electricity bill to get started on finding you a better deal on your utilities.")
    elif "another" in incoming_msg:
        msg.body("Okay, if you have another bill, please send me the next photo.")
    elif "done" in incoming_msg or "that's all" in incoming_msg:
        msg.body("Thanks! I’ll start comparing your plans now and send your weekly digest.")
    else:
        msg.body("Got it! (Sandbox: I’d process this bill now.)")

    return str(resp)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
