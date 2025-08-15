from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
from upstash_redis import Redis

# Load env vars (recommended so we don't hardcode secrets)
REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "https://whole-trout-5713.upstash.io")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "ARZRAAImcDE4ODNlOTI3ZTgwNTg0MWVkODhmMmI0Njg3NDZhYzMwNnAxNTcxMw")

# Connect to Upstash Redis
r = Redis(url=REDIS_URL, token=REDIS_TOKEN)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/redis_diag", methods=["GET"])
def redis_diag():
    try:
        test_ping = r.ping()
        return {
            "connected": True,
            "ping": test_ping
        }, 200
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
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
    elif "done" in incoming_msg:
        msg.body("Thanks! I’ll start comparing your plans now and send your weekly digest.")
    else:
        msg.body("Got it! (Sandbox: I’d process this bill now.)")

    return str(resp)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
