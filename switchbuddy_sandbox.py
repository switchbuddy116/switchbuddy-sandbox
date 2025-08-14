from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import redis
import os

# Connect to Redis
redis_url = os.environ.get("REDIS_URL")
r = redis.from_url(redis_url)

@app.route("/set_session")
def set_session():
    r.set("test_key", "Hello from Redis!")
    return "Session set!", 200

@app.route("/get_session")
def get_session():
    value = r.get("test_key")
    return f"Value from Redis: {value}", 200

from flask import Flask, request, session
from flask_session import Session



app = Flask(__name__)
# Redis configuration
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = redis.from_url(os.environ.get("REDIS_URL"))
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev_secret")  # Change for production

# Initialize session
Session(app)


@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

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
