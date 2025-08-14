from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = (request.values.get("Body") or "").lower()
    resp = MessagingResponse()
    msg = resp.message()

    if "hi" in incoming_msg:
        msg.body("Hi! Please send me a photo of your electricity bill to get started.")
    elif "another" in incoming_msg:
        msg.body("Okay, send me the next bill photo.")
    elif "done" in incoming_msg:
        msg.body("Thanks! I’ll start comparing your plans now and send your weekly digest.")
    else:
        msg.body("Got it! (Sandbox: I’d process this bill now.)")

    return str(resp)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
