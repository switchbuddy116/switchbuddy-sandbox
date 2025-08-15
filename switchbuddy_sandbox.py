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
    from_number = (request.values.get("From") or "").replace("whatsapp:", "")

    # --- Normalize incoming text ---
    raw = (request.values.get("Body") or "").strip()
    # unify common unicode punctuation/whitespace to ASCII
    normalized = (
        raw.replace("\u2019", "'")   # curly apostrophe -> '
           .replace("\u2018", "'")   # left single quote -> '
           .replace("\u201C", '"')   # left double quote -> "
           .replace("\u201D", '"')   # right double quote -> "
           .replace("\u2013", "-")   # en dash -> -
           .replace("\u2014", "-")   # em dash -> -
    )
    # collapse internal whitespace and lowercase
    incoming_msg = " ".join(normalized.split()).lower().strip(" .!?,;:'\"-")

    # Keys per user
    k_state = f"user:{from_number}:state"     # idle|collecting|done
    k_count = f"user:{from_number}:bill_count"

    # Init defaults if missing
    if r.get(k_state) is None:
        r.set(k_state, "idle")
    if r.get(k_count) is None:
        r.set(k_count, 0)

    state = r.get(k_state)
    bill_count = int(r.get(k_count) or 0)

    resp = MessagingResponse()
    msg = resp.message()

    # Helpers for intent checks
    def said_any(*phrases):
        return any(p in incoming_msg for p in phrases)

    # --- Intents ---
    if said_any("hi", "hello", "start"):
        r.set(k_state, "collecting")
        msg.body(
            "Hi! I’m SwitchBuddy ⚡️\n\n"
            "Please send a photo or PDF of your electricity bill.\n\n"
            "When you’re finished:\n"
            "• Reply: 'add another bill' to upload more\n"
            "• Reply: 'that's all' to finish"
        )
        return str(resp)

    if said_any("that's all", "thats all", "done", "finish", "finished", "all done"):
        r.set(k_state, "done")
        count = int(r.get(k_count) or 0)
        msg.body(
            f"All set! I’ve saved {count} bill(s).\n"
            "I’ll crunch the numbers and include them in your weekly digest. "
            "If you want to add more later, just say 'hi'."
        )
        return str(resp)

    if said_any("add another bill", "add another", "another bill", "add more", "upload another"):
        r.set(k_state, "collecting")
        msg.body("Cool — send the next bill photo/PDF. When finished, say “that's all”.")
        return str(resp)

    # Media?
    media_url = request.values.get("MediaUrl0")
    media_type = request.values.get("MediaContentType0")

    if state == "collecting" and media_url:
        # (Sandbox) just count it; prod would download+parse
        new_count = bill_count + 1
        r.set(k_count, new_count)
        msg.body(
            f"Bill received ✅ (#{new_count}).\n\n"
            "Reply 'add another bill' to add more, or 'that's all' to finish."
        )
        return str(resp)

    # Fallbacks
    if state == "collecting":
        msg.body(
            "I’m ready — please send a photo/PDF of your bill.\n"
            "Or say 'that's all' when you’re finished."
        )
    elif state == "done":
        msg.body("You’re done for now 🎉 Say 'hi' if you want to add more bills.")
    else:
        msg.body("Say 'hi' to get started with your bill upload.")

    return str(resp)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
