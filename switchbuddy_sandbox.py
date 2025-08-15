# switchbuddy_sandbox.py
# Full, paste-ready app file for Render + Twilio WhatsApp sandbox
# - No flask_session dependency
# - Redis is optional and safely handled
# - Includes /health, /redis_diag, /set_session, /get_session
# - WhatsApp webhook with simple state machine (bills -> offers)

import os
import re
import json
import time
import logging

import redis
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# ---------- Logging ----------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("switchbuddy")

# ---------- Flask App ----------
app = Flask(__name__)

# ---------- Redis (optional but preferred) ----------
REDIS_URL = (os.environ.get("REDIS_URL") or "").strip()
r = None
redis_error = None

if REDIS_URL and (REDIS_URL.startswith("redis://") or REDIS_URL.startswith("rediss://")):
    try:
        # Do not pass ssl=; rediss:// implies TLS automatically
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()  # verify connectivity so failures show up early
        log.info("Redis connected OK")
    except Exception as e:
        redis_error = f"{type(e).__name__}: {e}"
        log.exception("Redis init failed")
else:
    if REDIS_URL:
        redis_error = "Invalid REDIS_URL scheme (must start with redis:// or rediss://)"
    else:
        redis_error = "REDIS_URL is not set (running without Redis)"

def _default_session():
    return {"state": "NEW", "bills": [], "updated_at": time.time()}

def _sess_key(phone_number: str) -> str:
    return f"sess:{phone_number}"

def load_session(phone_number: str):
    """Load session from Redis if available; otherwise use in-memory default."""
    if not r:
        return _default_session()
    try:
        raw = r.get(_sess_key(phone_number))
        return json.loads(raw) if raw else _default_session()
    except Exception:
        log.exception("load_session failed; using default")
        return _default_session()

def save_session(phone_number: str, data: dict, ttl_seconds: int = 60 * 60 * 24 * 30):
    """Save session to Redis if available; no-op if Redis missing."""
    data["updated_at"] = time.time()
    if r:
        try:
            r.set(_sess_key(phone_number), json.dumps(data, separators=(",", ":")), ex=ttl_seconds)
        except Exception:
            log.exception("save_session failed (continuing)")

# ---------- Health & Diagnostics ----------
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/redis_diag", methods=["GET"])
def redis_diag():
    """
    Quick diagnostics for Redis connectivity.
    Returns JSON so we can see exactly what's going on in Render.
    """
    info = {
        "has_env": bool(REDIS_URL),
        "env_prefix": REDIS_URL.split("://", 1)[0] + "://" if REDIS_URL else None,
        "connected": False,
        "error": redis_error,
    }
    try:
        if r:
            pong = r.ping()
            info["connected"] = bool(pong)
            info["ping"] = pong
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return (info, 200)

@app.route("/set_session", methods=["GET"])
def set_session():
    try:
        if not r:
            return ("Redis not configured/connected. See /redis_diag", 503)
        r.set("test_key", "Hello from Redis!")
        return ("Session set!", 200)
    except Exception as e:
        log.exception("set_session failed")
        return (f"set_session error: {type(e).__name__}: {e}", 500)

@app.route("/get_session", methods=["GET"])
def get_session():
    try:
        if not r:
            return ("Redis not configured/connected. See /redis_diag", 503)
        val = r.get("test_key")
        return (f"Value from Redis: {val}", 200)
    except Exception as e:
        log.exception("get_session failed")
        return (f"get_session error: {type(e).__name__}: {e}", 500)

# ---------- Sandbox Data ----------
SAMPLE_BILL = {
    "retailer": "Alinta Energy",
    "period": "12 Jun → 13 Jul (32 days)",
    "kwh": 358.71,
    "supply_per_day": 0.95403,
    "blocks": [
        {"label": "Block 1", "kwh": 215.119, "rate": 0.27731},
        {"label": "Block 2", "kwh": 143.594, "rate": 0.28479},
    ],
    "notes": "No lock-in / No exit fee",
}

OFFERS = [
    {
        "id": 1,
        "retailer": "Lumo Energy",
        "plan": "Lumo Plus",
        "annual_cost": 1203,
        "annual_saving": 291,
        "supply_per_day": 0.84,
        "usage_desc": "Single rate ~25.5c/kWh",
        "exit_fee": "None",
        "cooling_off": 10,
        "handoff": "https://example.com/handoff?t=demo-lumo",
    },
    {
        "id": 2,
        "retailer": "Tango Energy",
        "plan": "eSelect",
        "annual_cost": 1232,
        "annual_saving": 262,
        "supply_per_day": 0.85,
        "usage_desc": "Single rate ~25.7c/kWh",
        "exit_fee": "None",
        "cooling_off": 10,
        "handoff": "https://example.com/handoff?t=demo-tango",
    },
]

def bill_summary_text(b):
    lines = [
        f"Retailer: {b['retailer']}",
        f"Period: {b['period']}",
        f"Usage: {b['kwh']} kWh",
        f"Daily supply: ${b['supply_per_day']:.5f}/day",
        "Usage blocks:",
        *(f"  • {blk['label']}: {blk['kwh']} kWh @ {blk['rate']*100:.3f} c/kWh" for blk in b["blocks"]),
        f"Notes: {b['notes']}",
    ]
    return "\n".join(lines)

# ---------- WhatsApp Webhook ----------
@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    try:
        from_number = request.form.get("From", "unknown")
        body = (request.form.get("Body") or "").strip()
        body_lower = body.lower()
        num_media = int(request.form.get("NumMedia", 0) or 0)

        s = load_session(from_number)

        resp = MessagingResponse()
        msg = resp.message()

        def reply(text):
            msg.body(text)
            return str(resp)

        # First-time / reset
        if s["state"] == "NEW":
            s["state"] = "WAITING_BILL"
            save_session(from_number, s)
            return reply(
                "Hi! I’m SwitchBuddy ⚡\n"
                "Send a photo or PDF of your latest *electricity* bill to begin.\n"
                "Tip: add multiple bills — after each, reply *another* or *done*."
            )

        # If a bill image/PDF arrives
        if num_media > 0:
            # In sandbox, we skip OCR and attach SAMPLE_BILL
            s["bills"].append(SAMPLE_BILL)
            s["state"] = "CONFIRM"
            save_session(from_number, s)
            return reply(
                "Got your bill 📄 (sandbox parse)\n\n"
                + bill_summary_text(SAMPLE_BILL)
                + "\n\nIf that looks right:\n"
                  "• reply *another* to add another bill\n"
                  "• reply *done* to compare offers\n"
                  "Or reply *fix* to correct details (sandbox skips fix)."
            )

        # Waiting for either "another" or "done"
        if s["state"] in ["WAITING_BILL", "CONFIRM"]:
            if body_lower in ["another", "+ another", "+another"]:
                s["state"] = "WAITING_BILL"
                save_session(from_number, s)
                return reply("Okay — send the next bill (photo or PDF).")
            if body_lower in ["done", "that's all", "thats all", "that’s all"]:
                s["state"] = "PRESENTING_OFFERS"
                save_session(from_number, s)
                o1, o2 = OFFERS
                return reply(
                    "Here are your top 2 plans (sandbox):\n\n"
                    f"1) {o1['retailer']} — {o1['plan']}\n"
                    f"   Est. annual cost: ${o1['annual_cost']} | Save: ${o1['annual_saving']}\n"
                    f"   Supply: ${o1['supply_per_day']}/day | {o1['usage_desc']}\n"
                    f"   Exit fee: {o1['exit_fee']} | Cooling-off: {o1['cooling_off']} business days\n"
                    "   Reply: *details 1* or *switch 1*\n\n"
                    f"2) {o2['retailer']} — {o2['plan']}\n"
                    f"   Est. annual cost: ${o2['annual_cost']} | Save: ${o2['annual_saving']}\n"
                    f"   Supply: ${o2['supply_per_day']}/day | {o2['usage_desc']}\n"
                    f"   Exit fee: {o2['exit_fee']} | Cooling-off: {o2['cooling_off']} business days\n"
                    "   Reply: *details 2* or *switch 2*\n\n"
                    "You can also reply *snooze 30* or *threshold 150* (demo)."
                )
            if body_lower.startswith("fix"):
                return reply("Sandbox note: *fix* isn’t wired up yet. Reply *another* to resend, or *done* to continue.")

        # Offers state
        if s["state"] == "PRESENTING_OFFERS":
            if re.match(r"details\s+1", body_lower):
                o = OFFERS[0]
                return reply(
                    f"**Details — {o['retailer']} {o['plan']}**\n"
                    f"Est. annual cost: ${o['annual_cost']} (save ${o['annual_saving']})\n"
                    f"Supply: ${o['supply_per_day']}/day\n"
                    f"Usage: {o['usage_desc']}\n"
                    f"Exit fee: {o['exit_fee']} | Cooling-off: {o['cooling_off']} business days\n"
                    "Reply *switch 1* to proceed or *back*."
                )
            if re.match(r"details\s+2", body_lower):
                o = OFFERS[1]
                return reply(
                    f"**Details — {o['retailer']} {o['plan']}**\n"
                    f"Est. annual cost: ${o['annual_cost']} (save ${o['annual_saving']})\n"
                    f"Supply: ${o['supply_per_day']}/day\n"
                    f"Usage: {o['usage_desc']}\n"
                    f"Exit fee: {o['exit_fee']} | Cooling-off: {o['cooling_off']} business days\n"
                    "Reply *switch 2* to proceed or *back*."
                )
            if re.match(r"switch\s+1", body_lower):
                return reply("Opening signup…\n"
                             f"Secure link: {OFFERS[0]['handoff']}\n"
                             "This is a demo link in sandbox.")
            if re.match(r"switch\s+2", body_lower):
                return reply("Opening signup…\n"
                             f"Secure link: {OFFERS[1]['handoff']}\n"
                             "This is a demo link in sandbox.")
            if body_lower == "back":
                s["state"] = "CONFIRM"
                save_session(from_number, s)
                return reply("Type *done* to see the two offers again, or *another* to add more bills.")
            if re.match(r"snooze\s+\d+", body_lower):
                return reply("Okay — I’ll snooze alerts. (Demo)")
            if re.match(r"threshold\s+\d+", body_lower):
                return reply("Threshold updated. (Demo)")

        # Basic greetings (helps if someone types 'hi' again)
        if body_lower in ["hi", "hello", "start"]:
            return reply("Hi! Send a photo/PDF of your latest bill. After I parse it, reply *another* or *done*.")

        # Fallback
        return reply("I didn’t catch that. Send a bill (photo/PDF), or type *done* once you’re ready to compare.")
    except Exception as e:
        log.exception("whatsapp_webhook error")
        resp = MessagingResponse()
        resp.message("Oops—something went wrong, but I’m still here. Please try again.")
        return str(resp)

# ---------- Local run ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
