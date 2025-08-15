import os, json, time, re
import redis
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "")
r = None
if REDIS_URL and (REDIS_URL.startswith("redis://") or REDIS_URL.startswith("rediss://")):
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True, ssl=REDIS_URL.startswith("rediss://"))
    except Exception:
        r = None

def _default_session():
    return {"state": "NEW", "bills": [], "updated_at": time.time()}

def load_session(phone_number: str):
    if not r:
        return _default_session()
    raw = r.get(f"sess:{phone_number}")
    try:
        return json.loads(raw) if raw else _default_session()
    except Exception:
        return _default_session()

def save_session(phone_number: str, data: dict, ttl_seconds: int = 60*60*24*30):
    data["updated_at"] = time.time()
    if r:
        r.set(f"sess:{phone_number}", json.dumps(data, separators=(",", ":")), ex=ttl_seconds)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/set_session")
def set_session():
    if not r:
        return "Redis not configured (check REDIS_URL)", 500
    r.set("test_key", "Hello from Redis!")
    return "Session set!", 200

@app.route("/get_session")
def get_session():
    if not r:
        return "Redis not configured (check REDIS_URL)", 500
    return f"Value from Redis: {r.get('test_key')}", 200

SAMPLE_BILL = {
    "retailer": "Alinta Energy",
    "period": "12 Jun → 13 Jul (32 days)",
    "kwh": 358.71,
    "supply_per_day": 0.95403,
    "blocks": [
        {"label": "Block 1", "kwh": 215.119, "rate": 0.27731},
        {"label": "Block 2", "kwh": 143.594, "rate": 0.28479},
    ],
    "notes": "No lock-in / No exit fee"
}

OFFERS = [
    {
        "id": 1, "retailer": "Lumo Energy", "plan": "Lumo Plus",
        "annual_cost": 1203, "annual_saving": 291, "supply_per_day": 0.84,
        "usage_desc": "Single rate ~25.5c/kWh", "exit_fee": "None", "cooling_off": 10,
        "handoff": "https://example.com/handoff?t=demo-lumo"
    },
    {
        "id": 2, "retailer": "Tango Energy", "plan": "eSelect",
        "annual_cost": 1232, "annual_saving": 262, "supply_per_day": 0.85,
        "usage_desc": "Single rate ~25.7c/kWh", "exit_fee": "None", "cooling_off": 10,
        "handoff": "https://example.com/handoff?t=demo-tango"
    }
]

def bill_summary_text(b):
    lines = [
        f"Retailer: {b['retailer']}",
        f"Period: {b['period']}",
        f"Usage: {b['kwh']} kWh",
        f"Daily supply: ${b['supply_per_day']:.5f}/day",
        "Usage blocks:",
        *(f"  • {blk['label']}: {blk['kwh']} kWh @ {blk['rate']*100:.3f} c/kWh" for blk in b["blocks"]),
        f"Notes: {b['notes']}"
    ]
    return "\n".join(lines)

@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    from_number = request.form.get("From", "unknown")
    body = (request.form.get("Body") or "").strip()
    body_lower = body.lower()
    num_media = int(request.form.get("NumMedia", 0))

    s = load_session(from_number)

    resp = MessagingResponse()
    msg = resp.message()
    def reply(text):
        msg.body(text)
        return str(resp)

    if s["state"] == "NEW":
        s["state"] = "WAITING_BILL"
        save_session(from_number, s)
        return reply(
            "Hi! I’m SwitchBuddy ⚡\n"
            "Send a photo or PDF of your latest *electricity* bill to begin.\n"
            "Tip: add multiple bills — after each, reply *another* or *done*."
        )

    if num_media > 0:
        s["bills"].append(SAMPLE_BILL)
        s["state"] = "CONFIRM"
        save_session(from_number, s)
        return reply(
            "Got your bill 📄 (sandbox parse)\n\n"
            + bill_summary_text(SAMPLE_BILL) + "\n\n"
            "If that looks right:\n"
            "• reply *another* to add another bill\n"
            "• reply *done* to compare offers\n"
            "Or reply *fix* to correct details (sandbox skips fix)."
        )

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
        if re.match(r"snooze\s+30", body_lower):
            return reply("Okay — I’ll snooze alerts for 30 days. (Demo)")
        if re.match(r"threshold\s+\d+", body_lower):
            return reply("Threshold updated. (Demo)")

    if body_lower in ["hi", "hello", "start"]:
        return reply("Hi! Send a photo/PDF of your latest bill. After I parse it, reply *another* or *done*.")

    return reply("I didn’t catch that. Send a bill (photo/PDF), or type *done* once you’re ready to compare.")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
