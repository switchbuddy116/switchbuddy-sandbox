# switchbuddy_sandbox.py

import os
import time
import json
import requests
from urllib.parse import quote

from flask import Flask, request, Response, redirect
from twilio.twiml.messaging_response import MessagingResponse
from upstash_redis import Redis

# -----------------------------
# Redis (Upstash) client
# -----------------------------
UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise RuntimeError("Missing UPSTASH_URL or UPSTASH_TOKEN environment variables.")

# Use `r` consistently
r = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)

# Your public base URL (e.g., https://switchbuddy-sandbox.onrender.com)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Helpers
# -----------------------------

# ---- Bill parsing (stub using OCR/API later) ----
def parse_bill_bytes(content: bytes, content_type: str) -> dict:
    """
    TODO: Replace with real OCR/API.
    For now, return a plausible structure so compare works end-to-end.
    """
    # Demo defaults
    return {
        "period_start": "2025-06-01",
        "period_end": "2025-08-01",
        "total_kwh": 1200,             # 2 months worth (example)
        "daily_kwh": round(1200 / 62, 2),
        "supply_cents_per_day": 110,   # $1.10/day
        "usage_cents_per_kwh_peak": 40,
        "usage_cents_per_kwh_shoulder": 28,
        "usage_cents_per_kwh_offpeak": 22,
        "tariff_type": "TOU",
        "distributor": "CitiPower",    # example
        "total_cost_inc_gst": 540.00
    }


def e164(raw: str) -> str:
    """
    Normalize a phone for Redis keys:
    - strips 'whatsapp:' prefix
    - keeps digits and leading '+'
    - adds '+' if missing
    """
    if not raw:
        return ""
    s = raw.strip().replace("whatsapp:", "")
    s = "".join(ch for ch in s if ch.isdigit() or ch == "+")
    if not s:
        return ""
    if s[0] != "+":
        s = "+" + s
    return s

def _normalize_text(s: str) -> str:
    """Normalize curly quotes, dashes, collapse whitespace, lowercase, strip punct."""
    if not s:
        return ""
    normalized = (
        s.strip()
         .replace("\u2019", "'")
         .replace("\u2018", "'")
         .replace("\u201C", '"')
         .replace("\u201D", '"')
         .replace("\u2013", "-")
         .replace("\u2014", "-")
    )
    return " ".join(normalized.split()).lower().strip(" .!?,;:'\"-")

def download_twilio_media(media_url: str, timeout=15):
    """
    Downloads a Twilio-hosted media URL using HTTP Basic Auth.
    Returns (content_bytes, content_type, content_length_int).
    Raises Exception on failure.
    """
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise RuntimeError("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN.")

    max_bytes = 10 * 1024 * 1024  # 10 MB cap for sandbox
    with requests.get(media_url, auth=(sid, token), stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        content_len_header = resp.headers.get("Content-Length")
        content_length = int(content_len_header) if content_len_header and content_len_header.isdigit() else None

        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Media exceeds {max_bytes} bytes limit for sandbox.")
        content = b"".join(chunks)

    return content, content_type, (content_length if content_length is not None else len(content))

def _build_digest_html(phone: str) -> str:
    """Builds a simple weekly digest HTML from what we’ve saved in Redis."""
    k_bills = f"user:{phone}:bills"
    entries = r.lrange(k_bills, 0, -1) or []
    total_bills = len(entries)

    # Pull last 5 as a preview list
    preview = []
    for raw in entries[-5:]:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            j = json.loads(raw)
        except Exception:
            j = {"media_type": "unknown", "ts": 0}
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(j.get("ts", 0)))
        mtype = (j.get("media_type") or "").split("/")[-1].upper()
        preview.append(f"<li>{when} — {mtype or 'UNKNOWN'}</li>")

    # Optional savings param for demo
    savings = request.args.get("savings", "100")
    try:
        savings_val = float(savings)
    except:
        savings_val = 100.0

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SwitchBuddy Weekly Digest</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color:#0f172a; margin:0; }}
    .wrap {{ max-width: 720px; margin: 0 auto; padding: 24px; }}
    .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; box-shadow:0 1px 2px rgba(0,0,0,0.03); }}
    h1 {{ font-size: 22px; margin:0 0 8px }}
    h2 {{ font-size: 18px; margin:16px 0 8px }}
    p  {{ margin: 8px 0 }}
    .cta {{ display:inline-block; padding:10px 14px; border-radius:10px; text-decoration:none; border:1px solid #0ea5e9; }}
    .cta-primary {{ background:#0ea5e9; color:white; border-color:#0ea5e9; }}
    ul {{ margin:8px 0 0 18px }}
    .muted {{ color:#64748b }}
    .grid {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .panel {{ border:1px solid #e2e8f0; border-radius:10px; padding:12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>⚡ SwitchBuddy — Weekly Digest</h1>
      <p class="muted">Phone: {phone}</p>

      <div class="grid">
        <div class="panel">
          <h2>Summary</h2>
          <p>Saved bills on file: <strong>{total_bills}</strong></p>
          <p>Projected annual savings: <strong>${savings_val:,.0f}</strong></p>
        </div>

        <div class="panel">
          <h2>Latest uploads</h2>
          {"<ul>" + "".join(preview) + "</ul>" if preview else "<p class='muted'>No recent bills.</p>"}
        </div>
      </div>

      <h2 style="margin-top:16px">Recommendation</h2>
      <p>Based on your recent usage and current market rates, you’re a good candidate to switch to a <strong>low daily charge</strong> plan.</p>
      <p class="muted">(*Demo text*) We’ll refine this once we parse TOU blocks and compare against live rates.</p>

      <p style="margin-top:16px">
        <a class="cta cta-primary" href="#">Switch plan</a>
        <a class="cta" href="#">See full comparison</a>
      </p>
    </div>
  </div>
</body>
</html>
"""
    return html

# ---- Tariff fetcher (stub) ----
def fetch_tariffs_vic():
    """
    TODO: Swap with live data later (VIC source).
    Return a few sample plans to compare against.
    All cents values include GST for simplicity.
    """
    return [
        {
            "name": "Flat Saver",
            "tariff_type": "FLAT",
            "supply_cents_per_day": 105,
            "usage_cents_per_kwh": 32
        },
        {
            "name": "TOU Value",
            "tariff_type": "TOU",
            "supply_cents_per_day": 95,
            "peak_cents": 42,
            "shoulder_cents": 28,
            "offpeak_cents": 20
        },
        {
            "name": "Daily Low",
            "tariff_type": "FLAT",
            "supply_cents_per_day": 80,
            "usage_cents_per_kwh": 36
        }
    ]


def estimate_annual_cost(profile: dict, plan: dict) -> float:
    """
    Very rough estimate. Improve once we parse TOU shares from bills.
    """
    # Infer annual kWh from bill period if total_kwh provided
    total_kwh = float(profile.get("total_kwh") or 0)
    start = profile.get("period_start")
    end = profile.get("period_end")

    # Default to 365 days and scale up
    days = 62  # if we can’t parse exact days, assume 2 months
    try:
        from datetime import datetime
        if start and end:
            ds = datetime.fromisoformat(start)
            de = datetime.fromisoformat(end)
            days = max((de - ds).days, 1)
    except Exception:
        pass

    daily_kwh = profile.get("daily_kwh")
    if daily_kwh is None and total_kwh:
        daily_kwh = total_kwh / days
    if daily_kwh is None:
        daily_kwh = 10  # fallback guess

    annual_kwh = daily_kwh * 365

    # Supply charge
    supply_cents = float(plan.get("supply_cents_per_day") or 0) * 365

    if plan.get("tariff_type") == "FLAT":
        usage_cents = annual_kwh * float(plan.get("usage_cents_per_kwh") or 0)
    else:
        # crude split until we have real TOU breakdowns
        peak = 0.5 * annual_kwh
        shoulder = 0.3 * annual_kwh
        offpeak = 0.2 * annual_kwh
        usage_cents = (
            peak * float(plan.get("peak_cents") or 0) +
            shoulder * float(plan.get("shoulder_cents") or 0) +
            offpeak * float(plan.get("offpeak_cents") or 0)
        )

    total_cents = supply_cents + usage_cents
    return round(total_cents / 100.0, 2)  # dollars


def compare_and_store(phone: str) -> dict:
    """
    Load the user’s latest parsed bill (or aggregate of bills),
    compute best plan, store a snapshot, and return it.
    """
    k_bills = f"user:{phone}:bills"
    raw_entries = r.lrange(k_bills, 0, -1) or []

    # Use the last bill that has a parsed profile if present; otherwise synthesize a basic profile
    latest_profile = None
    for raw in reversed(raw_entries):
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            j = json.loads(raw)
            if j.get("parsed"):
                latest_profile = j["parsed"]
                break
        except Exception:
            pass

    if not latest_profile:
        # Fallback profile so the pipeline runs (replace once parsing is in)
        latest_profile = {
            "total_kwh": 1200,
            "period_start": "2025-06-01",
            "period_end": "2025-08-01",
            "daily_kwh": round(1200/62, 2),
        }

    plans = fetch_tariffs_vic()
    scored = []
    for p in plans:
        cost = estimate_annual_cost(latest_profile, p)
        scored.append({"plan": p, "annual_cost": cost})
    scored.sort(key=lambda x: x["annual_cost"])

    best = scored[0]
    worst = scored[-1]
    # “Current” cost guess: use worst or flat mid—here we’ll use the median as a softer baseline
    median_cost = scored[len(scored)//2]["annual_cost"]
    savings = round(median_cost - best["annual_cost"], 2)

    snapshot = {
        "ts": int(time.time()),
        "profile": latest_profile,
        "ranked": scored,
        "best": best,
        "baseline_cost": median_cost,
        "projected_savings": savings
    }

    r.set(f"user:{phone}:last_comparison", json.dumps(snapshot))
    return snapshot


# -----------------------------
# Health & diagnostics
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/redis_diag", methods=["GET"])
def redis_diag():
    try:
        pong = r.ping()
        return {
            "connected": True,
            "error": None,
            "has_token": bool(UPSTASH_TOKEN),
            "has_url": bool(UPSTASH_URL),
            "ping": pong,
            "url_prefix": (UPSTASH_URL or "")[: (UPSTASH_URL.find(".upstash.io") + 12)] if UPSTASH_URL else ""
        }, 200
    except Exception as e:
        return {
            "connected": False,
            "error": f"{type(e).__name__}: {e}",
            "has_token": bool(UPSTASH_TOKEN),
            "has_url": bool(UPSTASH_URL),
        }, 500

# -----------------------------
# Simple session smoke tests
# -----------------------------
@app.route("/set_session", methods=["GET"])
def set_session():
    r.set("test_key", "Hello from Redis!")
    return "Session set!", 200

@app.route("/get_session", methods=["GET"])
def get_session():
    value = r.get("test_key")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return f"Value from Redis: {value}", 200

# -----------------------------
# Weekly digest (HTML)
# -----------------------------
@app.route("/digest_preview", methods=["GET"])
def digest_preview():
    phone = request.args.get("phone", "").strip()
    if not phone:
        return "Missing ?phone=E164 (e.g., ?phone=%2B6145...)", 400
    # Ensure it’s normalized like our keys
    phone = e164(phone)
    html = _build_digest_html(phone)
    return Response(html, mimetype="text/html")

# Backward-compat: /digest -> /digest_preview
@app.route("/digest", methods=["GET"])
def digest_redirect():
    phone = request.args.get("phone", "")
    if not phone:
        return redirect("/digest_preview")
    return redirect(f"/digest_preview?phone={quote(phone, safe='')}")

# -----------------------------
# WhatsApp webhook
# -----------------------------
@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    from_number_raw = request.values.get("From") or ""
    phone = e164(from_number_raw)

    raw = (request.values.get("Body") or "")
    incoming_msg = _normalize_text(raw)

    # Per-user keys
    k_state = f"user:{phone}:state"        # idle|collecting|done
    k_count = f"user:{phone}:bill_count"   # integer
    k_bills = f"user:{phone}:bills"        # LIST of JSON entries

    # Initialize defaults
    if r.get(k_state) is None:
        r.set(k_state, "idle")
    if r.get(k_count) is None:
        r.set(k_count, 0)

    state = r.get(k_state)
    if isinstance(state, bytes):
        state = state.decode("utf-8", errors="ignore")
    bill_count = int(r.get(k_count) or 0)

    resp = MessagingResponse()
    msg = resp.message()

    def said_any(*phrases) -> bool:
        return any(p in incoming_msg for p in phrases)

    # --- Intents ---

    # Start
    if said_any("hi", "hello", "start"):
        r.set(k_state, "collecting")
        msg.body(
            "Hi! I’m SwitchBuddy ⚡️\n\n"
            "Please send a photo or PDF of your electricity bill.\n\n"
            "When you’re finished:\n"
            "• Reply: 'add another bill' to upload more\n"
            "• Reply: 'that's all' to finish\n"
            "• Reply: 'list bills' to see what I’ve saved\n"
            "• Reply: 'digest' to get your weekly digest link"
        )
        return str(resp)

    # Share digest link on demand
    if said_any("digest", "show digest", "weekly digest"):
        if not PUBLIC_BASE_URL:
            msg.body("Digest preview is not available yet (missing PUBLIC_BASE_URL).")
            return str(resp)
        digest_url = f"{PUBLIC_BASE_URL}/digest_preview?phone={quote(phone, safe='')}"
        msg.body(f"Here’s your digest preview:\n{digest_url}")
        return str(resp)

    # Finish
    if said_any("that's all", "thats all", "done", "finish", "finished", "all done"):
        r.set(k_state, "done")
        count = int(r.get(k_count) or 0)

        if PUBLIC_BASE_URL:
            digest_url = f"{PUBLIC_BASE_URL}/digest_preview?phone={quote(phone, safe='')}"
            msg.body(
                f"All set! I’ve saved {count} bill(s).\n"
                "I’ll crunch the numbers and include them in your weekly digest.\n\n"
                f"Preview it here: {digest_url}\n\n"
                "If you want to add more later, just say 'hi'."
            )
        else:
            msg.body(
                f"All set! I’ve saved {count} bill(s).\n"
                "I’ll crunch the numbers and include them in your weekly digest. "
                "If you want to add more later, just say 'hi'."
            )
        return str(resp)

    # List bills
    if said_any("list bills", "show bills", "what have you saved"):
        entries = r.lrange(k_bills, -10, -1) or []
        if not entries:
            msg.body("I haven’t saved any bill files yet. Send a photo/PDF to add one.")
            return str(resp)

        lines = []
        start_index = max(1, bill_count - len(entries) + 1)
        for i, e in enumerate(entries, start=start_index):
            try:
                if isinstance(e, bytes):
                    e = e.decode("utf-8", errors="ignore")
                j = json.loads(e)
                media_type = (j.get("media_type") or "").lower()
                kind = "PDF" if media_type.endswith("pdf") or "pdf" in media_type else "Image"
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(j.get("ts", 0)))
                lines.append(f"#{i}: {kind} @ {ts}")
            except Exception:
                lines.append(f"#{i}: (unparsed)")
        msg.body("Recent saved bills:\n" + "\n".join(lines))
        return str(resp)

    # Add another
    if said_any("add another bill", "add another", "another bill", "add more", "upload another"):
        r.set(k_state, "collecting")
        msg.body("Cool — send the next bill photo/PDF. When finished, say “that's all”.")
        return str(resp)

    # Media handling (Twilio sends MediaUrl0/MediaContentType0 when files attached)
    media_url = request.values.get("MediaUrl0")
    media_type = request.values.get("MediaContentType0")

    if state == "collecting" and media_url:
        # Try to fetch the media from Twilio (auth required)
        downloaded_ok = False
        size_bytes = None
        fetched_type = None
        err = None
        try:
            content, fetched_type, size_bytes = download_twilio_media(media_url)
            downloaded_ok = True
            # (Sandbox) Not storing the actual file yet.
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        entry = {
            "media_url": media_url,                 # Twilio temp URL (expires)
            "media_type": media_type or fetched_type or "",
            "ts": int(time.time()),
            "downloaded_ok": downloaded_ok,
            "download_err": err,
            "download_size_bytes": size_bytes
        }
        r.rpush(k_bills, json.dumps(entry))
        new_count = bill_count + 1
        r.set(k_count, new_count)

        if downloaded_ok:
            if size_bytes is None:
                human_size = "unknown size"
            elif size_bytes < 1024:
                human_size = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                human_size = f"{size_bytes/1024:.1f} KB"
            else:
                human_size = f"{size_bytes/1024/1024:.2f} MB"

            msg.body(
                f"Bill received ✅ (#{new_count}).\n"
                f"(Fetched media: {human_size})\n\n"
                "Reply 'add another bill' to add more, 'list bills' to review, or 'that's all' to finish."
            )
        else:
            msg.body(
                f"Bill received ✅ (#{new_count}).\n"
                "Heads up: I couldn’t fetch the file from Twilio just now, but the link was saved. "
                "You can try sending it again, or proceed.\n\n"
                "Reply 'add another bill' to add more, 'list bills' to review, or 'that's all' to finish."
            )
        return str(resp)

    # Fallbacks based on state
    if state == "collecting":
        msg.body(
            "I’m ready — please send a photo/PDF of your bill.\n"
            "Or say 'that's all' when you’re finished."
        )
    elif state == "done":
        # Nudge digest link if we can
        if PUBLIC_BASE_URL:
            digest_url = f"{PUBLIC_BASE_URL}/digest_preview?phone={quote(phone, safe='')}"
            msg.body(f"You’re done for now 🎉\nPreview your digest any time: {digest_url}\nSay 'hi' to add more bills.")
        else:
            msg.body("You’re done for now 🎉 Say 'hi' if you want to add more bills.")
    else:
        msg.body("Say 'hi' to get started with your bill upload.")
    return str(resp)

# After content, fetched_type, size_bytes = download_twilio_media(...)
parsed = parse_bill_bytes(content, fetched_type or (media_type or ""))

entry = {
    "media_url": media_url,
    "media_type": media_type or fetched_type or "",
    "ts": int(time.time()),
    "downloaded_ok": True,
    "download_err": None,
    "download_size_bytes": size_bytes,
    "parsed": parsed,  # <--- store the structured result
}
r.rpush(k_bills, json.dumps(entry))

if said_any("that's all", "thats all", "done", "finish", "finished", "all done"):
    r.set(k_state, "done")
    count = int(r.get(k_count) or 0)
    cmp = compare_and_store(phone)
    savings = cmp.get("projected_savings", 0.0)
    msg.body(
        f"All set! I’ve saved {count} bill(s).\n"
        f"Projected annual savings right now: ${savings:,.0f}.\n"
        "I’ll include this in your weekly digest. Say 'digest' to preview."
    )
    return str(resp)

snap_raw = r.get(f"user:{phone}:last_comparison")
if snap_raw:
    try:
        if isinstance(snap_raw, bytes):
            snap_raw = snap_raw.decode("utf-8", errors="ignore")
        snap = json.loads(snap_raw)
        best_name = snap["best"]["plan"]["name"]
        best_cost = snap["best"]["annual_cost"]
        savings = snap["projected_savings"]
        # add a small “Best plan” panel into your existing HTML
        # (you can weave this into your template where you like)
    except Exception:
        pass


# -----------------------------
# Admin/testing: list & last bill
# -----------------------------
@app.route("/user_bills", methods=["GET"])
def user_bills():
    """
    Admin helper:
      GET /user_bills?phone=%2B61XXXXXXXXX
    Returns JSON of saved bill entries for that phone.
    """
    phone = e164(request.args.get("phone", ""))
    if not phone:
        return {"error": "Missing ?phone=E164 (encode + as %2B)"}, 400
    k_bills = f"user:{phone}:bills"
    entries = r.lrange(k_bills, 0, -1) or []
    parsed = []
    for e in entries:
        try:
            if isinstance(e, bytes):
                e = e.decode("utf-8", errors="ignore")
            parsed.append(json.loads(e))
        except Exception:
            parsed.append({"raw": str(e)})
    return {"phone": phone, "count": len(parsed), "bills": parsed}, 200

@app.route("/last_bill", methods=["GET"])
def last_bill():
    """Quick debug: /last_bill?phone=%2B61XXXXXXXXX -> shows last saved bill entry."""
    phone = e164(request.args.get("phone", ""))
    if not phone:
        return {"error": "Missing ?phone=E164 (encode + as %2B)"}, 400
    k_bills = f"user:{phone}:bills"
    last = r.lindex(k_bills, -1)
    if not last:
        return {"phone": phone, "last_bill": None}, 200
    try:
        if isinstance(last, bytes):
            last = last.decode("utf-8", errors="ignore")
        return {"phone": phone, "last_bill": json.loads(last)}, 200
    except Exception:
        return {"phone": phone, "last_bill": {"raw": str(last)}}, 200

# -----------------------------
# Entrypoint (local dev)
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
