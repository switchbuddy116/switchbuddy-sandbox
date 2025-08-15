# switchbuddy_sandbox.py

import os
import time
import json
import requests
from flask import Flask, request, Response
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

# -----------------------------
# Flask app (define BEFORE any @app.route)
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Helpers
# -----------------------------
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
    """Normalize curly quotes, dashes, excess whitespace, and strip punctuation."""
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
# Digest (HTML) — helper + route
# -----------------------------
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

@app.route("/digest", methods=["GET"])
def digest():
    """
    Preview the weekly digest in the browser.
    Usage:
      /digest?phone=+61457344494
      /digest?phone=+61457344494&savings=150
    """
    phone = request.args.get("phone", "").strip()
    if not phone:
        return {"error": "Missing ?phone=+61XXXXXXXXX"}, 400
    html = _build_digest_html(phone)
    return Response(html, mimetype="text/html")

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

    # Start
    if said_any("hi", "hello", "start"):
        r.set(k_state, "collecting")
        msg.body(
            "Hi! I’m SwitchBuddy ⚡️\n\n"
            "Please send a photo or PDF of your electricity bill.\n\n"
            "When you’re finished:\n"
            "• Reply: 'add another bill' to upload more\n"
            "• Reply: 'that's all' to finish\n"
            "• Reply: 'list bills' to see what I’ve saved"
        )
        return str(resp)

    # Finish
    if said_any("that's all", "thats all", "done", "finish", "finished", "all done"):
        r.set(k_state, "done")
        count = int(r.get(k_count) or 0)
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
        msg.body("You’re done for now 🎉 Say 'hi' if you want to add more bills.")
    else:
        msg.body("Say 'hi' to get started with your bill upload.")
    return str(resp)

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
# Weekly digest preview (HTML table)
# -----------------------------
@app.route("/digest_preview", methods=["GET"])
def digest_preview():
    phone = e164(request.args.get("phone", ""))
    if not phone:
        return "Missing ?phone=E164 (e.g., ?phone=%2B6145...)", 400

    k_bills = f"user:{phone}:bills"

    raw_entries = r.lrange(k_bills, 0, -1) or []
    entries = []
    for e in raw_entries:
        try:
            if isinstance(e, bytes):
                e = e.decode("utf-8", errors="ignore")
            entries.append(json.loads(e))
        except Exception:
            entries.append({"raw": str(e)})

    saved_count = len(entries)

    def fmt_size(b):
        if not b:
            return "—"
        try:
            b = int(b)
        except Exception:
            return str(b)
        if b < 1024:
            return f"{b} B"
        if b < 1024*1024:
            return f"{b/1024:.1f} KB"
        return f"{b/1024/1024:.2f} MB"

    rows = []
    for i, item in enumerate(entries, start=1):
        ts = item.get("ts")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "—"
        mtype = item.get("media_type", "—")
        ok = "yes" if item.get("downloaded_ok") else "no"
        err = item.get("download_err") or ""
        size = fmt_size(item.get("download_size_bytes"))
        rows.append(f"""
          <tr>
            <td style="padding:6px;border:1px solid #ddd;">{i}</td>
            <td style="padding:6px;border:1px solid #ddd;">{when}</td>
            <td style="padding:6px;border:1px solid #ddd;">{mtype}</td>
            <td style="padding:6px;border:1px solid #ddd;">{size}</td>
            <td style="padding:6px;border:1px solid #ddd;">{ok}</td>
            <td style="padding:6px;border:1px solid #ddd;max-width:280px;word-wrap:break-word;">{err}</td>
          </tr>
        """)

    table_html = (
        "<p>No bills saved yet for this number.</p>"
        if not rows else
        f"""
        <table style="border-collapse:collapse;border:1px solid #ddd;font-family:Arial,sans-serif;font-size:14px;">
          <thead>
            <tr style="background:#f7f7f7;">
              <th style="padding:6px;border:1px solid #ddd;">#</th>
              <th style="padding:6px;border:1px solid #ddd;">When</th>
              <th style="padding:6px;border:1px solid #ddd;">Type</th>
              <th style="padding:6px;border:1px solid #ddd;">Size</th>
              <th style="padding:6px;border:1px solid #ddd;">Fetched</th>
              <th style="padding:6px;border:1px solid #ddd;">Note</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        """
    )

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>SwitchBuddy – Weekly Digest Preview</title>
    </head>
    <body style="margin:24px;font-family:Arial,Helvetica,sans-serif;">
      <h2>Weekly Digest Preview</h2>
      <p><strong>Phone:</strong> {phone}</p>
      <p><strong>Bills saved:</strong> {saved_count}</p>
      {table_html}
      <hr>
      <p style="color:#666;">(Sandbox preview – data reflects your current uploads.)</p>
    </body>
    </html>
    """
    return html, 200

# -----------------------------
# Entrypoint (local dev)
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
