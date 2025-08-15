# switchbuddy_sandbox.py

import os
import time
import json
import requests
from flask import Flask, request, Markup
from twilio.twiml.messaging_response import MessagingResponse
from upstash_redis import Redis

# -----------------------------
# Redis (Upstash) client
# -----------------------------
UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")

if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise RuntimeError("Missing UPSTASH_URL or UPSTASH_TOKEN environment variables.")

r = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

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
        pref = ""
        if UPSTASH_URL and ".upstash.io" in UPSTASH_URL:
            pref = UPSTASH_URL[: UPSTASH_URL.find(".upstash.io") + len(".upstash.io")]
        return {
            "connected": True,
            "error": None,
            "has_token": bool(UPSTASH_TOKEN),
            "has_url": bool(UPSTASH_URL),
            "ping": pong,
            "url_prefix": pref
        }, 200
    except Exception as e:
        return {
            "connected": False,
            "error": f"{type(e).__name__}: {e}",
            "has_token": bool(UPSTASH_TOKEN),
            "has_url": bool(UPSTASH_URL),
        }, 500

# -----------------------------
# Twilio media download helper
# -----------------------------
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

    max_bytes = 10 * 1024 * 1024  # 10 MB soft cap for sandbox
    with requests.get(media_url, auth=(sid, token), stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        content_len_header = resp.headers.get("Content-Length")
        content_length = int(content_len_header) if content_len_header and content_len_header.isdigit() else None

        chunks, total = [], 0
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Media exceeds {max_bytes} bytes limit for sandbox.")
        content = b"".join(chunks)

    return content, content_type, (content_length if content_length is not None else len(content))

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
# Helper: normalize user text
# -----------------------------
def _normalize_text(s: str) -> str:
    if not s:
        return ""
    normalized = (
        s.strip()
         .replace("\u2019", "'").replace("\u2018", "'")
         .replace("\u201C", '"').replace("\u201D", '"')
         .replace("\u2013", "-").replace("\u2014", "-")
    )
    return " ".join(normalized.split()).lower().strip(" .!?,;:'\"-")

# -----------------------------
# WhatsApp webhook
# -----------------------------
@app.route("/whatsapp/webhook", methods=["POST"])
def whatsapp_webhook():
    from_number = (request.values.get("From") or "").replace("whatsapp:", "")
    raw = (request.values.get("Body") or "")
    incoming_msg = _normalize_text(raw)

    # Per-user keys
    k_state = f"user:{from_number}:state"        # idle|collecting|done
    k_count = f"user:{from_number}:bill_count"   # integer
    k_bills = f"user:{from_number}:bills"        # LIST of JSON entries

    # Init defaults
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
            "Send a photo or PDF of your bill.\n\n"
            "• 'add another bill' to upload more\n"
            "• 'that's all' to finish\n"
            "• 'list bills' to see saved files\n"
            "• 'digest' to get your weekly summary"
        )
        return str(resp)

    # Finish
    if said_any("that's all", "thats all", "done", "finish", "finished", "all done"):
        r.set(k_state, "done")
        count = int(r.get(k_count) or 0)
        msg.body(
            f"All set! I’ve saved {count} bill(s).\n"
            "I’ll crunch the numbers and include them in your weekly digest. "
            "You can say 'digest' any time to preview it."
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
                j = json.loads(e)
                media_type = (j.get("media_type") or "").lower()
                kind = "PDF" if media_type.endswith("pdf") or "pdf" in media_type else "Image"
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(j.get("ts", 0)))
                lines.append(f"#{i}: {kind} @ {ts}")
            except Exception:
                lines.append(f"#{i}: (unparsed)")
        msg.body("Recent saved bills:\n" + "\n".join(lines))
        return str(resp)

    # Digest (summary link + quick facts)
    if said_any("digest", "summary"):
        entries = r.lrange(k_bills, 0, -1) or []
        count = len(entries)
        last = entries[-1] if entries else None
        last_type = last_ts = last_size = last_fetched = None
        if last:
            try:
                j = json.loads(last)
                last_type = j.get("media_type") or ""
                last_ts = j.get("ts")
                last_size = j.get("download_size_bytes")
                last_fetched = j.get("downloaded_ok")
            except Exception:
                pass

        phone_num = from_number.lstrip("+")
        phone_param = f"%2B{phone_num}"
        preview_url = f"https://{request.host}/digest_preview?phone={phone_param}"

        facts = [
            f"Total bills saved: {count}",
            f"Last bill uploaded: {time.strftime('%Y-%m-%d %H:%M', time.localtime(last_ts)) if last_ts else 'n/a'}",
            f"Last bill type: {last_type or 'n/a'}",
            f"Fetched from Twilio: {'yes' if last_fetched else 'no' if last_fetched is not None else 'n/a'}",
            f"Last file size: {f'{last_size/1024:.0f} KB' if isinstance(last_size, int) else 'n/a'}",
        ]
        msg.body("Weekly Digest (preview)\n" + "\n".join(f"• {x}" for x in facts) + f"\n\n{preview_url}")
        return str(resp)

    # Add another
    if said_any("add another bill", "add another", "another bill", "add more", "upload another"):
        r.set(k_state, "collecting")
        msg.body("Cool — send the next bill photo/PDF. When finished, say “that's all”.")
        return str(resp)

    # --- Media handling ---
    media_url = request.values.get("MediaUrl0")
    media_type = request.values.get("MediaContentType0")

    if state == "collecting" and media_url:
        downloaded_ok = False
        size_bytes = None
        fetched_type = None
        err = None
        try:
            content, fetched_type, size_bytes = download_twilio_media(media_url)
            downloaded_ok = True
            # (Not persisting the file yet in sandbox)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        entry = {
            "media_url": media_url,
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
            human_size = (
                f"{size_bytes/1024:.1f} KB" if size_bytes and size_bytes < 1024*1024
                else (f"{size_bytes/1024/1024:.2f} MB" if size_bytes else "unknown size")
            )
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

    # Fallbacks
    if state == "collecting":
        msg.body("I’m ready — send a photo/PDF of your bill, or say 'that's all' when finished.")
    elif state == "done":
        msg.body("You’re done for now 🎉 Say 'hi' if you want to add more bills, or 'digest' for a summary.")
    else:
        msg.body("Say 'hi' to get started with your bill upload.")
    return str(resp)

# -----------------------------
# Digest preview page (HTML)
# -----------------------------
@app.route("/digest_preview", methods=["GET"])
def digest_preview():
    phone = (request.args.get("phone", "") or "").lstrip("+")
    if not phone:
        return "Missing ?phone=%2B<digits>", 400

    k_bills = f"user:+{phone}:bills"
    entries = r.lrange(k_bills, 0, -1) or []
    count = len(entries)

    last = entries[-1] if entries else None
    facts = {
        "Total bills saved": count,
        "Last bill uploaded": "n/a",
        "Last bill type": "n/a",
        "Fetched from Twilio": "n/a",
        "Last file size": "n/a"
    }
    if last:
        try:
            j = json.loads(last)
            ts = j.get("ts")
            facts["Last bill uploaded"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "n/a"
            mt = j.get("media_type") or ""
            facts["Last bill type"] = mt or "n/a"
            dlok = j.get("downloaded_ok")
            facts["Fetched from Twilio"] = "yes" if dlok else "no" if dlok is not None else "n/a"
            sz = j.get("download_size_bytes")
            facts["Last file size"] = (f"{sz/1024:.0f} KB" if isinstance(sz, int) else "n/a")
        except Exception:
            pass

    # Simple HTML (unstyled on purpose for now)
    rows = "\n".join(f"<li><b>{Markup.escape(k)}:</b> {Markup.escape(str(v))}</li>" for k, v in facts.items())
    html = f"""
    <html>
    <head><title>Weekly Digest (Preview)</title></head>
    <body>
      <h2>Weekly Digest (Preview)</h2>
      <p>for +{phone}</p>
      <ul>
        {rows}
      </ul>
      <p>Next step: we’ll parse usage from PDFs/images and compare plan options.</p>
    </body>
    </html>
    """
    return html, 200

# -----------------------------
# Admin/testing helpers
# -----------------------------
@app.route("/user_bills", methods=["GET"])
def user_bills():
    phone = request.args.get("phone", "").strip()
    if not phone:
        return {"error": "Missing ?phone=+61..."}, 400
    k_bills = f"user:{phone}:bills"
    entries = r.lrange(k_bills, 0, -1) or []
    parsed = []
    for e in entries:
        try:
            parsed.append(json.loads(e))
        except Exception:
            parsed.append({"raw": e})
    return {"phone": phone, "count": len(parsed), "bills": parsed}, 200

@app.route("/last_bill", methods=["GET"])
def last_bill():
    phone = request.args.get("phone", "").strip()
    if not phone:
        return {"error": "Missing ?phone=+61..."}, 400
    k_bills = f"user:{phone}:bills"
    last = r.lindex(k_bills, -1)
    if not last:
        return {"phone": phone, "last_bill": None}, 200
    try:
        return {"phone": phone, "last_bill": json.loads(last)}, 200
    except Exception:
        return {"phone": phone, "last_bill": {"raw": last}}, 200

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
