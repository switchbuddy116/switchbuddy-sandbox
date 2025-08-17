# switchbuddy_sandbox.py

import os
import io
import re
import time
import json
import requests
from urllib.parse import quote
from datetime import datetime

from flask import Flask, request, Response, redirect
from twilio.twiml.messaging_response import MessagingResponse
from upstash_redis import Redis

# ----- Optional OCR libs -----
# These imports are optional; we check availability at runtime.
try:
    from google.cloud import vision as gcv
    _HAS_GCV = True
except Exception:
    _HAS_GCV = False

try:
    import pytesseract
    from PIL import Image
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

try:
    from PyPDF2 import PdfReader
    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False

# -----------------------------
# Redis (Upstash) client
# -----------------------------
UPSTASH_URL = os.environ.get("UPSTASH_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_TOKEN")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise RuntimeError("Missing UPSTASH_URL or UPSTASH_TOKEN environment variables.")

r = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)

# Your public base URL (e.g., https://switchbuddy-sandbox.onrender.com)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# -----------------------------
# Google Vision bootstrap (optional)
# -----------------------------
def _maybe_bootstrap_gcv() -> bool:
    """
    If GOOGLE_APPLICATION_CREDENTIALS_JSON is set, write it to /tmp and
    point GOOGLE_APPLICATION_CREDENTIALS there so google-cloud-vision works.
    """
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json:
        return False
    try:
        path = "/tmp/gcp_creds.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        return True
    except Exception:
        return False

_GCV_READY = False
if _HAS_GCV:
    _GCV_READY = _maybe_bootstrap_gcv()

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Helpers
# -----------------------------
def e164(raw: str) -> str:
    """Normalize a WhatsApp/phone value to +E164 for Redis keys."""
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
    """
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        raise RuntimeError("Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN.")

    max_bytes = 10 * 1024 * 1024  # 10 MB cap
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
                    raise ValueError(f"Media exceeds {max_bytes} bytes limit.")
        content = b"".join(chunks)

    return content, content_type, (content_length if content_length is not None else len(content))

# ---------- OCR & parsing pipeline ----------
def ocr_extract_text(content: bytes, content_type: str) -> str:
    """
    Extract text from bytes using:
      1) PDF text via PyPDF2 (if PDF)
      2) Google Cloud Vision OCR (if set up)
      3) Tesseract OCR (if available)
    Returns a (possibly empty) string.
    """
    ct = (content_type or "").lower()

    # PDFs: try direct text first (works for text-based PDFs)
    if "pdf" in ct and _HAS_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(content))
            texts = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                    if t:
                        texts.append(t)
                except Exception:
                    pass
            merged = "\n".join(texts).strip()
            if merged:
                return merged
        except Exception:
            pass  # fall through to OCR if available

    # Google Vision OCR
    if _HAS_GCV and _GCV_READY:
        try:
            client = gcv.ImageAnnotatorClient()
            gimg = gcv.Image(content=content)
            resp = client.document_text_detection(image=gimg)
            if resp and resp.full_text_annotation and resp.full_text_annotation.text:
                return resp.full_text_annotation.text
        except Exception:
            pass

    # Tesseract OCR for common image types (and as a last resort, a PDF first page image)
    if _HAS_TESS:
        try:
            if os.environ.get("TESSERACT_CMD"):
                pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
            img = Image.open(io.BytesIO(content))
            txt = pytesseract.image_to_string(img)
            if txt:
                return txt
        except Exception:
            pass

    # Nothing worked
    return ""

def parse_bill_text(txt: str) -> dict:
    """
    Parse a block of bill text into structured fields using line-level context.
    Robust handling for tiered 'Step 1/Step 2' even when OCR inserts spaces,
    plus kWh@rate pairs across line wraps. Prefers TIERED if any step markers appear.
    """
    def _clean_num(s: str) -> float:
        s = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
        try:
            return float(s)
        except Exception:
            return 0.0

    def _to_cents(value: float, unit: str) -> float:
        unit = (unit or "").strip()
        if unit in ("$", "usd", "aud", "nz$", "£", "€"):
            return value * 100.0  # dollars -> cents
        return value

    # Normalise spacing
    txt_norm = re.sub(r"[ \t]+", " ", txt)
    lines = [ln.strip() for ln in txt_norm.splitlines() if ln.strip()]
    lower = [ln.lower() for ln in lines]
    nospace_lower = [re.sub(r"\s+", "", ln).lower() for ln in lines]

    out: dict = {}

    # ---------- 1) Billing period ----------
    date_pat = re.compile(
        r"(?P<d>\d{1,2})[\/\-\s](?P<m>\d{1,2}|[A-Za-z]{3,9})[\/\-\s](?P<y>\d{2,4})",
        re.IGNORECASE,
    )
    month_map = {
        'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
        'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
        'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
    }

    def _parse_date(m):
        d = int(m.group("d"))
        mm = m.group("m")
        y = int(m.group("y"))
        if y < 100:
            y += 2000
        mo = int(mm) if mm.isdigit() else month_map.get(mm.lower(), 1)
        try:
            return datetime(y, mo, d)
        except Exception:
            return None

    period_found = False
    for i, ln in enumerate(lower):
        if any(k in ln for k in ("billing period", "for the period", "supply period", "usage period", "period from")):
            m_all = list(date_pat.finditer(lines[i]))
            if len(m_all) >= 2:
                ds, de = _parse_date(m_all[0]), _parse_date(m_all[-1])
                if ds and de and de >= ds:
                    out["period_start"] = ds.date().isoformat()
                    out["period_end"] = de.date().isoformat()
                    period_found = True
                    break
        if " from " in ln and " to " in ln:
            m_all = list(date_pat.finditer(lines[i]))
            if len(m_all) >= 2:
                ds, de = _parse_date(m_all[0]), _parse_date(m_all[-1])
                if ds and de and de >= ds:
                    out["period_start"] = ds.date().isoformat()
                    out["period_end"] = de.date().isoformat()
                    period_found = True
                    break
    if not period_found:
        candidates = []
        for i, ln in enumerate(lower):
            if "period" in ln or " from " in ln or " to " in ln:
                for m in date_pat.finditer(lines[i]):
                    d = _parse_date(m)
                    if d:
                        candidates.append((i, d))
        if len(candidates) >= 2:
            candidates.sort(key=lambda x: (x[0], x[1]))
            ds = candidates[0][1]
            de = max(candidates, key=lambda x: x[1])[1]
            if ds and de and de >= ds:
                out["period_start"] = ds.date().isoformat()
                out["period_end"] = de.date().isoformat()

    # ---------- 2) Detect if bill is tiered (OCR-safe) ----------
    step1_idxs = [i for i, ns in enumerate(nospace_lower) if "step1" in ns]
    step2_idxs = [i for i, ns in enumerate(nospace_lower) if "step2" in ns]
    prefer_tier = bool(step1_idxs or step2_idxs)

    # ---------- 3) Total kWh (generic headline) ----------
    kwh_token_pat = re.compile(r'(?<![\d\.])(?P<kwh>\d{1,6}(?:[,\d]{0,6})?(?:\.\d+)?)\s*kwh\b', re.IGNORECASE)
    total_kwh_val = None
    for i, ln in enumerate(lower):
        if any(k in ln for k in ("total usage", "electricity used", "usage this bill", "total kwh", "energy used", "your usage")):
            m = kwh_token_pat.search(lines[i])
            if m:
                total_kwh_val = _clean_num(m.group("kwh"))
                break
    if total_kwh_val is not None:
        out["total_kwh"] = round(total_kwh_val, 3)

    # ---------- 4) Supply charge (per day) ----------
    supply_pat = re.compile(
        r'(supply|daily)\s+(charge|service|fixed).*?(?P<val>\d{1,4}(?:\.\d+)?)\s*(?P<unit>[c¢]|\$)\s*\/?\s*(day|d)\b',
        re.IGNORECASE
    )
    for ln in lines:
        m = supply_pat.search(ln)
        if m:
            cents = _to_cents(float(m.group("val")), m.group("unit"))
            out["supply_cents_per_day"] = round(cents, 4)
            break

    # ---------- 5) Tier pairs & fallbacks ----------
    # kWh @ rate pairs (inline or wrapped)
    pair_pat_inline = re.compile(
        r'(?P<k>\d{1,6}(?:[,\d]{0,6})?(?:\.\d+)?)\s*kwh[^@\n]{0,60}(?:@|at|x)\s*(?P<r>\d{1,4}(?:\.\d+)?)\s*(?P<u>[c¢]|\$)\s*\/?\s*kwh',
        re.IGNORECASE
    )
    kwh_only_pat  = re.compile(r'(?P<k>\d{1,6}(?:[,\d]{0,6})?(?:\.\d+)?)\s*kwh\b', re.IGNORECASE)
    rate_only_pat = re.compile(r'(?P<r>\d{1,4}(?:\.\d+)?)\s*(?P<u>[c¢]|\$)\s*\/?\s*kwh', re.IGNORECASE)

    def _window_blob(i: int, span: int = 3) -> str:
        parts = []
        for j in range(i - span, i + span + 1):
            if 0 <= j < len(lines):
                parts.append(lines[j])
        return " ".join(parts)

    pairs = []
    # same-line
    for ln in lines:
        for m in pair_pat_inline.finditer(ln):
            k = _clean_num(m.group("k"))
            r = _to_cents(_clean_num(m.group("r")), m.group("u"))
            if k > 0 and r > 0:
                pairs.append((k, r))
    # wrapped (adjacent)
    for i in range(len(lines) - 1):
        blob = lines[i] + " " + lines[i+1]
        m_inline = pair_pat_inline.search(blob)
        if m_inline:
            k = _clean_num(m_inline.group("k"))
            r = _to_cents(_clean_num(m_inline.group("r")), m_inline.group("u"))
            if k > 0 and r > 0:
                pairs.append((k, r))
                continue
        mk = kwh_only_pat.search(lines[i]);   mr = rate_only_pat.search(lines[i+1])
        mk2 = kwh_only_pat.search(lines[i+1]); mr2 = rate_only_pat.search(lines[i])
        for (km, rm) in ((mk, mr), (mk2, mr2)):
            if km and rm:
                k = _clean_num(km.group("k"))
                r = _to_cents(_clean_num(rm.group("r")), rm.group("u"))
                if k > 0 and r > 0:
                    pairs.append((k, r))

    # de-dup
    deduped = []
    for k, r in pairs:
        if not any(abs(k - k2) < 0.01 and abs(r - r2) < 0.01 for (k2, r2) in deduped):
            deduped.append((k, r))
    pairs = deduped

    # If we see explicit step markers, prefer TIERED no matter what
    if prefer_tier:
        out["tariff_type"] = "TIERED"

    # Map pairs in order to steps
    if pairs:
        if "tariff_type" not in out:
            out["tariff_type"] = "TIERED"
        if len(pairs) >= 1:
            out["step1_kwh"] = round(pairs[0][0], 3)
            out["usage_cents_per_kwh_step1"] = round(pairs[0][1], 3)
        if len(pairs) >= 2:
            out["step2_kwh"] = round(pairs[1][0], 3)
            out["usage_cents_per_kwh_step2"] = round(pairs[1][1], 3)
        # If no explicit total_kwh or pair-sum looks more complete, use pair sum
        pair_sum = round(sum(k for k, _ in pairs), 3)
        if "total_kwh" not in out or pair_sum > float(out["total_kwh"]) + 0.5:
            out["total_kwh"] = pair_sum

    # Per-step nearest fallback if we have markers but missed one side (kWh or rate)
    def _nearest_kwh(idx: int) -> float | None:
        for j in range(max(0, idx-3), min(len(lines), idx+4)):
            mk = kwh_only_pat.search(lines[j])
            if mk:
                return _clean_num(mk.group("k"))
        return None

    def _nearest_rate_c(idx: int) -> float | None:
        for j in range(max(0, idx-3), min(len(lines), idx+4)):
            mr = rate_only_pat.search(lines[j])
            if mr:
                return _to_cents(_clean_num(mr.group("r")), mr.group("u"))
        return None

    if prefer_tier:
        # Step 1
        if step1_idxs:
            if "step1_kwh" not in out:
                k = _nearest_kwh(step1_idxs[0])
                if k is not None:
                    out["step1_kwh"] = round(k, 3)
            if "usage_cents_per_kwh_step1" not in out:
                r_c = _nearest_rate_c(step1_idxs[0])
                if r_c is not None:
                    out["usage_cents_per_kwh_step1"] = round(r_c, 3)
        # Step 2
        if step2_idxs:
            if "step2_kwh" not in out:
                k = _nearest_kwh(step2_idxs[0])
                if k is not None:
                    out["step2_kwh"] = round(k, 3)
            if "usage_cents_per_kwh_step2" not in out:
                r_c = _nearest_rate_c(step2_idxs[0])
                if r_c is not None:
                    out["usage_cents_per_kwh_step2"] = round(r_c, 3)
        # If total still missing, sum
        if "total_kwh" not in out:
            tot = (out.get("step1_kwh") or 0.0) + (out.get("step2_kwh") or 0.0)
            if tot > 0:
                out["total_kwh"] = round(tot, 3)

    # ---------- 6) TOU / FLAT fallback ONLY if we never declared TIERED ----------
    if out.get("tariff_type") is None:
        rate_any = r'(?P<val>\d{1,4}(?:\.\d+)?)\s*(?P<unit>[c¢]|\$)\s*\/?\s*kwh'
        tou_specs = [
            ("usage_cents_per_kwh_peak", re.compile(r'peak[^.\n]*?' + rate_any, re.IGNORECASE)),
            ("usage_cents_per_kwh_shoulder", re.compile(r'shoulder[^.\n]*?' + rate_any, re.IGNORECASE)),
            ("usage_cents_per_kwh_offpeak", re.compile(r'(off-?peak|off peak)[^.\n]*?' + rate_any, re.IGNORECASE)),
        ]
        found_tou = False
        for key, pat in tou_specs:
            for ln in lines:
                m = pat.search(ln)
                if m:
                    val = _to_cents(float(m.group("val")), m.group("unit"))
                    out[key] = round(val, 4)
                    found_tou = True
                    break
        if found_tou:
            out["tariff_type"] = "TOU"
        else:
            flat_pats = [
                re.compile(r'(anytime|single\s*rate|general|usage|energy)[^.\n]*?' + rate_any, re.IGNORECASE),
                re.compile(r'(?<!peak)(?<!off)[^.\n]*?' + rate_any, re.IGNORECASE),
            ]
            for pat in flat_pats:
                for ln in lines:
                    m = pat.search(ln)
                    if m:
                        val = _to_cents(float(m.group("val")), m.group("unit"))
                        out["tariff_type"] = "FLAT"
                        out["usage_cents_per_kwh"] = round(val, 4)
                        break
                if "usage_cents_per_kwh" in out:
                    break

    # ---------- 7) Total cost (incl GST) ----------
    money_pat = re.compile(r'(\$?\s*\d{1,6}(?:,\d{3})*\.\d{2})')  # must have cents
    preferred_cost_keys = ("current charges", "total for this bill", "electricity charges",
                           "charges this bill", "amount due", "new charges", "total balance")
    amounts = []
    for i, ln in enumerate(lower):
        if any(k in ln for k in preferred_cost_keys):
            for m in money_pat.finditer(lines[i]):
                amt = _clean_num(m.group(1))
                if amt > 0:
                    amounts.append((i, amt))
    if amounts:
        best = max(amounts, key=lambda x: x[1])[1]
        out["total_cost_inc_gst"] = round(best, 2)
    else:
        all_amts = [_clean_num(m.group(1)) for m in money_pat.finditer(txt)]
        if all_amts:
            out["total_cost_inc_gst"] = round(max(all_amts), 2)

    # ---------- 8) Derive daily_kwh ----------
    if "total_kwh" in out and "period_start" in out and "period_end" in out:
        try:
            ds = datetime.fromisoformat(out["period_start"])
            de = datetime.fromisoformat(out["period_end"])
            days = max((de - ds).days, 1)
            out["daily_kwh"] = round(float(out["total_kwh"]) / days, 2)
        except Exception:
            pass

    return out


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

# --- Debug & utility endpoints ---
VERSION = "SBY-2025-08-17-parse2"

import inspect
from hashlib import sha1

@app.route("/version", methods=["GET"])
def version():
    # Quick proof of which build is live
    src = inspect.getsource(parse_bill_text)
    sig = sha1(src.encode("utf-8")).hexdigest()[:12]
    return {
        "version": VERSION,
        "parse_bill_text_sha": sig,
        "has_TIERED_literal": ("TIERED" in src),
        "first_line": inspect.getsourcelines(parse_bill_text)[1],
        "num_lines": len(inspect.getsourcelines(parse_bill_text)[0]),
    }, 200

@app.route("/_debug/parser", methods=["GET"])
def debug_parser():
    # Returns the first few lines of the active parser, so we can confirm it’s the new one
    src = inspect.getsource(parse_bill_text)
    preview = "\n".join(src.splitlines()[:20])
    return {
        "starts_at": inspect.getsourcelines(parse_bill_text)[1],
        "preview_first_20_lines": preview,
    }, 200

@app.route("/force_reparse", methods=["POST","GET"])
def force_reparse():
    """
    Re-downloads the last bill for the given phone and re-parses it with the
    CURRENT code, writing a new entry. No WhatsApp re-send needed.
      Usage: /force_reparse?phone=%2B61XXXXXXXXX
    """
    phone = e164(request.args.get("phone", ""))
    if not phone:
        return {"error": "Missing ?phone=E164 (encode + as %2B)"}, 400

    k_bills = f"user:{phone}:bills"
    last_raw = r.lindex(k_bills, -1)
    if not last_raw:
        return {"error": "No bills saved for this phone"}, 404

    try:
        if isinstance(last_raw, bytes):
            last_raw = last_raw.decode("utf-8", errors="ignore")
        last = json.loads(last_raw)
    except Exception as e:
        return {"error": f"Corrupt last bill JSON: {e}"}, 500

    media_url = last.get("media_url")
    media_type = last.get("media_type")
    if not media_url:
        return {"error": "Last bill has no media_url"}, 500

    # Try fresh fetch (Twilio links can expire)
    try:
        content, fetched_type, size_bytes = download_twilio_media(media_url)
    except Exception as e:
        return {"error": f"Re-download failed: {type(e).__name__}: {e}"}, 502

    # Parse with the code that’s live right now
    parsed = parse_bill_bytes(content, fetched_type or (media_type or ""))
    ocr_text = ocr_extract_text(content, fetched_type or (media_type or "")) or ""
    ocr_excerpt = ocr_text[:2000] if ocr_text else None

    new_entry = {
        "media_url": media_url,
        "media_type": fetched_type or media_type or "",
        "ts": int(time.time()),
        "downloaded_ok": True,
        "download_err": None,
        "download_size_bytes": size_bytes,
        "parsed": parsed,
    }
    if ocr_excerpt:
        new_entry["ocr_excerpt"] = ocr_excerpt

    r.rpush(k_bills, json.dumps(new_entry))
    return {
        "status": "reparsed",
        "new_ts": new_entry["ts"],
        "parsed": parsed
    }, 200


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

# --- Debug: version ping ---
VERSION = "SBY-2025-08-17-parse2"

@app.route("/version", methods=["GET"])
def svc_version():
    return {"version": VERSION}, 200



# -----------------------------
# Weekly digest (HTML)
# -----------------------------
@app.route("/digest_preview", methods=["GET"])
def digest_preview():
    phone = request.args.get("phone", "").strip()
    if not phone:
        return "Missing ?phone=E164 (e.g., ?phone=%2B6145...)", 400
    phone = e164(phone)
    html = _build_digest_html(phone)
    return Response(html, mimetype="text/html")

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
    if said_any("hi", "hello", "start"):
        r.set(k_state, "collecting")
        msg.body(
            "Hi! I'm SwitchBuddy.\n\n"
            "Please send a photo or PDF of your electricity bill.\n\n"
            "When you're finished:\n"
            "• Reply: 'add another bill' to upload more\n"
            "• Reply: 'that's all' to finish\n"
            "• Reply: 'list bills' to see what I've saved\n"
            "• Reply: 'digest' to get your weekly digest link"
        )
        return str(resp)

    if said_any("digest", "show digest", "weekly digest"):
        if not PUBLIC_BASE_URL:
            msg.body("Digest preview is not available yet (missing PUBLIC_BASE_URL).")
            return str(resp)
        digest_url = f"{PUBLIC_BASE_URL}/digest_preview?phone={quote(phone, safe='')}"
        msg.body(f"Here's your digest preview:\n{digest_url}")
        return str(resp)

    if said_any("that's all", "thats all", "done", "finish", "finished", "all done"):
        r.set(k_state, "done")
        count = int(r.get(k_count) or 0)

        cmp = compare_and_store(phone)
        savings = cmp.get("projected_savings", 0.0)

        if PUBLIC_BASE_URL:
            digest_url = f"{PUBLIC_BASE_URL}/digest_preview?phone={quote(phone, safe='')}"
            msg.body(
                f"All set! I've saved {count} bill(s).\n"
                f"Projected annual savings: ${savings:,.0f}.\n\n"
                f"Preview your weekly digest here: {digest_url}\n\n"
                "If you want to add more later, just say 'hi'."
            )
        else:
            msg.body(
                f"All set! I've saved {count} bill(s).\n"
                f"Projected annual savings: ${savings:,.0f}.\n"
                "I'll include this in your weekly digest. If you want to add more later, just say 'hi'."
            )
        return str(resp)

    if said_any("list bills", "show bills", "what have you saved"):
        entries = r.lrange(k_bills, -10, -1) or []
        if not entries:
            msg.body("I haven't saved any bill files yet. Send a photo/PDF to add one.")
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

    if said_any("add another bill", "add another", "another bill", "add more", "upload another"):
        r.set(k_state, "collecting")
        msg.body("Cool — send the next bill photo/PDF. When finished, say 'that's all'.")
        return str(resp)

    # Media handling (Twilio sends MediaUrl0/MediaContentType0 when files attached)
    media_url = request.values.get("MediaUrl0")
    media_type = request.values.get("MediaContentType0")

    if state == "collecting" and media_url:
        downloaded_ok = False
        size_bytes = None
        fetched_type = None
        err = None
        parsed = None
        ocr_excerpt = None

        try:
            content, fetched_type, size_bytes = download_twilio_media(media_url)
            downloaded_ok = True

            # Parse the bill content into a structured profile (stub or real OCR)
            parsed = parse_bill_bytes(content, fetched_type or (media_type or ""))

            # Optional: save a short OCR excerpt for debugging (bounded to 2k chars)
            try:
                ocr_text = ocr_extract_text(content, fetched_type or (media_type or ""))
                if ocr_text:
                    ocr_excerpt = ocr_text[:2000]
            except Exception:
                pass

        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        entry = {
            "media_url": media_url,                 # Twilio temp URL (expires)
            "media_type": media_type or fetched_type or "",
            "ts": int(time.time()),
            "downloaded_ok": downloaded_ok,
            "download_err": err,
            "download_size_bytes": size_bytes,
        }
        if parsed:
            entry["parsed"] = parsed
        if ocr_excerpt:
            entry["ocr_excerpt"] = ocr_excerpt

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
                f"Bill received \u2705 (#{new_count}).\n"
                f"(Fetched media: {human_size})\n\n"
                "Reply 'add another bill' to add more, 'list bills' to review, or 'that's all' to finish."
            )
        else:
            msg.body(
                f"Bill received \u2705 (#{new_count}).\n"
                "Heads up: I couldn't fetch the file from Twilio just now, but the link was saved. "
                "You can try sending it again, or proceed.\n\n"
                "Reply 'add another bill' to add more, 'list bills' to review, or 'that's all' to finish."
            )
        return str(resp)

    # Fallbacks based on state
    if state == "collecting":
        msg.body(
            "I'm ready — please send a photo/PDF of your bill.\n"
            "Or say 'that's all' when you're finished."
        )
    elif state == "done":
        if PUBLIC_BASE_URL:
            digest_url = f"{PUBLIC_BASE_URL}/digest_preview?phone={quote(phone, safe='')}"
            msg.body(f"You're done for now.\nPreview your digest any time: {digest_url}\nSay 'hi' to add more bills.")
        else:
            msg.body("You're done for now. Say 'hi' if you want to add more bills.")
    else:
        msg.body("Say 'hi' to get started with your bill upload.")
    return str(resp)

# -----------------------------
# Admin/testing
# -----------------------------
@app.route("/user_bills", methods=["GET"])
def user_bills():
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
