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

@app.route("/_routes", methods=["GET"], endpoint="_routes_dump")
def _routes_dump():
    routes = []
    for r in app.url_map.iter_rules():
        methods = sorted(m for m in r.methods if m not in {"HEAD", "OPTIONS"})
        routes.append({"rule": str(r.rule), "endpoint": r.endpoint, "methods": methods})
    return {"routes": routes}, 200


# One global place to set parser/build version

PARSER_VERSION = "SBY-2025-08-18-parse7"

# version ping
@app.route("/version", methods=["GET"], endpoint="version_ping")
def version():

    return {"parser_version": PARSER_VERSION}, 200


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
    Parse a block of bill text into structured fields.
    Priority:
      1) TIERED (Step/First/Next/Remaining) -> tariff_type=TIERED
      2) TOU (peak/shoulder/off-peak) ONLY if >=2 distinct TOU rates
      3) FLAT/anytime fallback
    Totals prefer: 'New charges', 'Total for this bill', 'Amount due', 'Total balance'.
    """
    import re
    from datetime import datetime

    # -------- helpers --------
    def _clean_num(s: str) -> float:
        s = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
        try:
            return float(s)
        except Exception:
            return 0.0

    def _to_cents(value: float, unit: str) -> float:
        unit = (unit or "").strip().lower()
        if unit in ("$", "aud", "usd", "nz$", "£", "€"):
            return value * 100.0
        # treat c / ¢ as already cents
        return value

    # Normalise spacing and split
    txt_norm = re.sub(r"[ \t]+", " ", txt)
    lines = [ln.strip() for ln in txt_norm.splitlines() if ln.strip()]
    lower = [ln.lower() for ln in lines]

    result: dict = {}

    # Common tokens (OCR-friendly)
    KWH_TOKEN = r"k[\s\.\-·]*w[\s\.\-·]*h"        # matches: kwh, k w h, kW·h, etc.
    PER_TOKEN = r"(?:/|per)\s*"                   # "/" or "per"
    MONEY = re.compile(r"\$\s*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)")
    KWH_NUM = re.compile(rf"(?<![\d\.])(\d{{1,6}}(?:\.\d+)?)\s*{KWH_TOKEN}\b", re.IGNORECASE)
    RATE_ANY = re.compile(rf"(\d{{1,4}}(?:\.\d+)?)\s*([c¢]|\$)\s*{PER_TOKEN}{KWH_TOKEN}\b", re.IGNORECASE)

    # -------- 1) Billing period --------
    date_pat = re.compile(r"(?P<d>\d{1,2})[\/\-\s](?P<m>\d{1,2}|[A-Za-z]{3,9})[\/\-\s](?P<y>\d{2,4})", re.IGNORECASE)
    month_map = {
        'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
        'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
        'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
    }
    def _parse_date(m):
        d = int(m.group("d")); mm = m.group("m"); y = int(m.group("y"))
        if y < 100: y += 2000
        mo = int(mm) if mm.isdigit() else month_map.get(mm.lower(), 1)
        try: return datetime(y, mo, d)
        except Exception: return None

    period_found = False
    for i, ln in enumerate(lower):
        if any(k in ln for k in ("billing period", "for the period", "supply period", "usage period", "period from")):
            m_all = list(date_pat.finditer(lines[i]))
            if len(m_all) >= 2:
                ds, de = _parse_date(m_all[0]), _parse_date(m_all[-1])
                if ds and de and de >= ds:
                    result["period_start"] = ds.date().isoformat()
                    result["period_end"] = de.date().isoformat()
                    period_found = True
                    break
        if " from " in ln and " to " in ln:
            m_all = list(date_pat.finditer(lines[i]))
            if len(m_all) >= 2:
                ds, de = _parse_date(m_all[0]), _parse_date(m_all[-1])
                if ds and de and de >= ds:
                    result["period_start"] = ds.date().isoformat()
                    result["period_end"] = de.date().isoformat()
                    period_found = True
                    break
    if not period_found:
        candidates = []
        for i, ln in enumerate(lower):
            if "period" in ln or " from " in ln or " to " in ln:
                for m in date_pat.finditer(lines[i]):
                    d = _parse_date(m)
                    if d: candidates.append((i, d))
        if len(candidates) >= 2:
            candidates.sort(key=lambda x: (x[0], x[1]))
            ds = candidates[0][1]; de = max(candidates, key=lambda x: x[1])[1]
            if ds and de and de >= ds:
                result["period_start"] = ds.date().isoformat()
                result["period_end"] = de.date().isoformat()

    # -------- 2) Bill total (prefer explicit labels) --------
    preferred_keys = ("new charges", "total for this bill", "charges this bill", "amount due", "total balance")
    amounts = []
    for i, ln in enumerate(lower):
        if any(k in ln for k in preferred_keys):
            for m in MONEY.finditer(lines[i]):        # same line
                amt = _clean_num(m.group(1))
                if amt > 0: amounts.append((i, amt))
            if i + 1 < len(lines):                    # look-ahead line
                for m in MONEY.finditer(lines[i + 1]):
                    amt = _clean_num(m.group(1))
                    if amt > 0: amounts.append((i + 1, amt))
    if amounts:
        result["total_cost_inc_gst"] = round(max(amounts, key=lambda x: x[1])[1], 2)
    else:
        all_amts = [_clean_num(m.group(1)) for m in MONEY.finditer(txt)]
        if all_amts:
            result["total_cost_inc_gst"] = round(max(all_amts), 2)

    # -------- 3) TIERED (Step/First/Next/Remaining) BEFORE TOU --------
    step1_kw = re.compile(r"\b(step\s*1|first|block\s*1)\b", re.IGNORECASE)
    step2_kw = re.compile(r"\b(step\s*2|next|remaining|thereafter|block\s*2)\b", re.IGNORECASE)
    at_kw = re.compile(r"\b(at|@)\b", re.IGNORECASE)

    def _scan_step(window_text: str):
        """Find kWh and rate within a small window blob."""
        # kWh
        m_kwh = KWH_NUM.search(window_text)
        kwh = _clean_num(m_kwh.group(1)) if m_kwh else None
        # rate like '27.731 c/kWh' or '$0.27731 per kWh'
        m_rate = RATE_ANY.search(window_text)
        rate = round(_to_cents(_clean_num(m_rate.group(1)), m_rate.group(2)), 4) if m_rate else None
        return kwh, rate

    step1 = {}
    step2 = {}
    for i in range(len(lines)):
        # 3-line window to catch broken layouts
        blob = " ".join(lines[i:j] for j in range(i, min(i + 3, len(lines))))
        blob = lines[i] + " " + (lines[i + 1] if i + 1 < len(lines) else "") + " " + (lines[i + 2] if i + 2 < len(lines) else "")
        if not step1 and (step1_kw.search(lower[i]) or (i + 1 < len(lower) and step1_kw.search(lower[i + 1]))):
            k, r = _scan_step(blob)
            if k is not None: step1["kwh"] = k
            if r is not None: step1["rate_cents"] = r
        if not step2 and (step2_kw.search(lower[i]) or (i + 1 < len(lower) and step2_kw.search(lower[i + 1]))):
            k, r = _scan_step(blob)
            if k is not None: step2["kwh"] = k
            if r is not None: step2["rate_cents"] = r

    if step1 or step2:
        result["tariff_type"] = "TIERED"
        if "rate_cents" in step1: result["usage_cents_per_kwh_step1"] = step1["rate_cents"]
        if "kwh" in step1:        result["step1_kwh"] = round(step1["kwh"], 3)
        if "rate_cents" in step2: result["usage_cents_per_kwh_step2"] = step2["rate_cents"]
        if "kwh" in step2:        result["step2_kwh"] = round(step2["kwh"], 3)

    # -------- 4) TOU (require >=2 TOU rates) --------
    if "tariff_type" not in result:
        rate_any = rf"(?P<val>\d{{1,4}}(?:\.\d+)?)\s*(?P<unit>[c¢]|\$)\s*{PER_TOKEN}{KWH_TOKEN}\b"
        tou_specs = [
            ("usage_cents_per_kwh_peak", re.compile(r"\bpeak\b[^.\n]*?" + rate_any, re.IGNORECASE)),
            ("usage_cents_per_kwh_shoulder", re.compile(r"\bshoulder\b[^.\n]*?" + rate_any, re.IGNORECASE)),
            ("usage_cents_per_kwh_offpeak", re.compile(r"(?:off-?peak|off\s*peak)[^.\n]*?" + rate_any, re.IGNORECASE)),
        ]
        found = 0
        for key, pat in tou_specs:
            for ln in lines:
                m = pat.search(ln)
                if m:
                    val = _to_cents(float(m.group("val")), m.group("unit"))
                    result[key] = round(val, 4)
                    found += 1
                    break
        if found >= 2:
            result["tariff_type"] = "TOU"

    # -------- 5) FLAT fallback --------
    if "tariff_type" not in result:
        flat_pat = re.compile(rf"(anytime|single\s*rate|general|flat|usage|energy)[^.\n]*?(\d{{1,4}}(?:\.\d+)?)\s*([c¢]|\$)\s*{PER_TOKEN}{KWH_TOKEN}\b", re.IGNORECASE)
        for ln in lines:
            m = flat_pat.search(ln)
            if m:
                val = _to_cents(float(m.group(2)), m.group(3))
                result["tariff_type"] = "FLAT"
                result["usage_cents_per_kwh"] = round(val, 4)
                break

    # -------- 6) Supply (c/day or $/day with weird spacing) --------
    supply_pat = re.compile(
        r"(supply|daily)\s+(charge|service|fixed)[^.\n]*?(\d{1,4}(?:\.\d+)?)\s*([c¢]|\$)\s*(?:/|per)?\s*(day|d)\b",
        re.IGNORECASE,
    )
    for ln in lines:
        m = supply_pat.search(ln)
        if m:
            result["supply_cents_per_day"] = round(_to_cents(float(m.group(3)), m.group(4)), 4)
            break

    # -------- 7) Total kWh --------
    total_kwh = None
    kwh_pref = re.compile(r"(total usage|usage this bill|electricity used|total kwh|energy used)", re.IGNORECASE)
    for i, ln in enumerate(lower):
        if kwh_pref.search(ln):
            m = KWH_NUM.search(lines[i])
            if m:
                total_kwh = _clean_num(m.group(1)); break
            if i + 1 < len(lines):
                m = KWH_NUM.search(lines[i + 1])
                if m:
                    total_kwh = _clean_num(m.group(1)); break

    if total_kwh is None and ("step1_kwh" in result or "step2_kwh" in result):
        total_kwh = 0.0
        if "step1_kwh" in result: total_kwh += float(result["step1_kwh"])
        if "step2_kwh" in result: total_kwh += float(result["step2_kwh"])

    if total_kwh is None:
        m_all = [m for m in KWH_NUM.finditer(txt)]
        if m_all:
            total_kwh = max(_clean_num(m.group(1)) for m in m_all)

    if total_kwh is not None:
        result["total_kwh"] = round(total_kwh, 3)

    # -------- 8) Derive daily_kwh --------
    if "total_kwh" in result and "period_start" in result and "period_end" in result:
        try:
            ds = datetime.fromisoformat(result["period_start"])
            de = datetime.fromisoformat(result["period_end"])
            days = max((de - ds).days, 1)
            result["daily_kwh"] = round(float(result["total_kwh"]) / days, 2)
        except Exception:
            pass

    result.setdefault("tariff_type", "FLAT")
    return result


    # If not found, sum steps if available
    if total_kwh is None and ("step1_kwh" in result or "step2_kwh" in result):
        total_kwh = 0.0
        if "step1_kwh" in result:
            total_kwh += float(result["step1_kwh"])
        if "step2_kwh" in result:
            total_kwh += float(result["step2_kwh"])

    # Last resort: highest explicit kWh anywhere (guarded by 'kwh' token)
    if total_kwh is None:
        m_all = [m for m in kwh_num.finditer(txt)]
        if m_all:
            total_kwh = max(_clean_num(m.group(1)) for m in m_all)

    if total_kwh is not None:
        result["total_kwh"] = round(total_kwh, 3)

    # -------- 8) Derive daily_kwh --------
    if "total_kwh" in result and "period_start" in result and "period_end" in result:
        try:
            ds = datetime.fromisoformat(result["period_start"])
            de = datetime.fromisoformat(result["period_end"])
            days = max((de - ds).days, 1)
            result["daily_kwh"] = round(float(result["total_kwh"]) / days, 2)
        except Exception:
            pass

    # Conservative default
    result.setdefault("tariff_type", "FLAT")
    return result

# ---------- Bytes -> structured bill profile ----------
def parse_bill_bytes(content: bytes, content_type: str) -> dict:
    """
    Extract text from PDF/image bytes, parse with parse_bill_text,
    and fall back to a tiny demo profile if OCR found nothing.
    """
    txt = ocr_extract_text(content, content_type)
    parsed = parse_bill_text(txt) if txt else {}

    if parsed:
        # Be conservative: if tariff_type is still unset, default to FLAT rather than TOU.
        parsed.setdefault("tariff_type", parsed.get("tariff_type") or "FLAT")
        return parsed

    # Fallback so the pipeline still runs
    return {
        "period_start": "2025-06-01",
        "period_end": "2025-08-01",
        "total_kwh": 1200,
        "daily_kwh": round(1200 / 62, 2),
        "supply_cents_per_day": 95,
        "usage_cents_per_kwh": 32,
        "tariff_type": "FLAT",
        "total_cost_inc_gst": 540.00,
    }


# ---------- Digest HTML (simple) ----------
def _build_digest_html(phone: str) -> str:
    k_bills = f"user:{phone}:bills"
    entries = r.lrange(k_bills, 0, -1) or []
    total_bills = len(entries)

    preview_items = []
    for raw in entries[-5:]:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            j = json.loads(raw)
        except Exception:
            j = {"media_type": "unknown", "ts": 0}
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(j.get("ts", 0)))
        mtype = (j.get("media_type") or "").split("/")[-1].upper()
        preview_items.append(f"<li>{when} — {mtype or 'UNKNOWN'}</li>")

    savings = 100.0
    try:
        q = request.args.get("savings")
        if q:
            savings = float(q)
    except Exception:
        pass

    return f"""<!doctype html>
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
          <p>Projected annual savings: <strong>${savings:,.0f}</strong></p>
        </div>

        <div class="panel">
          <h2>Latest uploads</h2>
          {("<ul>" + "".join(preview_items) + "</ul>") if preview_items else "<p class='muted'>No recent bills.</p>"}
        </div>
      </div>

      <h2 style="margin-top:16px">Recommendation</h2>
      <p>Based on your recent usage and current market rates, you look suited to a <strong>low daily charge</strong> plan.</p>
      <p class="muted">(*Demo text*) This will refine once TOU/tiers are parsed from your bills.</p>

      <p style="margin-top:16px">
        <a class="cta cta-primary" href="#">Switch plan</a>
        <a class="cta" href="#">See full comparison</a>
      </p>
    </div>
  </div>
</body>
</html>"""


# ---------- Comparison stubs (so 'that’s all' & digest work) ----------
def fetch_tariffs_vic():
    return [
        {"name": "Flat Saver", "tariff_type": "FLAT", "supply_cents_per_day": 105, "usage_cents_per_kwh": 32},
        {"name": "TOU Value",  "tariff_type": "TOU",  "supply_cents_per_day": 95,  "peak_cents": 42, "shoulder_cents": 28, "offpeak_cents": 20},
        {"name": "Daily Low",  "tariff_type": "FLAT", "supply_cents_per_day": 80,  "usage_cents_per_kwh": 36},
    ]


def estimate_annual_cost(profile: dict, plan: dict) -> float:
    total_kwh = float(profile.get("total_kwh") or 0)
    start = profile.get("period_start")
    end = profile.get("period_end")

    days = 62
    try:
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
        daily_kwh = 10

    annual_kwh = daily_kwh * 365
    supply_cents = float(plan.get("supply_cents_per_day") or 0) * 365

    if plan.get("tariff_type") == "FLAT":
        usage_cents = annual_kwh * float(plan.get("usage_cents_per_kwh") or 0)
    else:
        peak = 0.5 * annual_kwh
        shoulder = 0.3 * annual_kwh
        offpeak = 0.2 * annual_kwh
        usage_cents = (
            peak * float(plan.get("peak_cents") or 0) +
            shoulder * float(plan.get("shoulder_cents") or 0) +
            offpeak * float(plan.get("offpeak_cents") or 0)
        )

    total_cents = supply_cents + usage_cents
    return round(total_cents / 100.0, 2)


def compare_and_store(phone: str) -> dict:
    k_bills = f"user:{phone}:bills"
    raw_entries = r.lrange(k_bills, 0, -1) or []

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
        latest_profile = {
            "total_kwh": 1200,
            "period_start": "2025-06-01",
            "period_end": "2025-08-01",
            "daily_kwh": round(1200/62, 2),
        }

    plans = fetch_tariffs_vic()
    scored = [{"plan": p, "annual_cost": estimate_annual_cost(latest_profile, p)} for p in plans]
    scored.sort(key=lambda x: x["annual_cost"])

    best = scored[0]
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
    # Returns the first few lines of the active parser, so we can confirm itâ€™s the new one
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

    # Parse with the code thatâ€™s live right now
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
            "â€¢ Reply: 'add another bill' to upload more\n"
            "â€¢ Reply: 'that's all' to finish\n"
            "â€¢ Reply: 'list bills' to see what I've saved\n"
            "â€¢ Reply: 'digest' to get your weekly digest link"
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
        msg.body("Cool â€” send the next bill photo/PDF. When finished, say 'that's all'.")
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
        # Tag with current parser version if defined
        try:
            entry["parser_version"] = PARSER_VERSION
        except NameError:
            pass

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
                f"Bill received âœ… (#{new_count}).\n"
                f"(Fetched media: {human_size})\n\n"
                "Reply 'add another bill' to add more, 'list bills' to review, or 'that's all' to finish."
            )
        else:
            msg.body(
                f"Bill received âœ… (#{new_count}).\n"
                "Heads up: I couldn't fetch the file from Twilio just now, but the link was saved. "
                "You can try sending it again, or proceed.\n\n"
                "Reply 'add another bill' to add more, 'list bills' to review, or 'that's all' to finish."
            )
        return str(resp)

    # Fallbacks based on state
    if state == "collecting":
        msg.body(
            "I'm ready â€” please send a photo/PDF of your bill.\n"
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


@app.route("/purge_bills", methods=["POST", "GET"])
def purge_bills():
    phone = e164(request.args.get("phone", "") or request.values.get("phone", ""))
    if not phone:
        return {"error": "Missing ?phone=E164 (encode + as %2B)"}, 400
    k_bills = f"user:{phone}:bills"
    r.delete(k_bills)
    r.delete(f"user:{phone}:bill_count")
    r.delete(f"user:{phone}:last_comparison")
    return {"ok": True}, 200

@app.route("/reparse_last_from_ocr", methods=["GET", "POST"])
def reparse_last_from_ocr():
    """
    Re-parse the last saved bill for this phone using OCR text already stored
    (ocr_excerpt). If not present, will try to re-download from Twilio and OCR.
    Returns JSON with the new parsed block. Does NOT crash on common edge cases.
    """
    try:
        phone = e164(request.args.get("phone", ""))
        if not phone:
            return {"error": "Missing ?phone=+E164 (encode + as %2B)"}, 400

        k_bills = f"user:{phone}:bills"
        last = r.lindex(k_bills, -1)
        if not last:
            return {"error": "No bills found for this phone."}, 404

        if isinstance(last, bytes):
            last = last.decode("utf-8", errors="ignore")

        try:
            j = json.loads(last)
        except Exception as e:
            return {"error": f"Corrupt last bill JSON: {type(e).__name__}: {e}", "raw": last[:400]}, 500

        # Prefer the stored OCR text (fast, avoids expired Twilio URLs)
        source_text = j.get("ocr_excerpt") or ""

        # If no OCR text, try to re-download and OCR (may fail if Twilio URL expired)
        if not source_text:
            media_url = j.get("media_url")
            media_type = j.get("media_type") or ""
            if media_url:
                try:
                    content, fetched_type, _ = download_twilio_media(media_url)
                    source_text = ocr_extract_text(content, fetched_type or media_type)
                except Exception as ex:
                    return {
                        "error": "No ocr_excerpt on last bill and media re-download failed.",
                        "details": f"{type(ex).__name__}: {ex}",
                    }, 422
            else:
                return {"error": "No ocr_excerpt and no media_url on last bill record."}, 422

        # Re-parse using the OCR text with the latest parser
        new_parsed = parse_bill_text(source_text) if source_text else {}

        # If we derived TIERED kWh parts but no total, sum them for convenience
        if new_parsed and "total_kwh" not in new_parsed:
            total = 0.0
            for key in ("step1_kwh", "step2_kwh", "step3_kwh"):
                if key in new_parsed:
                    try:
                        total += float(new_parsed[key])
                    except Exception:
                        pass
            if total > 0:
                new_parsed["total_kwh"] = round(total, 3)

        # Save back to Redis
        j["parsed"] = new_parsed
        j["parser_version"] = PARSER_VERSION
        r.lset(k_bills, -1, json.dumps(j))

        # Also stash a comparison snapshot (optional, safe if it fails)
        try:
            r.set(f"user:{phone}:last_comparison", json.dumps(compare_and_store(phone)))
        except Exception:
            pass

        return {"reparsed": new_parsed, "parser_version": PARSER_VERSION}, 200

    except Exception as e:
        # never 500 with a blank body — show something helpful
        return {"error": f"Unexpected failure: {type(e).__name__}: {e}"}, 500


# -----------------------------
# Entrypoint (local dev)
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting SwitchBuddy on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

