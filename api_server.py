import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError

from scene_analysis import (
    DEFAULT_LANGUAGE,
    InappropriateImageError,
    analyze_scene,
    describe_scene,
    measure_scene,
)

logger = logging.getLogger("daka")

# Frontend origins allowed to call this API: the web build, then the Capacitor iOS and Android
# WebViews, which send capacitor://localhost and http://localhost as their Origin. A missing origin
# fails every scan as a generic network error, so the app reports "Analysis failed" with nothing
# pointing at the cause — and Cloudflare's per-build preview subdomains cannot be listed in advance.
#
# ALLOWED_ORIGINS in the Render dashboard REPLACES this list rather than extending it.
#
# CORS is browser-enforced, not access control: it stops other sites spending our key through a
# user's browser, and does nothing against a direct curl.
DEFAULT_ALLOWED_ORIGINS = ",".join([
    "https://dakaba.pages.dev",
    "capacitor://localhost",
    "http://localhost",
])
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",") if o.strip()
]

# A scan frame is ~1024px at quality 0.7, so well under 1 MB in practice. 8 MB leaves room for a
# file picked from the gallery while keeping a hostile upload from being buffered into memory on a
# free-tier instance.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic", "image/heif"}

# Cost control, not security: the paid path costs two OpenAI calls (moderation + vision), so an
# unmetered endpoint is an open wallet. Per-IP, in-memory, and therefore reset by every cold start
# on the free plan — a speed bump against casual abuse, not a real quota. Move to a shared store
# if the service is ever scaled past one instance.
#
# One scan is now two requests (/measure then /describe), so this allows ten scans a minute rather
# than twenty. That is still far above real use — nobody photographs a friend ten times a minute —
# and it halves rather than doubles the worst case spend, so the number stays where it is.
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_S = 60
_MAX_TRACKED_IPS = 4096

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_hits: "defaultdict[str, deque]" = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Render terminates TLS at a proxy, so request.client is the proxy.

    X-Forwarded-For is client-spoofable — fine for the rate limiter's purpose (slowing casual
    abuse) but never treat it as identity.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > RATE_LIMIT_WINDOW_S:
        window.popleft()
    if len(window) >= RATE_LIMIT_REQUESTS:
        return True
    # Bound the dict: expiry above is lazy, so an IP seen once keeps its timestamp forever. Evict
    # on the NEWEST hit being outside the window, not on the deque being empty, or a flood of
    # one-shot IPs is never collected. Dropping it is free — an IP with no recent hits cannot be
    # rate-limited.
    if len(_hits) > _MAX_TRACKED_IPS:
        for stale in [
            k for k, v in _hits.items()
            if k != ip and (not v or now - v[-1] > RATE_LIMIT_WINDOW_S)
        ]:
            del _hits[stale]
    window.append(now)
    return False


async def _read_capped(file: UploadFile) -> "bytes | None":
    """Read the upload in chunks, aborting past the cap. Returns None if too large.

    Reading in chunks matters: `await file.read()` would buffer the whole body first, so the cap
    would be checked only after the damage was done.
    """
    chunks, total = [], 0
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _scan(request: Request, file: UploadFile, run):
    """Guards, temp file and error mapping, shared by all three scan routes."""
    if _rate_limited(_client_ip(request)):
        return JSONResponse(
            status_code=429,
            content={"error": "Too many scans in a row — wait a moment, then scan again."},
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={"error": "That file isn't a supported image — use a JPEG, PNG or WebP."},
        )

    contents = await _read_capped(file)
    if contents is None:
        return JSONResponse(
            status_code=413,
            content={"error": "That image is too large — it must be under 8 MB."},
        )
    if not contents:
        return JSONResponse(status_code=400, content={"error": "The upload was empty."})

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
    with open(tmp_path, "wb") as f:
        f.write(contents)
    try:
        return run(tmp_path)
    except InappropriateImageError:
        # Must precede ValueError — InappropriateImageError subclasses it.
        return JSONResponse(status_code=400, content={"error": "Image not suitable for analysis"})
    except (UnidentifiedImageError, ValueError):
        # A truncated or mislabelled upload is the caller's problem, not a server fault.
        return JSONResponse(
            status_code=400,
            content={"error": "That image couldn't be read — try scanning again."},
        )
    except Exception:
        # Log the detail, return none of it: the old handler echoed str(e) and the exception class
        # name straight to the client.
        logger.exception("scan failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Analysis failed on the server — try again in a moment."},
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/measure")
async def measure(request: Request, file: UploadFile = File(...)):
    """The marker and the capture gates. Pixels only, so it answers in well under a second.

    No `lang`: nothing it returns is prose. No moderation either — the image is not sent anywhere,
    and what comes back is measurements of it rather than anything written about it.
    """
    return await _scan(request, file, measure_scene)


@app.post("/describe")
async def describe(request: Request, file: UploadFile = File(...),
                   lang: str = Form(DEFAULT_LANGUAGE), side: str = Form("")):
    """Scene name, filter, hashtags and the depth sentence. The slow half, on its own.

    `side` is where /measure already put the marker, handed back so the model's sentence agrees
    with the geometry instead of contradicting it. Anything but "left" or "right" is ignored and
    the prompt falls back to its own default, because a wrong side is worse than no side.
    """
    placement = {"left": {"x": 0.333}, "right": {"x": 0.667}}.get(side)
    return await _scan(request, file, lambda path: describe_scene(path, lang, placement))


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...),
                  lang: str = Form(DEFAULT_LANGUAGE)):
    """Both halves in one call. Still here because a cached copy of the old page still calls it,
    and because the file-picker path has no coaching to get on with and gains nothing from the
    split.

    `lang` is optional: older clients send none, and analyze_scene falls back on a value it does
    not know rather than rejecting a scan that has already been paid for.
    """
    return await _scan(request, file, lambda path: analyze_scene(path, lang))
