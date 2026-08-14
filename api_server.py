import logging
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError

from scene_analysis import analyze_scene, InappropriateImageError

logger = logging.getLogger("daka")

# Frontend origins allowed to call this API. Production is the Netlify site; override with a
# comma-separated ALLOWED_ORIGINS in the Render dashboard to add a Netlify preview deploy or a
# local dev server (e.g. "https://daka-ai.netlify.app,http://localhost:8080"). Note that CORS is
# a browser-enforced policy, not access control — it stops other sites from spending our key
# through a user's browser, and does nothing against a direct curl.
DEFAULT_ALLOWED_ORIGINS = "https://daka-ai.netlify.app"
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",") if o.strip()
]

# A scan frame is ~1024px at quality 0.7, so well under 1 MB in practice. 8 MB leaves room for a
# file picked from the gallery while keeping a hostile upload from being buffered into memory on a
# free-tier instance.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic", "image/heif"}

# Cost control, not security: every /analyze costs two OpenAI calls (moderation + vision), so an
# unmetered endpoint is an open wallet. Per-IP, in-memory, and therefore reset by every cold start
# on the free plan — a speed bump against casual abuse, not a real quota. Move to a shared store
# if the service is ever scaled past one instance.
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
    # Bound the dict. Expiry above is lazy — it only runs for the IP being checked — so an IP seen
    # once and never again keeps its timestamp forever. Evict on the NEWEST hit being outside the
    # window rather than on the deque being empty, or a flood of one-shot IPs would never be
    # collected. Dropping that state is free: an IP with no recent hits cannot be rate-limited.
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


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
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
        return analyze_scene(tmp_path)
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
        logger.exception("analyze failed")
        return JSONResponse(
            status_code=500,
            content={"error": "Analysis failed on the server — try again in a moment."},
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
