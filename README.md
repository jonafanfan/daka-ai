# 打卡吧！ (Dǎkǎ ba!)

A mobile web app that tells you **where to stand** for a good check-in photo.

Point the camera at an empty scene and tap. The backend analyses the frame — light direction,
visual balance, background clutter, blown-out regions — and returns the spot where a person should
stand for the best composition. The app draws a marker there, your subject stands on it, you shoot.

The guiding idea: **subject positioning, framing and basic colour grading are what make the photo.**
Posing is left to the person being photographed.

---

## Live

| | |
|---|---|
| Frontend | <https://daka-ai.netlify.app> (Netlify) |
| Backend | <https://daka-backend-9bfz.onrender.com> (Render, free plan) |
| Health check | `GET /health` → `{"status": "ok"}` |

The backend is on Render's free plan, so it **spins down when idle** and the first request after a
cold start takes ~50s. A cron job pings `/health` to keep it warm. The frontend already handles the
slow case with a 60s timeout and a "server may be waking up" message.

---

## The flow

```
Home  →  camera  →  point at an empty scene, tap shutter
                         ↓
                    POST /analyze  (frame capped at 1024px, JPEG q0.7)
                         ↓
         capture gate: reject if lighting is Poor or the frame is blurry
                         ↓
         standing marker drawn at the returned placement point
                         ↓
              subject stands on it  →  take the keeper photo
                         ↓
         results: filter preview, hashtag pills, share/save with watermark
```

---

## How it works

| Piece | File | Notes |
|---|---|---|
| Frontend (all of it) | [`web/index.html`](web/index.html) | ~850 lines, HTML + CSS + JS inline. No build step. |
| API | [`api_server.py`](api_server.py) | FastAPI. Two routes: `/health`, `POST /analyze`. |
| Engine | [`scene_analysis.py`](scene_analysis.py) | OpenCV measurements + one vision-model call. |
| Response contract | [`CONTRACT.md`](CONTRACT.md) | **Read this before changing the response shape.** |

### The engine is deliberately split in two

**OpenCV does the measurable half** — brightness, colour balance, Canny edge density, Laplacian
blur, spectral-residual saliency, Hough-line horizon tilt, and the placement heuristic. This is the
part that carries the product.

**The vision model does the subjective half** — scene name, filter choice, hashtags. Three fields,
and nothing else depends on them.

That split is why a vision-call failure returns a usable scan with `scene_type: "Unknown"` instead
of failing: the positioning advice never needed the model. One consequence for consumers — **a
`200` is not proof the model ran.** See [`CONTRACT.md`](CONTRACT.md) §3.6.

`_compute_placement` is the core of it: it fuses visual balance (stand opposite the scene's focal
mass), background cleanliness (prefer the emptier side), and light direction (stand on the dimmer
side so light falls on your face), plus a hard **backlight veto** so you're never placed in front of
a blown-out window. Each signal only votes when it's reliable for that scene.

---

## Local development

Requires **Python 3.13** (pinned in [`.python-version`](.python-version)).

### Backend

```bash
python -m venv .venv

# Windows
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

# run it
.venv/Scripts/python -m uvicorn api_server:app --reload
```

Serves on <http://127.0.0.1:8000>. It **starts without an API key** — the OpenAI client is
constructed lazily — so `/health` works immediately and you only need a key to actually scan:

```bash
export OPENAI_API_KEY=sk-...        # bash
$env:OPENAI_API_KEY = "sk-..."      # PowerShell
```

Never commit the key. `.env`, `.env.*`, `*.key` and `daka_openai_key.txt` are all gitignored.

### Frontend

It's one static file with no build step, but **don't open it with `file://`** — the camera and
sensor APIs need a secure context, and a `file://` page sends `Origin: null`, which CORS can never
match. Serve it over HTTP:

```bash
cd web && python -m http.server 8080
```

### Pointing the frontend at your local backend

Two things have to change, and neither is obvious:

1. **The API URL is hardcoded** at [`web/index.html:325`](web/index.html#L325). Change it to
   `http://localhost:8000`.
2. **CORS will reject your origin.** The backend only allows the production Netlify domain by
   default, so set this in the shell running your *local* backend:

   ```bash
   export ALLOWED_ORIGINS=http://localhost:8080
   ```

If you skip step 2, every scan fails with a generic network error — browsers don't report CORS
failures usefully, so the app just toasts "Analysis failed" with nothing pointing at the cause.
It's a confusing hour if you don't know to look for it.

### Testing on a phone

Camera, `devicemotion` and `deviceorientation` all need HTTPS on a real device. The path of least
resistance is to push a branch and use the Netlify deploy preview — but note that **preview URLs
are a different origin and are CORS-blocked** by the production backend, so a preview can load the
UI but not scan. To scan from a preview you'd need to add its URL to `ALLOWED_ORIGINS` in the Render
dashboard.

---

## Tests

```bash
.venv/Scripts/python -m pytest        # or bare `pytest`
```

**168 tests, ~2 seconds.** No API key needed and no network calls — the OpenAI client is faked, and
the fixture makes constructing a real one a test failure.

| File | Tests | Covers |
|---|---|---|
| [`test_openai_paths.py`](tests/test_openai_paths.py) | 65 | moderation, every degradation path, request shapes, `_encode_image` |
| [`test_assessments.py`](tests/test_assessments.py) | 40 | lighting, composition, blueprint — thresholds at their boundaries |
| [`test_api.py`](tests/test_api.py) | 27 | endpoint guards: size cap, rate limit, error mapping, CORS |
| [`test_placement.py`](tests/test_placement.py) | 22 | every directional claim in `_compute_placement` |
| [`test_features.py`](tests/test_features.py) | 14 | `extract_features` on synthetic scenes, all three blur regimes |

CI runs the suite on every PR to `main` ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).
Test-only dependencies live in `requirements-dev.txt` so Render's build stays lean.

### Conventions worth keeping

**Mutation-test new tests.** A green suite proves nothing about whether it *can* fail. Break the
thing on purpose, confirm a test catches it, revert. This caught three pieces of unreachable or
redundant code, and one test guard that had silently stopped working. **Commit before you start** —
the revert step is `git checkout --`, which will happily delete uncommitted work.

**The response contract is additive.** Never rename, remove or retype an existing field without
bumping the version in [`CONTRACT.md`](CONTRACT.md) and telling the team.

**Assert real values, not tautologies.** `assert x in (a, b)` where those are the only two possible
values tests nothing. Two such assertions shipped here before mutation testing found them.

---

## Deployment

Both sides auto-deploy from `main`.

| | Config | Notes |
|---|---|---|
| Backend | [`render.yaml`](render.yaml) | Build `pip install -r requirements.txt`, start `uvicorn api_server:app`. Health check `/health`. |
| Frontend | Netlify dashboard | No config in-repo; publish directory is `web/`. |

### Environment variables

Set in the Render dashboard (Environment tab):

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | **yes** | — | Never committed (`sync: false` in `render.yaml`). |
| `ALLOWED_ORIGINS` | no | `https://daka-ai.netlify.app` | Comma-separated. **Replaces** the default rather than adding to it, so keep the production origin in the list. |

Dependencies are pinned exactly (`==`) in `requirements.txt`, and Python is pinned in
`.python-version` — which CI reads too, so CI and production can't drift apart. To upgrade
something, bump one line and run the tests; don't bulk-refresh.

---

## Security posture

`/analyze` is **unauthenticated by design** — this is a school project and adding auth was
considered and declined. What protects it is cost control, not access control:

- **CORS** pinned to the Netlify origin. Browser-enforced only; does nothing against a direct `curl`.
- **Rate limit** 20 requests / 60s per IP. In-memory, so it resets on every cold start, and it keys
  on a spoofable `X-Forwarded-For`.
- **Size cap** 8 MB, enforced while reading in chunks so a hostile body can't be buffered first.
- **Error bodies** are fixed user-safe strings; exception detail is logged, never returned.

Each `/analyze` costs two OpenAI calls (moderation + vision), so the endpoint spends money. If the
URL ever leaks widely, the rate limit is a speed bump, not a wall.

**Moderation fails open**: if the moderation call itself errors, the image is treated as safe. That
was judged better than refusing every scan during an outage, but it is a bypass. It's logged.

---

## Known issues

**The two-dot framing overlay is unreliable.** It reads yaw from `deviceorientation.alpha`, but a
phone held upright with the rear camera on the horizon sits at `beta ≈ 90°` — exactly the gimbal-lock
singularity of the W3C `Z-X'-Y''` angle sequence, where `alpha` and `gamma` become degenerate.
`alpha` swings wildly while the phone barely moves, so the dots jump and won't settle. This is not a
tuning problem and `AIM_YAW_SIGN` won't fix it. The standing marker and the level slider are
unaffected and work fine. Options: drop the lock and keep the marker; use gravity for pitch/roll
only (no yaw); or integrate `devicemotion.rotationRate` for true short-horizon yaw.

**Extreme blur escapes the blur gate.** Past a point every edge smears below Canny's threshold, edge
density hits zero, and the "too plain to judge" escape hatch passes the frame — a plain wall and a
destroyed image are indistinguishable by edge density alone. Probably narrower on real broadband
scenes than on synthetic tests. Characterised in `test_features.py`.

**Two bits of dead-but-harmless code**, both found by mutation testing and documented in the tests:
the `y` clamp in `_compute_placement` is unreachable (`y` is only ever `0.62`, `0.667` or `0.70`),
and the `if not content` guard in `_analyze_with_gpt` is redundant with the parse guard below it.

**Generated but never displayed:** `lighting.tip`, `composition` and `blueprint.notes`. All free
(pure OpenCV, no tokens). `composition.horizon` and `blueprint.notes` speak directly to framing, so
they're the most natural things to surface next.

`pose_tips` was removed in contract `0.10` — see [`CONTRACT.md`](CONTRACT.md) for why. The prompt is
recoverable from `git show 3878ccb` if it's ever wanted back.
