# 打卡吧！(Dǎkǎ ba!)

**DaKaBa tells you exactly where your friend should stand for a good photo, and why.**

Point your phone at a scene and tap once. The app works out the best spot for a person to stand,
draws a marker there, and explains its choice: "light falls on your face", "cleaner background
here", "out of the window glare". Then it talks you through the shot, picks a filter, and tags it.

<img width="1169" height="1986" alt="Example image" src="https://github.com/user-attachments/assets/142abfee-44a2-4ae1-9b50-6c0767405d30" />

| | |
|---|---|
| Try it | <https://dakaba.pages.dev> |
| iOS app | Capacitor shell in [`ios/`](ios), built from the same page |
| Backend | <https://daka-backend-9bfz.onrender.com> |

Open it on a phone. It needs a rear camera and motion sensors, and it works best with two people,
one holding the phone and one standing in the scene.

The idea behind it: subject positioning, framing and colour grading are what make a photo. Posing is
left to the person being photographed.

## The flow

```
Home  ->  camera  ->  point at an empty scene, tap the shutter
                          |
         POST /measure  (frame capped at 1024px, JPEG q0.7)  ~1s
         POST /describe  fired at the same time, answers later
                          |
          capture gate: reject if the light is Poor or the frame is blurry
                          |
     standing marker drawn at the returned point, with a depth sentence above
     the shutter ("Stand in front of the blue door")
                          |
       MediaPipe pose tracks the subject's ankles against it, cueing them
       left / right / closer / back, then the camera's own tilt and horizon
                          |
         marker turns green once they are on the spot  ->  take the photo
                          |
          results: filter strip, hashtag pills, share, save
```

## How it works

| Piece | File | Notes |
|---|---|---|
| Frontend (all of it) | [`web/index.html`](web/index.html) | HTML, CSS and JS inline. No build step. |
| API | [`api_server.py`](api_server.py) | FastAPI. `/health`, and `/measure`, `/describe`, `/analyze`. |
| Engine | [`scene_analysis.py`](scene_analysis.py) | OpenCV measurements plus one vision-model call. |
| Response contract | [`CONTRACT.md`](CONTRACT.md) | Read before changing the response shape. |
| iOS shell | [`capacitor.config.json`](capacitor.config.json), [`ios/`](ios) | Capacitor 8. Adds the camera roll and the tip jar. |

### The engine is split in two

**OpenCV does the measurable half.** Brightness, colour balance, Canny edge density, Laplacian
blur, spectral-residual saliency, Hough-line horizon tilt, and the placement decision.

**The vision model does the subjective half.** Scene name, filter choice, hashtags, and the depth
sentence that refines the marker. Four fields, and the positioning does not depend on any of them.

So a failed vision call still returns a usable scan, with `scene_type: "Unknown"`. One consequence
for anyone reading the response: **a `200` is not proof the model ran.** See
[`CONTRACT.md`](CONTRACT.md) §3.6.

The two halves are also two routes. `/measure` is pixels only and answers in about a second;
`/describe` carries both OpenAI calls and took between 3.7 and 11.1 seconds for the same image. The
client fires them together and draws the marker on the first, so nobody stands with their arm up
waiting for a scene name. Fields that have not arrived yet look exactly like fields the model
failed to produce, which the app already handled: the hint hides when empty, the hashtag card is
not rendered without tags, and the filter defaults to Vivid.

`_compute_placement` is the core. It fuses visual balance (stand opposite the scene's focal mass),
background cleanliness (prefer the emptier side) and light direction (stand on the dimmer side, so
the light falls on your face), each voting only when it is reliable for that scene. Over the top
sits a backlight veto, so nobody is ever placed in front of a blown-out window.

### Where that second goes

Measured against the live deployment, posting a 768x1024 JPEG at q0.7, which is the size the client
sends:

| | |
|---|---|
| Round trip to Render for a route that does nothing (`/health`) | ~250ms |
| `measure_scene` on a laptop | 16ms |
| the same call on Render's free plan | ~700ms |

The free plan is **0.1 CPU**, a tenth of a core. So almost the whole wait is 16ms of work being
executed slowly, and neither the network nor the algorithm is the thing to optimise. A plan with
0.5 CPU is the only change that shortens it.

Two costs on top of that: the first request of a session adds roughly 370ms (measured 1.34s, then
0.97s and 0.98s), and a request that has to wake a sleeping service takes about 50 seconds.

### The tip jar

RevenueCat, native builds only, and nothing is gated. No entitlement is read anywhere, and
[`tests/test_client_loop.py`](tests/test_client_loop.py) enforces that by scanning the page for the
words that would prove otherwise. The button appears only after a photo, and only once RevenueCat
answers with a package that can actually be bought.

## Running it locally

Requires **Python 3.13**, pinned in [`.python-version`](.python-version).

```bash
python -m venv .venv

# Windows
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

.venv/Scripts/python -m uvicorn api_server:app --reload
```

Serves on <http://127.0.0.1:8000>. It starts without an API key, so `/health` works immediately. You
only need a key to scan:

```bash
export OPENAI_API_KEY=sk-...        # bash
$env:OPENAI_API_KEY = "sk-..."      # PowerShell
```

`.env`, `.env.*`, `*.key` and `daka_openai_key.txt` are gitignored.

### Frontend

One static file, no build step. Serve it over HTTP rather than opening it with `file://`, which
sends `Origin: null` and can never match CORS:

```bash
cd web && python -m http.server 8080
```

To point it at your local backend, two things have to change:

1. The API URL is hardcoded in the `API` constant at the top of the script in
   [`web/index.html`](web/index.html). Change it to `http://localhost:8000`.
2. Set `ALLOWED_ORIGINS=http://localhost:8080` in the shell running your local backend.

Skip the second and every scan fails as a generic network error, because browsers do not report
CORS failures usefully.

### iOS

```bash
npm install
npm run sync            # cap sync ios
npm run open            # opens Xcode
```

Capacitor 8 uses Swift Package Manager, so there is no workspace and no `pod install`. The camera
roll write lives in [`ios/App/App/SavePhotoPlugin.swift`](ios/App/App/SavePhotoPlugin.swift).

### Testing on a phone

Camera and motion sensors need HTTPS on a real device. Push a branch and use the Cloudflare Pages
preview, but note that preview URLs are a different origin and are CORS-blocked, so a preview can
show the UI without scanning.

## Tests

```bash
.venv/Scripts/python -m pytest
ruff check .
```

**429 tests, under 10 seconds.** No API key and no network: the OpenAI client is faked, and
constructing a real one is a test failure.

The four `test_client_*` files run the page's own JavaScript in node against a stubbed DOM. They
exist because two bugs reached a phone that `node --check` could not see. Both were valid syntax,
and both were out-of-scope identifiers that only failed when the code actually ran.

| File | Tests | Covers |
|---|---|---|
| [`test_openai_paths.py`](tests/test_openai_paths.py) | 89 | Moderation, every degradation path, request shapes, the `lang` prompt |
| [`test_client_loop.py`](tests/test_client_loop.py) | 96 | The render loop, marker, cue ladder, camera lifecycle, horizon maths, tap to focus, tip jar |
| [`test_guidance.py`](tests/test_guidance.py) | 42 | Placement reason, dead-space tilt, depth hint, client/engine filter agreement |
| [`test_assessments.py`](tests/test_assessments.py) | 40 | Lighting, composition, blueprint, at their thresholds |
| [`test_client_overlay.py`](tests/test_client_overlay.py) | 32 | Marker drawing, `visibleCrop`, camera-region layout |
| [`test_api.py`](tests/test_api.py) | 39 | Size cap, rate limit, error mapping, CORS, `lang` |
| [`test_client_i18n.py`](tests/test_client_i18n.py) | 26 | The language switch, both string tables, every key asked for |
| [`test_placement.py`](tests/test_placement.py) | 23 | Every directional claim in `_compute_placement`, including the veto |
| [`test_filters.py`](tests/test_filters.py) | 16 | Baked filters match the CSS preview exactly |
| [`test_features.py`](tests/test_features.py) | 14 | `extract_features` on synthetic scenes, all three blur regimes |
| [`test_client_theme.py`](tests/test_client_theme.py) | 12 | The light/dark choice, and what it does and does not repaint |

CI runs on every PR to `main` ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)), reading
the same `.python-version` Render does.

### Conventions

**Mutation-test new tests.** A green suite proves nothing about whether it can fail. Break the thing
on purpose, confirm a test catches it, revert. This found three pieces of dead code and one test
guard that had silently stopped working. Commit first: the revert step is `git checkout --`.

**The contract is additive.** Never rename, remove or retype a field without bumping the version in
[`CONTRACT.md`](CONTRACT.md).

**Assert real values.** `assert x in (a, b)` where those are the only options tests nothing. Two
such assertions shipped here before mutation testing found them.

## Deployment

Both sides auto-deploy from `main`.

| | Config | Notes |
|---|---|---|
| Backend | [`render.yaml`](render.yaml) | `uvicorn api_server:app`, health check `/health`. |
| Frontend | Cloudflare Pages dashboard | No build command, output directory `web/`. |

Set in the Render dashboard:

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | none | Never committed (`sync: false` in `render.yaml`). |
| `ALLOWED_ORIGINS` | no | `https://dakaba.pages.dev`, plus the two Capacitor origins | Comma-separated. Replaces the default rather than adding to it. |

The backend is on Render's free plan, so it sleeps after 15 minutes idle and the first request
after that takes about 50 seconds. A cron job pings `/health` every 5 minutes to keep it warm, which
leaves room for two missed pings. At 10 minutes, two in a row put a visitor in front of the 50
second wait.

Free instance hours are **750 a month across the whole workspace**, and keeping one service awake
around the clock for a 31 day month costs 744 of them. It fits, but only just, and only while this
is the only free service in the workspace.

Dependencies are pinned exactly in `requirements.txt`. To upgrade something, bump one line and run
the tests.

## Cost and abuse

`/analyze` is unauthenticated by design. This is a school project, and auth was considered and
declined. What protects it is cost control rather than access control:

- **CORS** pinned to the live origins. Browser-enforced only, and does nothing against `curl`.
- **Rate limit** of 20 requests per minute per IP, in memory, keyed on a spoofable header.
- **Size cap** of 8 MB, enforced while reading in chunks.
- **Error bodies** are fixed strings. Exception detail is logged, never returned.

Every scan costs two OpenAI calls, so the endpoint spends money. If the URL leaked widely, the rate
limit is a speed bump rather than a wall.

**Moderation fails open.** If the moderation call itself errors, the image is treated as safe. That
beats refusing every scan during an outage, but it is a bypass, and it is logged.

## Known issues

[`docs/gotchas.md`](docs/gotchas.md) covers the device behaviour we had to find the hard way: why
the filters use an SVG colour matrix, why detection runs on a 480px copy, why the compass drift
warning was deleted.

**Tap to focus does nothing on iOS.** `pointsOfInterest` is an image-capture extension Chromium
ships and WebKit does not, so the capability check fails and the tap is ignored, reticle included. It
works in Chrome. Deliberately silent rather than cosmetic: see [`docs/gotchas.md`](docs/gotchas.md).

**No "you have moved since scanning" warning.** The compass cannot measure it in the pose you hold a
phone to take a photo. Doing it properly needs the gyroscope integrated over the few seconds between
scanning and shooting.

**Extreme blur escapes the blur gate.** Past a point every edge smears below Canny's threshold, edge
density hits zero, and the "too plain to judge" escape hatch passes the frame. Characterised in
[`test_features.py`](tests/test_features.py).

**Generated but never shown:** `lighting.tip`, `composition` and `blueprint.notes`. All free to
compute. `composition.horizon` and `blueprint.notes` are the most natural things to surface next.
