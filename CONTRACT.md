# `/analyze` Response Contract

**Owner:** AI Engine (@Zuil909) · **Consumers:** `web/index.html` · **Version:** `0.9` (shipped)

This document describes the JSON that `POST /analyze` **actually returns today**, as implemented in
[`scene_analysis.py`](scene_analysis.py) and served by [`api_server.py`](api_server.py). Section 6
records the larger `framing` design that was drafted but is **not implemented** — it is kept as a
proposal, not a promise.

> **History.** Earlier revisions of this file specified a `1.0` shape (`framing`, `objects`,
> `framing_suggestions`, `scene_yap`, `contract_version`, `coord_space`) that the engine never
> emitted, while omitting fields it does emit (`placement`, `blurry`, `blur_var`,
> `edge_sharpness`). It has been rewritten to match reality. Version is `0.9` to make clear that
> the `1.0` name is still unclaimed.

> **Golden rule — every change is additive.** Never rename, remove, or change the type of an
> existing field. New fields default to `[]`, `""`, or `null` so an older client keeps working.
> A breaking change needs a version bump and a heads-up to the team.

---

## 1. The one coordinate convention (read this first)

Every position in this contract uses **one** convention:

```
normalized [0, 1]   origin = TOP-LEFT of the frame   x → right   y → DOWN
```

- To draw on screen: `px = x * videoClientWidth`, `py = y * videoClientHeight`.
- This matches the CSS grid in [`web/index.html`](web/index.html) (`top: 33.33% / 66.66%`,
  `left: 33.33% / 66.66%`), so the four rule-of-thirds intersections are
  `x ∈ {0.333, 0.667} × y ∈ {0.333, 0.667}`.
- Coordinates are **resolution-independent**: the scan frame (capped at 1024px) and the keeper
  frame (capped at 2560px) share the same field of view, so the same normalized point lands
  correctly on both.

Note that the response does **not** currently carry a `coord_space` marker — see §5.

---

## 2. Full example response

Every field below is always present on a `200`. There are no optional keys.

```jsonc
{
  "scene_type":     "Café",

  "blueprint":      { "orientation": "landscape",
                      "grid": "rule_of_thirds",
                      "notes": ["strong rule-of-thirds alignment"] },

  "lighting":       { "quality": "Good",
                      "tone": "Warm",
                      "tip": "Good natural light, shoot facing forward" },

  "composition":    { "focus": "Sharp",
                      "horizon": "Slightly tilted",
                      "balance": "Balanced" },

  "blurry":         false,
  "blur_var":       184.3,      // diagnostic — for tuning the blur gate
  "edge_sharpness": 21.47,      // diagnostic — for tuning the blur gate

  "placement":      { "x": 0.667, "y": 0.667 },

  "pose_tips":      ["Lean against the window frame on the left",
                     "Angle your shoulders toward the soft window light",
                     "Hold the coffee cup low, hands relaxed"],

  "hashtags":       ["#cafevibes", "#coffeetime", "#goldenhour"],

  "filter":         "Vivid Warm"
}
```

---

## 3. Field reference

### 3.1 `placement` — where the subject should stand *(drives the standing marker)*

| Field | Type | Notes |
|---|---|---|
| `x` | number | **snapped to a rule-of-thirds line: `0.333` or `0.667`** |
| `y` | number | clamped to `[0.60, 0.72]` — adapts to where saliency mass sits vertically |

Computed by [`_compute_placement`](scene_analysis.py#L39-L113), which fuses three gated signals —
visual **balance** (stand opposite the scene's focal mass), background **cleanliness** (prefer the
side whose body-band is emptier), and **light direction** (stand on the dimmer side so light falls
on the face) — plus a hard **backlight veto** so the subject is never placed in front of a
blown-out region. Never raises; falls back to `{0.667, 0.667}`.

> ⚠️ **This is a composition target, not a point to aim the camera at.** `x` is already snapped to
> a thirds line for the framing that was scanned. Panning the camera until this point reaches
> screen-centre would drag the subject to dead-centre and discard the placement the engine solved
> for. Draw it as a fixed in-frame marker. (This exact confusion was a live bug; see the
> `standPos` / `aim` comment block in `index.html`.)

### 3.2 `lighting` — *(gates capture)*

| Field | Type | Values |
|---|---|---|
| `quality` | enum | `Good` (brightness 100–200) · `Fair` (60–100 or 200–230) · `Poor` (otherwise) |
| `tone` | enum | `Warm` (`color_ratio` < 0.7) · `Cool` (> 0.9) · `Neutral` (between) |
| `tip` | string | one of nine fixed strings, keyed by `(quality, tone)` |

**`quality == "Poor"` blocks the flow** — the client shows the capture gate and forces a rescan.

`color_ratio` is an internal feature, not part of this response. It is
`avg_color[0] / avg_color[2]` over an image `cv2.imread` loads as **BGR**, so it is
**blue ÷ red** — a *higher* ratio means *more blue*, i.e. a **cooler** scene. The neutral band sits
near 0.8 rather than 1.0 because most scenes carry a mild red bias.

> **Behaviour change (`tone` inversion fixed).** `assess_lighting` previously mapped the *high*
> (blue-dominant) end of `color_ratio` to `"Warm"`, so `tone` — and therefore the `tip` string —
> came out backwards: a golden-hour café was told "Nice cool tones, use them for a clean
> aesthetic". The two comparisons have been swapped; thresholds are unchanged, so the neutral band
> stays where it was calibrated and `quality` is unaffected. The nine `tip` strings were already
> written for the correct semantics and now route correctly. `filter` was never affected — the
> model picks that independently. **Any `tone` value recorded before this fix is inverted.**

### 3.3 `blurry`, `blur_var`, `edge_sharpness` — *(gates capture)*

| Field | Type | Notes |
|---|---|---|
| `blurry` | bool | **`true` blocks the flow** and forces a rescan |
| `blur_var` | number | variance of Laplacian, 1 d.p. — diagnostic only |
| `edge_sharpness` | number | mean \|Laplacian\| at Canny edges, 2 d.p.; `-1.0` means "too plain to judge" |

The gate is deliberately **content-robust**: variance-of-Laplacian alone flags any low-texture
scene (plain wall, minimalist café) as blurry, which blocked the whole flow. Instead the engine
judges the sharpness of the edges that actually exist, and gives a near-featureless frame the
benefit of the doubt. Thresholds are lenient — over-rejecting is the worse failure — and tunable
against `blur_var` / `edge_sharpness` from real photos. See
[`scene_analysis.py:29-37`](scene_analysis.py#L29-L37).

### 3.4 `composition` — descriptive assessment

| Field | Type | Values |
|---|---|---|
| `focus` | enum | `Sharp` · `Soft` · `Blurry` (from Canny edge density) |
| `horizon` | enum | `Level` · `Slightly tilted` · `Tilted` (from Hough-line deviation) |
| `balance` | enum | `Balanced` · `Slightly off` · `Unbalanced` (from left/right saliency split) |

Derived from measured features, not from the model — reliable. Currently **unused by the UI**.

### 3.5 `blueprint` — orientation + advisory notes

| Field | Type | Notes |
|---|---|---|
| `orientation` | enum | `portrait` (h ≥ w) · `landscape` |
| `grid` | const | always `"rule_of_thirds"` |
| `notes` | string[] | 0–3 of: tilted horizon · unbalanced composition · strong rule-of-thirds alignment |

Currently **unused by the UI**.

### 3.6 `scene_type`, `pose_tips`, `hashtags`, `filter` — the model's output

All four come from a single vision call in
[`_analyze_with_gpt`](scene_analysis.py#L128-L163), and all four **degrade rather than fail**: a
truncated, empty, refused, or unparseable completion yields `{}`, and each field falls back to its
default rather than 500ing the scan.

| Field | Type | Fallback | Notes |
|---|---|---|---|
| `scene_type` | string | `"Unknown"` | concise name, e.g. `"Café"`, `"City Street"`, `"Temple"` |
| `pose_tips` | string[] | `[]` | asked for exactly 3, scene-specific. **Generated but unused by the UI** |
| `hashtags` | string[] | `[]` | asked for exactly 3, lowercase, with `#` |
| `filter` | enum | `"Vivid"` | **server-validated** against the list below; anything else becomes `"Vivid"` |

Closed `filter` set — the client maps these 1:1 to CSS filter strings:

```
Vivid  Vivid Warm  Vivid Cool
Dramatic  Dramatic Warm  Dramatic Cool
Silvertone  Noir
```

Counts are *requested*, not enforced — the engine passes the arrays through unchanged, so treat
lengths defensively.

---

## 4. Errors

| Status | Body | Cause |
|---|---|---|
| `400` | `{"error": "Image not suitable for analysis"}` | flagged by `omni-moderation-latest` |
| `500` | `{"error": "<message>", "type": "<ExceptionName>"}` | anything else |

Two things worth knowing:

- **Moderation fails open.** If the moderation call itself errors,
  [`_moderate_image`](scene_analysis.py#L116-L125) returns "safe". Deliberate, but it is a bypass.
- **`500` leaks internals.** The raw exception string and class name are returned to the client
  ([`api_server.py:34`](api_server.py#L34)). Should be replaced with an opaque message plus a
  server-side log.

The client treats a missing `lighting` key as a bad response regardless of status, so `lighting`
is load-bearing for validity detection.

---

## 5. Consumer cheat-sheet — who reads what

| Field | Consumed by `index.html` | How |
|---|:---:|---|
| `lighting.quality` | ✅ | capture gate — `Poor` blocks |
| `blurry` | ✅ | capture gate — `true` blocks |
| `scene_type` | ✅ | badge on camera + results |
| `placement` | ✅ | fixed standing marker |
| `filter` | ✅ | preview + baked into the saved pixels |
| `hashtags` | ✅ | tappable pills, copy-all, share text |
| `lighting` (presence) | ✅ | response-validity check |
| `lighting.tip` | — | generated, not shown |
| `pose_tips` | — | generated, not shown |
| `composition` | — | generated, not shown |
| `blueprint` | — | generated, not shown |
| `blur_var`, `edge_sharpness` | — | diagnostics, for tuning only |

**Client must-honour guarantees**

1. All coords are normalized, top-left origin, `[0,1]`.
2. `filter` comes only from the closed set in §3.6; unknown values must fall back, not throw.
3. `placement` is an in-frame composition target — render it fixed, never chase it with the camera.
4. Live device roll (the level slider) is entirely client-owned, read from `devicemotion`. The
   engine reports **scene** tilt via `composition.horizon`. Don't merge the two into one indicator.

**Known gaps** (cheap, additive, worth doing)

- No `contract_version` field — clients cannot tell which engine build answered.
- No `coord_space: "normalized_topleft"` self-documenting marker.
- `pose_tips` and `lighting.tip` are paid for on every scan and thrown away.

---

## 6. Proposed, NOT implemented

Everything in this section is design work, not API surface. **Do not build against it.** It is
retained because the geometry is worked out and most of it is derivable from features the engine
already computes.

### 6.1 `framing` — bake the math server-side

The idea: the engine precomputes guidance so the client reads enums instead of doing geometry.

```jsonc
"framing": {
  "subject":  { "detected": true, "source": "saliency", "label": "salient_region",
                "confidence": 0.82, "center": {"x":0.52,"y":0.61},
                "bbox": null, "size": 0.70 },
  "target":   { "intersection": "bottom-left", "x": 0.333, "y": 0.667 },
  "guidance": { "move_subject_x": "left", "move_subject_y": "up", "distance": "closer",
                "dx": -0.187, "dy": -0.057, "strength": 0.31 },
  "level":    { "scene_horizon_tilt_deg": 3.4, "needs_straightening": true,
                "source_alignment": 0.71 },
  "reason":   "Stand at the left third by the window so soft light hits your face"
}
```

Feasibility from today's code:

| Sub-object | Status |
|---|---|
| `level` | **Easy** — `features["alignment"]` already exists; `needs_straightening` is `alignment < 0.7` |
| `target` | **Easy** — `placement` already is the nearest strong thirds point |
| `subject` | **Needs live tracking.** `/analyze` is one-shot on an *empty* scene, so there is no subject to detect. This only becomes meaningful with in-browser per-frame tracking |
| `guidance` | Depends on `subject` — it is `target − subject`, so it needs the above first |

> If `guidance` is ever built, settle the sign convention **first**: `move_subject_*` moves the
> subject in-frame; moving the *camera* is the opposite direction. Pick one and name it explicitly.

### 6.2 `objects[]` — raw detections

`[{ "label": "window", "box": {x,y,w,h}, "confidence": 0.88 }]`, ≤ 8 entries, advisory only. Would
let the client avoid placing the subject on top of furniture. Requires adding object detection to
the vision prompt and validating/clamping the boxes server-side.

### 6.3 `framing_suggestions[]` — semantic placement directives

Up to 3 ordered `{target, instruction, anchor}` directives, distinct from `pose_tips` (which are
body language, not placement). The `anchor` was to be a closed set mapping 1:1 to UI affordances:

```
left_third  right_third  center  upper_third  lower_third      ← subject grid cells
tilt_up  tilt_down  pan_left  pan_right                        ← camera rotation
step_back  step_closer  raise_camera  lower_camera  level_horizon  ← camera position / level
```

### 6.4 `scene_yap` — shareable one-liner

One on-brand sentence (≤ ~90 chars) for the share caption, alongside the hashtag pills and the
"Shot with 打卡AI" watermark. Open question: English voice with `打卡` allowed inline, ≤ 1 emoji, no
hashtags inside — and whether the emoji survives into the watermark.

### 6.5 Out of scope

- **Lens awareness.** The 0.5× ultra-wide toggle is a client-side capture concern; `/analyze` is
  not lens-aware and has no `lens` request field. Revisit only if ultra-wide distortion is found
  to skew placement advice.
- **Real-time tracking in the engine.** `/analyze` is one-shot per scan. Any live tracking is
  client-owned polish layered on top of `placement`, not an engine dependency.
