# `/analyze` Response Contract

**Owner:** AI Engine (@Zuil909) · **Consumers:** Capture, Share · **Version:** `1.0` (draft)

This document freezes the JSON that `POST /analyze` returns, so the **Capture** teammate
(on-screen guidance / directional arrows) and the **Share** teammate (filter, hashtags,
caption) can build against a stable interface while the engine is implemented.

> **Golden rule — every change is additive.** The engine never renames, removes, or
> changes the type of an existing field. New fields default to `[]`, `""`, or `null`, so
> an older client that ignores them keeps working. If you need a breaking change, bump
> `contract_version` and tell the team first.

---

## 1. The one coordinate convention (read this first)

Every position in this contract uses **one** convention. State it once, honour it everywhere —
this is the single most common integration bug.

```
normalized [0, 1]   origin = TOP-LEFT of the frame   x → right   y → DOWN
```

- To draw on screen: `px = x * videoClientWidth`, `py = y * videoClientHeight`.
- This matches the existing CSS grid in `web/index.html` (`top: 33.33% / 66.66%`,
  `left: 33.33% / 66.66%`), so the 4 rule-of-thirds intersections are
  `x ∈ {0.333, 0.667} × y ∈ {0.333, 0.667}`.
- Coordinates are **resolution-independent**: the scan frame (~1024px) and the keeper
  frame (~1440px) share the same field of view, so the same normalized point lands
  correctly on both.
- Angles (tilt) are **degrees, signed**: positive = the scene's horizon is rolled
  **clockwise** (so the user rotates counter-clockwise to correct).

`"coord_space": "normalized_topleft"` is included in every response as a self-documenting marker.

---

## 2. Full example response

Existing fields (already shipped) are abbreviated; **new fields are shown in full.**

```jsonc
{
  "contract_version": "1.0",
  "coord_space": "normalized_topleft",

  // ── EXISTING (unchanged) ──────────────────────────────────────────────
  "scene_type": "Café",
  "blueprint":   { "orientation": "portrait", "grid": "rule_of_thirds",
                   "notes": ["strong rule-of-thirds alignment"] },
  "lighting":    { "quality": "Good", "tone": "Warm",
                   "tip": "Good natural light, shoot facing forward" },
  "composition": { "focus": "Sharp", "horizon": "Slightly tilted", "balance": "Balanced" },
  "pose_tips":   ["Lean against the window frame on the left",
                  "Angle your shoulders toward the soft window light",
                  "Hold the coffee cup low, hands relaxed"],
  "hashtags":    ["#cafevibes", "#coffeetime", "#打卡"],
  "filter":      "Vivid Warm",

  // ── NEW · raw detection (AI Engine) ───────────────────────────────────
  "objects": [
    { "label": "window", "box": { "x": 0.05, "y": 0.10, "w": 0.30, "h": 0.55 }, "confidence": 0.88 },
    { "label": "table",  "box": { "x": 0.40, "y": 0.62, "w": 0.45, "h": 0.30 }, "confidence": 0.81 }
  ],

  // ── NEW · authoritative framing guidance → Capture ────────────────────
  "framing": {
    "subject": {
      "detected":   true,
      "source":     "gpt",                                  // "gpt" | "saliency"
      "label":      "person",                               // "salient_region" when saliency-only
      "confidence": 0.82,
      "center":     { "x": 0.52, "y": 0.61 },
      "bbox":       { "x": 0.34, "y": 0.30, "w": 0.30, "h": 0.55 },  // may be null
      "size":       0.70                                    // suggested subject height (fraction of frame)
    },
    "target":   { "intersection": "bottom-left", "x": 0.333, "y": 0.667 },
    "guidance": {
      "move_subject_x": "left",     // "left" | "right" | "ok"
      "move_subject_y": "up",       // "up"   | "down"  | "ok"
      "distance":       "closer",   // "closer" | "farther" | "ok"
      "dx": -0.187, "dy": -0.057,   // signed offset (target − subject); use for arrow length
      "strength": 0.31              // 0..1 magnitude of (dx,dy); arrow size / when to snap to "ok"
    },
    "level": {
      "scene_horizon_tilt_deg": 3.4,   // tilt of the SCENE's horizon (static); + = clockwise
      "needs_straightening":    true,  // == alignment < 0.7, precomputed
      "source_alignment":       0.71   // raw features.alignment, passed through
    },
    "reason": "Stand at the left third by the window so soft light hits your face"
  },

  // ── NEW · generate branch → Capture (directives) + Share (caption) ────
  "framing_suggestions": [
    { "target": "subject", "instruction": "Stand in the lower-left third under the hanging plant", "anchor": "left_third" },
    { "target": "camera",  "instruction": "Step back two paces to catch the full doorway arch",    "anchor": "step_back" },
    { "target": "camera",  "instruction": "Tilt down slightly so the tabletop leads into frame",    "anchor": "tilt_down" }
  ],
  "scene_yap": "Cosy corner, golden light — this spot was made for a 打卡. ☕"
}
```

---

## 3. Field reference

### 3.1 `objects[]` — raw detected things *(advisory)*
What the engine sees in the (empty) scene. Use it to avoid placing the subject on top of
furniture, or to label what framing should keep/avoid. **Advisory only** — do not build core
logic on it (boxes can drift; the engine may occasionally hallucinate). Drive placement from
`framing` instead.

| Field | Type | Notes |
|---|---|---|
| `label` | string | free-text, e.g. `"window"`, `"table"` |
| `box` | `{x,y,w,h}` | normalized top-left xywh |
| `confidence` | number | `0..1` |

Guarantees: array, length **≤ 8**, may be `[]`. Low-confidence / malformed entries are dropped by the engine.

### 3.2 `framing` — the authoritative guidance object *(Capture builds arrows from this)*
The engine **bakes the math** so the frontend stays dumb — you read enums, you don't compute geometry.

**`framing.subject`** — the subject and where it currently is.
| Field | Type | Notes |
|---|---|---|
| `detected` | bool | **`false` ⇒ hide arrows, fall back to the static grid prompt** |
| `source` | enum | `"gpt"` (real detection) or `"saliency"` (OpenCV fallback) |
| `label` | string | `"person"`, or `"salient_region"` when saliency-only |
| `confidence` | number | `0..1` — gate jittery arrows on this if you like |
| `center` | `{x,y}` | subject centroid |
| `bbox` | `{x,y,w,h}` \| null | optional; null when saliency-only |
| `size` | number | suggested subject height as a fraction of frame height |

**`framing.target`** — where the subject *should* go (nearest strong rule-of-thirds point).
| Field | Type | Notes |
|---|---|---|
| `intersection` | enum | `top-left` \| `top-right` \| `bottom-left` \| `bottom-right` \| `center` |
| `x`, `y` | number | the point the arrow aims at |

**`framing.guidance`** — precomputed directions (deadband ±0.04).
| Field | Type | Notes |
|---|---|---|
| `move_subject_x` | enum | `left` \| `right` \| `ok` |
| `move_subject_y` | enum | `up` \| `down` \| `ok` |
| `distance` | enum | `closer` \| `farther` \| `ok` |
| `dx`, `dy` | number | signed `target − subject`; arrow vector |
| `strength` | number | `0..1`; magnitude of the offset |

> ⚠️ **TEAM DECISION (default chosen): `move_subject_*` moves the SUBJECT in-frame.**
> "Move the camera" is the **opposite** direction. If Capture prefers camera-relative arrows,
> tell the engine and it will emit a parallel `move_camera_x/y` instead of flipping meanings.

**`framing.level`** — static scene tilt (NOT live device roll).
| Field | Type | Notes |
|---|---|---|
| `scene_horizon_tilt_deg` | number | signed; tilt of the captured scene's horizon |
| `needs_straightening` | bool | `alignment < 0.7` |
| `source_alignment` | number | raw `features.alignment` |

> The **live** camera-roll bubble stays frontend-owned (the existing `deviceorientation`
> gamma listener in `index.html`). `framing.level` only describes the analysed *scene*. Don't
> double-count them in one indicator.

**`framing.reason`** — one short human string explaining the placement (good for a tip line).

### 3.3 `framing_suggestions[]` — semantic placement directives *(generate branch)*
Exactly **3** ordered (most-impactful-first) directives. Distinct from `pose_tips`
(body language) — these are about *placement* of subject/camera.

| Field | Type | Notes |
|---|---|---|
| `target` | enum | `"subject"` or `"camera"` |
| `instruction` | string | imperative, ≤ ~12 words; safe to show on screen verbatim |
| `anchor` | enum (closed set) | maps 1:1 to a UI affordance — see below |

**Closed `anchor` set** (switch on these; fall back to showing `instruction` text if unknown):
```
left_third  right_third  center  upper_third  lower_third          ← subject grid cells
tilt_up  tilt_down  pan_left  pan_right                            ← camera rotation
step_back  step_closer  raise_camera  lower_camera  level_horizon  ← camera position / level
```

### 3.4 `scene_yap` — shareable one-liner *(Share branch)*
One fun, on-brand sentence (≤ ~90 chars) for the quick-share caption, next to the
hashtag pills and the "Shot with 打卡AI" watermark.

> ⚠️ **TEAM DECISION (default chosen): English voice, the token `打卡` allowed inline,
> ≤ 1 emoji, no hashtags inside.** All other fields stay strictly English. Confirm whether
> Share strips the emoji before baking it into the watermark image.

---

## 4. Consumer cheat-sheet — who reads what

| Field | Capture | Share | Current UI |
|---|:---:|:---:|:---:|
| `scene_type` | — | — | ✅ badge |
| `lighting`, `composition`, `blueprint` | ✅ copy | — | partial |
| `pose_tips` | ✅ pose panel | — | ✅ swiper |
| `objects` | ✅ avoid-overlap | — | — |
| **`framing`** | ✅ **arrows + marker + level** | — | — |
| **`framing_suggestions`** | ✅ **directive arrows / zones** | — | — |
| `filter` | — | ✅ auto-filter | ✅ preview |
| `hashtags` | — | ✅ tags | ✅ pills |
| **`scene_yap`** | — | ✅ **caption** | — |

**Capture's must-honour guarantees**
1. All coords are normalized, top-left origin, `[0,1]`.
2. `guidance` / `anchor` values come only from the closed enum sets above.
3. When `framing.subject.detected == false`, **degrade to the static grid prompt** — do not draw arrows.
4. Live device roll = your `deviceorientation` listener; `framing.level` = static scene tilt. Keep them separate.

**Share's must-honour guarantees**
1. Read `scene_yap`, `hashtags`, `filter` only; ignore `framing*`.
2. Treat all as optional — `scene_yap` may be `""` on older responses.

---

## 5. Reliability notes (how the engine fills these)

- **`framing.subject`** comes from the GPT vision call first; if GPT omits a placement, the engine
  falls back to an OpenCV saliency centroid / strongest rule-of-thirds intersection. `source`
  tells you which. **`framing.target.{x,y}` and a usable subject point are *always* present.**
- **`objects`** are advisory and may be empty.
- **`framing.level`** is derived from the existing OpenCV alignment metric — reliable.
- The engine validates & clamps everything server-side; consumers should still defensively
  default missing fields rather than assume.

---

## 6. Explicitly out of scope for v1

- **Lens awareness.** The 0.5× ultra-wide lens is a client-side capture concern; `/analyze`
  is **not** lens-aware in v1 (no `lens` request field, no `capture` block). Revisit only if
  ultra-wide distortion is found to skew framing advice.
- **Real-time per-frame tracking.** `/analyze` is one-shot per scan. Any live person-tracking
  (e.g. in-browser MediaPipe) is optional Capture-owned client polish, layered *on top* of the
  `framing.target` this contract provides — not an engine dependency.

---

## 7. Open questions to close before freezing `1.0`

1. **Arrow semantics** — confirm `move_subject_*` (subject-relative) vs. add `move_camera_*` (§3.2).
2. **`scene_yap` language/emoji policy** — confirm the default in §3.4.
3. **Object filtering** — agree a confidence floor / max count so the subject is never placed on phantom furniture.

Once these are signed off, change the version line to `1.0` (frozen) and treat any later change as additive.
