# Things that bit us

Behaviour we had to discover by watching the app fail on a real phone, kept here rather than inline
so the code stays readable. Each of these cost at least an evening.

---

## `hue-rotate` cannot tint a neutral colour

Our "cool" filters shifted a grey wall by **exactly zero**. Grey has no hue, so there is nothing for
a hue rotation to rotate. And `sepia()` — the only CSS primitive that *will* tint a neutral — only
warms, and desaturates as it does, which is why `Vivid Warm` measured *less* vivid than plain
`Vivid`.

White balance is a channel gain, and CSS filter functions cannot express one. The grades use an SVG
`feColorMatrix` via `url(#warmTint)`, with a matching `tint()` in the pixel-baking path so the
preview and the saved file agree.

**Don't** replace the `url()` tokens with filter functions. It will look like a simplification and
will silently do nothing.

---

## OpenCV loads images as BGR, so our warmth score was inverted

`color_ratio` is `blue / red`, computed over an array OpenCV gives you in **B, G, R** order. A warm
scene is red-dominant and therefore scores *low*. The code had the high end labelled "Warm" and told
golden-hour cafés they had "nice cool tones".

The thresholds in `assess_lighting` are correct; only the labels were ever wrong. The neutral band
sits near 0.8 rather than 1.0 because most scenes carry a mild red bias.

---

## 4096px is the maximum GPU texture size on a great many phones

MediaPipe uploads whatever you hand `detectForVideo` as a GPU texture. When we raised the capture
track to 4096×3072, detection stopped entirely — not slowly, it **threw**, and our `catch` swallowed
it. There was no way to tell a model that had found nobody from one failing on every call.

Detection now runs on a 480px copy. Landmarks come back normalised and both axes scale by the same
factor, so the coordinates are identical to what the full frame produced.

---

## A `<video>` inside a `display:none` screen stays suspended

Returning to the viewfinder from the results screen, the stream is never detached — but the element
is paused, `videoWidth` is 0, and the render loop returns on its first line every frame. Autoplay
does not re-fire when the element is shown again. It has to be played explicitly.

Worse, if the page loses visibility while the camera screen is hidden, **iOS releases the camera
track outright**. `readyState` goes to `"ended"` and never recovers; the only fix is a second
`getUserMedia`.

---

## A layout measured while hidden comes back 0

`setCamState` ends in `resizeControls`, which measures the controls bar to animate its height. Call
it before `show('camera')` and the measurement is 0, so the bar is left at `auto` with no pinned
height — and the next state change has nothing to animate *from*. The symptom was the frame only
beginning to glide from the second scan onwards.

A pinned height must also be **released** afterwards. Left in place it combines with
`overflow: hidden` to clip anything that arrives later — and things do arrive later:
`detectLenses` awaits `enumerateDevices`, so the 1×/0.5× toggle lands long after the bar was sized.

---

## The compass is degenerate in the one pose you hold a phone to take a photo

We built a drift warning using `deviceorientation.alpha`. Held upright with the lens on the horizon,
a phone sits at β ≈ 90° — the gimbal-lock singularity of the W3C Z-X'-Y'' sequence, where yaw and
roll become indistinguishable. The reading swung wildly while the phone sat still.

This is not a tuning problem and not a sign error: the sensor cannot separate the two in that pose.
The feature was deleted. The level bar is unaffected because it reads **gravity**, not orientation.

---

## iOS reports `accelerationIncludingGravity` with the opposite sign to everyone else

A phone face-up on a table reads `z = -9.8` on iOS and `+9.8` on Android, which shifts the computed
roll by exactly 180°.

We don't detect the platform. A **line has no ends** — drawn at θ or θ+180 it looks identical — so
folding the angle into (−90, 90] makes the question disappear. (Detecting iOS via
`DeviceMotionEvent.requestPermission` does not work: Chromium defines it too, so that test reports
iOS on a Windows desktop.)

---

## No web page can write to the camera roll

Not on iOS, not on Android. There is no API. The system share sheet's "Save Image" is the only
route, which is why the web build's Save button opens the sheet rather than downloading — a
`download` attribute lands in Files, or in a Downloads folder the gallery never indexes, and reports
no failure either way.

The native build gets around it with a small Swift plugin (`ios/App/App/SavePhotoPlugin.swift`),
using add-only Photos authorisation so the app never asks to *read* the library.

---

## The viewfinder is 3:4 and the sensor is 4:3

The crop the user frames is the middle **56%** of the track's width. At a 1920×1440 request that
leaves 1080×1440 — 1.6MP — which is what a saved photo was actually worth. The crop cannot change
without changing what the app shows, so the only lever is the `getUserMedia` request.

`visibleCrop()` exists to keep the scanned frame, the coaching overlay and the saved photo all
agreeing on that same rectangle. When they disagreed, the engine analysed scenery the user could not
see and drew the marker off screen.

---

## `pointsOfInterest` is a Chromium extension, not a web standard

Tap to focus sets `focusMode` and `pointsOfInterest` through `applyConstraints`. Chromium implements
both. WebKit does not offer them, so on an iPhone, which is the platform this app ships on, the
constraint is absent and the tap does nothing.

The check is `navigator.mediaDevices.getSupportedConstraints()`, **not** `track.getCapabilities()`.
The spec lists `pointsOfInterest` under settings and constraints but not under capabilities, so
reading it off the capabilities would be false everywhere, including on the browsers where it works.

Where it is absent the tap draws no reticle either. A reticle animating over a lens that never moved
is the same lie as the Save button that said "Saved" without saving.

---

## A pose model is trained to find a person, not to decide whether one is there

Shown an empty room it will still offer its best guess, and the marker turned green with nobody in
shot. Three independent checks before believing it: landmark visibility floors raised to 0.7 (the
0.5 default is tuned for the other question), a minimum head-to-ankle span, and three consecutive
confirming frames.
