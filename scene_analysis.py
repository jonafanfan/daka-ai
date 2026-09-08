import base64
import io
import json
import logging
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

# Both OpenAI calls degrade rather than fail the scan, which means an outage is otherwise
# completely invisible: a bad completion and a dead API produce the same empty result. Log at
# warning so "why is every scene Unknown?" is answerable from the Render logs.
logger = logging.getLogger("daka.engine")

_openai_client = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _encode_image(image_path: str) -> str:
    """Resize to max 768px and encode as base64 JPEG to keep payload small."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((768, 768), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


VALID_FILTERS = ["Vivid", "Vivid Warm", "Vivid Cool", "Dramatic", "Dramatic Warm", "Dramatic Cool", "Silvertone", "Noir"]

# Blur gate (content-robust). Variance-of-Laplacian alone flags ANY low-texture scene
# (plain wall, minimalist cafe) as blurry, which blocks the whole flow. Instead we judge the
# SHARPNESS of the edges that actually exist, and give a near-featureless frame the benefit of
# the doubt (nothing to be blurry -> pass). Defaults are lenient (over-rejecting is the worse
# failure) and TUNABLE against real photos — /analyze echoes blur_var + edge_sharpness so you
# can read real values and calibrate.
MIN_EDGE_DENSITY = 0.008      # Canny edge fraction below which the scene is "too plain to judge"
EDGE_SHARPNESS_MIN = 8.0      # mean |Laplacian| at edges below this = genuinely soft / blurred


# Why the marker landed where it did. Kept short — this is drawn under the marker on a phone, so
# anything much longer than this wraps or runs off the frame.
PLACEMENT_REASONS = {
    "backlight":        "Out of the window glare",
    "light":            "Light falls on your face",
    "balance":          "Balances the busy side",
    "clean_background": "Cleaner background here",
    "default":          "Classic rule-of-thirds spot",
}


def _compute_placement(gray, saliency_map) -> dict:
    """Where a standing subject should stand -> normalized {x, y}, top-left origin, plus WHY.

    Fuses cheap signals — visual BALANCE (counterweight the scene's focal mass), background
    CLEANLINESS (over the vertical band the body occupies), and LIGHT direction (stand on the
    dimmer side so the light falls on the face) — each gated to only vote when it's reliable for
    this scene, plus a hard BACKLIGHT VETO (never stand in front of a blown-out region, which
    would silhouette the subject). x snaps to a rule-of-thirds line. Never raises; falls back
    to {2/3, 2/3}.

    `reason` names the signal that actually decided the side, so the client can explain the marker
    rather than showing an unexplained dot. The work was already being done and thrown away.
    """
    FALLBACK = {
        "x": round(2 / 3, 3), "y": round(2 / 3, 3),
        "reason": "default", "reason_text": PLACEMENT_REASONS["default"],
    }
    try:
        sal = np.asarray(saliency_map, dtype=np.float32)
        g = np.asarray(gray, dtype=np.float32)
        if sal.ndim != 2 or g.ndim != 2 or sal.size == 0:
            return FALLBACK
        H, W = sal.shape
        X_LEFT, X_RIGHT = 1 / 3, 2 / 3

        # Only trust a signal when it's meaningful for THIS scene (avoids deciding on noise).
        saliency_reliable = sal.std() > 0.010 and (float(sal.max()) - float(sal.min())) > 0.05
        light_reliable = float(g.mean()) > 40.0

        # Each signal's signed vote, kept separately rather than summed into one number, so the
        # winning signal can be named afterwards. +ve -> right (2/3), -ve -> left (1/3).
        contributions = {}

        if saliency_reliable:
            # Balance: stand opposite the scene's horizontal focal centre of mass.
            col = sal.sum(axis=0)
            tot = float(col.sum())
            cx = (float((np.arange(W) * col).sum() / tot) / W) if tot > 1e-6 else 0.5
            contributions["balance"] = 1.0 if cx < 0.5 else -1.0
            # Cleanliness: prefer the side whose body-band background is emptier.
            top = int(H * 0.30)

            def _clutter(nx):
                c = int(nx * W)
                return float(sal[top:, max(0, c - W // 6):min(W, c + W // 6)].mean())

            contributions["clean_background"] = 1.0 if _clutter(X_RIGHT) < _clutter(X_LEFT) else -1.0

        if light_reliable:
            # Light: stand on the DIMMER side so the brighter side lights the face.
            lb = float(g[:, :W // 2].mean())
            rb = float(g[:, W // 2:].mean())
            if abs(rb - lb) / (lb + rb + 1e-6) > 0.04:
                contributions["light"] = 1.2 if lb > rb else -1.2

        votes = sum(contributions.values())

        # Backlight veto: a blown-out half silhouettes the subject -> forbid standing there.
        hot = g > 245
        lhot = float(hot[:, :W // 2].mean())
        rhot = float(hot[:, W // 2:].mean())
        HOT = 0.06
        reason = None
        if rhot > HOT and rhot > lhot * 1.5:
            right = False
            reason = "backlight"
        elif lhot > HOT and lhot > rhot * 1.5:
            right = True
            reason = "backlight"
        elif votes > 0.15:
            right = True
        elif votes < -0.15:
            right = False
        else:
            right = not (rhot > lhot)   # no confident signal: avoid the hotter half; tie -> right
        x = X_RIGHT if right else X_LEFT

        if reason is None:
            # Credit the strongest signal that actually pointed the way we went. A signal that
            # voted the other way and lost is not the reason, even if it was the loudest.
            agreeing = {k: abs(v) for k, v in contributions.items() if v != 0 and (v > 0) == right}
            reason = max(agreeing, key=agreeing.get) if agreeing else "default"

        # Headroom: adapt y to where the saliency mass sits vertically.
        y = 2 / 3
        if saliency_reliable:
            m = float(sal.mean()) + 1e-6
            if float(sal[:H // 3].mean()) > 1.6 * m:
                y = 0.70
            elif float(sal[2 * H // 3:].mean()) > 1.8 * m:
                y = 0.62
        return {
            "x": round(float(x), 3),
            "y": round(min(0.72, max(0.60, y)), 3),
            "reason": reason,
            "reason_text": PLACEMENT_REASONS[reason],
        }
    except Exception:
        return FALLBACK


def _clean_hint(value) -> str:
    """The model's placement sentence, or "" when it gave us nothing usable.

    Shown verbatim above the shutter, so it is validated rather than trusted: wrong type, empty,
    or rambling all collapse to "" and the client renders no hint at all. A sentence that overflows
    the panel is worse than no sentence, and this is free-text from a model — the one field here
    that is not drawn from a closed set.
    """
    if not isinstance(value, str):
        return ""
    hint = " ".join(value.split()).rstrip(".")
    return hint if 0 < len(hint) <= 60 else ""


def _detect_dead_space(saliency_map) -> dict:
    """Is a third of the frame carrying nothing? Then aim the camera off it.

    Returns {"direction": "up" | "down" | "ok", "reason": str}. "down" means aim LOWER, because the
    dead space is above — blank ceiling or featureless sky eating the top of the shot. Advice no
    phone camera gives you, and it costs nothing: the saliency map is already computed for
    placement. Never raises.
    """
    OK = {"direction": "ok", "reason": ""}
    try:
        sal = np.asarray(saliency_map, dtype=np.float32)
        if sal.ndim != 2 or sal.size == 0 or sal.shape[0] < 3:
            return OK
        # Same reliability gate as placement: on a flat, low-contrast map the thirds are all noise
        # and any comparison between them is meaningless.
        if not (sal.std() > 0.010 and float(sal.max()) - float(sal.min()) > 0.05):
            return OK

        H = sal.shape[0]
        top = float(sal[:H // 3].mean())
        mid = float(sal[H // 3:2 * H // 3].mean())
        bot = float(sal[2 * H // 3:].mean())

        # A band is "dead" only if it is far emptier than the rest of the frame AND emptier than
        # the opposite band — otherwise a uniformly plain scene would trigger it constantly.
        DEAD = 0.45
        if top < DEAD * ((mid + bot) / 2) and top < bot:
            return {"direction": "down", "reason": "Empty space above — aim a little lower"}
        if bot < DEAD * ((top + mid) / 2) and bot < top:
            return {"direction": "up", "reason": "Empty floor below — aim a little higher"}
        return OK
    except Exception:
        return OK


def _moderate_image(b64: str) -> bool:
    """Returns True if the image is safe, False if flagged. Defaults to safe on API error."""
    try:
        response = _get_openai_client().moderations.create(
            model="omni-moderation-latest",
            input=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}],
        )
        return not response.results[0].flagged
    except Exception:
        # Fails OPEN: a moderation outage must not block every scan. This is a safety bypass, so
        # log it — silently waving images through is how you find out months later.
        logger.warning("moderation call failed; treating image as safe", exc_info=True)
        return True


def _analyze_with_gpt(b64: str, placement: dict | None = None) -> dict:
    """Scene name, filter and hashtags from the vision model. Never raises — returns {} instead.

    Every failure mode degrades to {}, which analyze_scene turns into safe defaults
    (scene_type "Unknown", no hashtags, filter "Vivid"). That matters because the OpenCV half of
    the scan — placement, framing, lighting, blur — does not depend on the model at all, so an
    OpenAI incident should cost the scene label, not the whole feature.
    """
    # Tell the model which side the geometry already picked, so its sentence agrees with the
    # marker instead of contradicting it. extract_features runs before this call, so it is known.
    side = "left" if (placement or {}).get("x", 0.667) < 0.5 else "right"
    try:
        response = _get_openai_client().chat.completions.create(
            model="gpt-5.4-nano",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "You are analysing a photo for a 打卡 (check-in) photography app used in China.\n"
                        "Return a JSON object with exactly these fields IN ENGLISH:\n"
                        "- \"scene_type\": concise scene name (e.g. \"Café\", \"City Street\", \"Beach\", \"Temple\")\n"
                        "- \"filter\": pick the best from exactly: "
                        "\"Vivid\", \"Vivid Warm\", \"Vivid Cool\", \"Dramatic\", \"Dramatic Warm\", \"Dramatic Cool\", \"Silvertone\", \"Noir\". "
                        "Use the Warm variants for cosy/golden-hour scenes, Cool for clean/urban/overcast scenes, "
                        "Dramatic for moody or high-contrast scenes, and the black & white options (Silvertone soft, Noir high-contrast) "
                        "only when colour adds little.\n"
                        "- \"hashtags\": array of exactly 3 relevant hashtags with # symbol, all lowercase\n"
                        "- \"placement_hint\": ONE short instruction, at most 8 words, telling the person "
                        "where to stand. Anchor it to something actually visible in the photo, and make "
                        "the DEPTH clear — how far INTO the scene to stand. The app already shows the "
                        "left/right position on screen but cannot show depth, so depth is the whole point "
                        "of this field. Use phrasing like \"in front of\", \"just behind\", \"beside\", "
                        "\"level with\". Examples: \"Stand in front of the blue door\", "
                        "\"Stand just behind the low wall\", \"Stand beside the window, nearer than the plant\". "
                        "No trailing full stop.\n"
                        f"The spot is on the {side} side of the frame — keep the instruction consistent "
                        "with that side."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            max_completion_tokens=500,
        )
    except Exception:
        # Timeout, rate limit, auth failure, outage. Degrades like a bad completion: the OpenCV
        # features are already computed by the time this is called, so failing the scan here would
        # throw away work that succeeded and break placement/framing, which never needed the model.
        logger.warning("vision call failed; falling back to defaults", exc_info=True)
        return {}

    # A truncated, empty, or refused completion must degrade — not 500 the whole scan.
    # analyze_scene reads every field via gpt.get(...), so {} falls back to safe defaults.
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if (choice and choice.message) else None
    if not content:
        return {}
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    # response_format=json_object should guarantee an object, but a bare array/string/null would
    # reach analyze_scene's gpt.get(...) calls and raise AttributeError — a 500 from the one
    # function whose contract is never to cause one.
    if not isinstance(parsed, dict):
        logger.warning("vision call returned non-object JSON (%s); ignoring", type(parsed).__name__)
        return {}
    return parsed


def extract_features(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image '{image_path}' could not be loaded.")

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    # cv2 loads BGR, so avg_color is [blue, green, red] and color_ratio is BLUE / RED.
    # Mind the direction: a HIGHER ratio means MORE BLUE, i.e. a COOLER scene. (assess_lighting
    # had this backwards and reported warm scenes as cool.)
    avg_color = np.mean(img, axis=(0, 1))
    color_ratio = float(avg_color[0] / (avg_color[2] + 1e-5))

    edges = cv2.Canny(gray, 100, 200)
    sharpness = float(np.sum(edges > 0) / (h * w + 1e-6))     # Canny edge density

    # Content-robust blur: judge whether the edges that DO exist are crisp; a near-featureless
    # frame has nothing to be blurry -> not blurry. (blur_var kept as a diagnostic.)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    edge_mask = edges > 0
    if sharpness < MIN_EDGE_DENSITY:
        edge_sharpness = -1.0            # too plain to judge -> benefit of the doubt
        blurry = False
    else:
        edge_sharpness = float(np.abs(lap[edge_mask]).mean())
        blurry = edge_sharpness < EDGE_SHARPNESS_MIN

    saliency_engine = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency_engine.computeSaliency(img)

    intersections = [
        (h // 3, w // 3), (h // 3, 2 * w // 3),
        (2 * h // 3, w // 3), (2 * h // 3, 2 * w // 3),
    ]
    roi_h, roi_w = int(h * 0.1), int(w * 0.1)
    thirds_scores = []
    for (y, x) in intersections:
        roi = saliency_map[
            max(0, y - roi_h):min(h, y + roi_h),
            max(0, x - roi_w):min(w, x + roi_w),
        ]
        thirds_scores.append(float(np.mean(roi)))
    rule_of_thirds = max(thirds_scores) if thirds_scores else 0.0

    # Suggested subject placement (normalized, top-left origin): balance / background cleanliness /
    # light direction fused, with a backlight veto. See _compute_placement.
    placement = _compute_placement(gray, saliency_map)
    # Dead-space check: reuses the same saliency map, so this costs one more pass over an array
    # that is already in memory.
    camera_tilt = _detect_dead_space(saliency_map)

    left_weight = float(np.mean(saliency_map[:, :w // 2]))
    right_weight = float(np.mean(saliency_map[:, w // 2:]))
    balance = 1.0 - abs(left_weight - right_weight) / (left_weight + right_weight + 1e-5)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    alignment = 1.0
    if lines is not None:
        deviations = []
        for l in lines:
            # HoughLinesP shape varies by OpenCV build: (N,1,4) or (N,4). Flatten to be safe.
            x1, y1, x2, y2 = np.asarray(l).ravel()[:4]
            deviations.append(abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 90)
        deviations = [d if d < 45 else 90 - d for d in deviations]
        critical = [d for d in deviations if 0.5 < d < 20]
        if critical:
            alignment = max(0.0, 1.0 - (float(np.mean(critical)) / 15.0))

    return {
        "brightness": brightness,
        "color_ratio": color_ratio,
        "sharpness": sharpness,
        "blur_var": blur_var,
        "edge_sharpness": edge_sharpness,
        "blurry": blurry,
        "rule_of_thirds": rule_of_thirds,
        "alignment": alignment,
        "balance": float(balance),
        "placement": placement,
        "camera_tilt": camera_tilt,
        "width": int(w),
        "height": int(h),
    }


def build_blueprint(features: dict) -> dict:
    h, w = features["height"], features["width"]
    notes = []
    if features["alignment"] < 0.7:
        notes.append("tilted horizon — consider straightening")
    if features["balance"] < 0.6:
        notes.append("unbalanced composition — subject may be off-centre")
    if features["rule_of_thirds"] > 0.5:
        notes.append("strong rule-of-thirds alignment")
    return {"orientation": "portrait" if h >= w else "landscape", "grid": "rule_of_thirds", "notes": notes}


def assess_lighting(features: dict) -> dict:
    brightness = features["brightness"]
    color_ratio = features["color_ratio"]

    if 100 < brightness < 200:
        quality = "Good"
    elif 60 < brightness <= 100 or 200 <= brightness < 230:
        quality = "Fair"
    else:
        quality = "Poor"

    # color_ratio is blue/red (see extract_features): high = blue-dominant = Cool.
    # Thresholds are unchanged from the original calibration — only the labels were swapped —
    # so the neutral band still sits where it was tuned (most scenes carry a mild red bias,
    # which is why the band is centred near 0.8 rather than 1.0).
    if color_ratio > 0.9:
        tone = "Cool"
    elif color_ratio < 0.7:
        tone = "Warm"
    else:
        tone = "Neutral"

    tips = {
        ("Good", "Warm"): "Good natural light, shoot facing forward",
        ("Good", "Cool"): "Nice cool tones, use them for a clean aesthetic",
        ("Good", "Neutral"): "Balanced light, great for any angle",
        ("Fair", "Warm"): "Slightly dim, move closer to the light source",
        ("Fair", "Cool"): "A bit dim, try adjusting white balance",
        ("Fair", "Neutral"): "Slightly dim, try opening a curtain or moving nearer a window",
        ("Poor", "Warm"): "Too dark or overexposed, find a better-lit spot",
        ("Poor", "Cool"): "Harsh or insufficient light, reposition or wait for better conditions",
        ("Poor", "Neutral"): "Lighting is off, look for softer indirect light",
    }
    return {"quality": quality, "tone": tone, "tip": tips.get((quality, tone), "Adjust your position for better light")}


def assess_composition(features: dict) -> dict:
    sharpness, alignment, balance = features["sharpness"], features["alignment"], features["balance"]
    focus = "Sharp" if sharpness > 0.1 else "Soft" if sharpness > 0.05 else "Blurry"
    horizon = "Level" if alignment > 0.8 else "Slightly tilted" if alignment > 0.6 else "Tilted"
    symmetry = "Balanced" if balance > 0.8 else "Slightly off" if balance > 0.6 else "Unbalanced"
    return {"focus": focus, "horizon": horizon, "balance": symmetry}


class InappropriateImageError(ValueError):
    pass


def analyze_scene(image_path: str) -> dict:
    b64 = _encode_image(image_path)
    if not _moderate_image(b64):
        raise InappropriateImageError("Image flagged as inappropriate")
    features = extract_features(image_path)
    gpt = _analyze_with_gpt(b64, features["placement"])

    filter_name = gpt.get("filter", "Vivid")
    if filter_name not in VALID_FILTERS:
        filter_name = "Vivid"

    return {
        "scene_type":   gpt.get("scene_type", "Unknown"),
        "blueprint":    build_blueprint(features),
        "lighting":     assess_lighting(features),
        "blurry":       features["blurry"],
        "blur_var":     round(features["blur_var"], 1),          # diagnostic — for tuning the gate
        "edge_sharpness": round(features["edge_sharpness"], 2),  # diagnostic — for tuning the gate
        "composition":  assess_composition(features),
        "placement":    features["placement"],
        "camera_tilt":  features["camera_tilt"],
        "placement_hint": _clean_hint(gpt.get("placement_hint")),
        "hashtags":     gpt.get("hashtags", []),
        "filter":       filter_name,
    }
