import base64
import io
import json
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

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

# --- AI Engine output contract (see CONTRACT.md) ---
CONTRACT_VERSION = "1.0"
PLACEMENT_DEADBAND = 0.04     # normalised; |offset| below this reads as "ok"
OBJECT_CONF_FLOOR = 0.0       # TODO(team): objects[] confidence floor — pending sign-off (CONTRACT.md §7)
MAX_OBJECTS = 8

# Rule-of-thirds intersections as (name, x, y), normalised, top-left origin.
# Order MUST match the `intersections` list in extract_features (TL, TR, BL, BR).
_THIRDS = [
    ("top-left", 1 / 3, 1 / 3), ("top-right", 2 / 3, 1 / 3),
    ("bottom-left", 1 / 3, 2 / 3), ("bottom-right", 2 / 3, 2 / 3),
]

# framing_suggestions[] closed enums (CONTRACT.md §3.3) + scene_yap cap (§3.4).
# Anchors are grouped by target so the two can't contradict (subject can't get a camera anchor).
FRAMING_TARGETS = {"subject", "camera"}
SUBJECT_ANCHORS = {"left_third", "right_third", "center", "upper_third", "lower_third"}
CAMERA_ANCHORS = {
    "tilt_up", "tilt_down", "pan_left", "pan_right",
    "step_back", "step_closer", "raise_camera", "lower_camera", "level_horizon",
}
FRAMING_ANCHORS = SUBJECT_ANCHORS | CAMERA_ANCHORS
SCENE_YAP_MAXLEN = 100        # contract §3.4 says <= ~90 chars; small margin

# Candidate target points (4 rule-of-thirds intersections + centre) for labelling target.intersection.
_TARGET_POINTS = [
    ("top-left", 1 / 3, 1 / 3), ("top-right", 2 / 3, 1 / 3),
    ("bottom-left", 1 / 3, 2 / 3), ("bottom-right", 2 / 3, 2 / 3),
    ("center", 0.5, 0.5),
]


def _moderate_image(b64: str) -> bool:
    """Returns True if the image is safe, False if flagged. Defaults to safe on API error."""
    try:
        response = _get_openai_client().moderations.create(
            model="omni-moderation-latest",
            input=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}],
        )
        return not response.results[0].flagged
    except Exception:
        return True


def _analyze_with_gpt(b64: str) -> dict:
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
                    "- \"pose_tips\": array of exactly 3 tips about the PERSON's body — posture, gaze, hands, "
                    "shoulder angle — specific to this scene's light and space. Never mention where to stand "
                    "or where the camera goes (that is framing_suggestions).\n"
                    "- \"objects\": array of UP TO 8 notable things you can see, each "
                    "{\"label\": short name, \"box\": {\"x\":num,\"y\":num,\"w\":num,\"h\":num}, \"confidence\": num 0..1}. "
                    "ALL coordinates are NORMALISED 0..1 with the ORIGIN at the TOP-LEFT of the frame; "
                    "box x,y = top-left corner, w,h = width,height as fractions of the frame.\n"
                    "- \"subject_placement\": where a PERSON should STAND for the best shot here. "
                    "The person is NOT in the frame yet — judge from the empty scene, its light and its space. "
                    "{\"point\": {\"x\":num,\"y\":num} normalised top-left origin = where the person should stand, "
                    "\"size\": suggested person height as a fraction of frame height, "
                    "\"anchor\": \"feet\" or \"center\", \"reason\": one short sentence why}.\n"
                    "- \"framing_suggestions\": array of up to 3 objects, each "
                    "{\"target\": \"subject\" or \"camera\", \"instruction\": imperative <=12 words on WHERE to "
                    "place the subject or camera, \"anchor\": a tag matching the target}. "
                    "If target is \"subject\", anchor MUST be one of "
                    "[left_third,right_third,center,upper_third,lower_third]. "
                    "If target is \"camera\", anchor MUST be one of "
                    "[tilt_up,tilt_down,pan_left,pan_right,step_back,step_closer,raise_camera,lower_camera,level_horizon]. "
                    "About PLACEMENT/FRAMING (use lines, doorways, windows, empty space); never body language.\n"
                    "- \"camera_tilt\": whether the camera should be tilted up or down for the best composition. "
                    "Pick \"up\" when interesting detail (ceiling, sky, upper architecture) sits above centre "
                    "and should be included. Pick \"down\" when the foreground detail (table, floor, leading lines) "
                    "should anchor the frame. \"ok\" when the scene is balanced as-is. "
                    "{\"direction\": \"up\" or \"down\" or \"ok\", "
                    "\"degrees\": integer 0-30 estimating how many degrees to tilt, "
                    "\"reason\": one short sentence why}.\n"
                    "- \"scene_yap\": ONE fun, shareable sentence (max ~90 chars) in a hyped app voice about the "
                    "vibe of this scene. Flavour, not advice. English, but you MAY include the word 打卡. "
                    "No hashtags, at most one emoji.\n"
                    "Match these nested key names EXACTLY (confidence is a decimal 0..1, NOT a percentage): "
                    "{\"objects\":[{\"label\":\"window\",\"box\":{\"x\":0.05,\"y\":0.10,\"w\":0.30,\"h\":0.55},\"confidence\":0.88}],"
                    "\"subject_placement\":{\"point\":{\"x\":0.33,\"y\":0.62},\"size\":0.7,\"anchor\":\"feet\",\"reason\":\"...\"},"
                    "\"camera_tilt\":{\"direction\":\"up\",\"degrees\":5,\"reason\":\"Tilt up to capture the ceiling\"},"
                    "\"framing_suggestions\":[{\"target\":\"subject\",\"instruction\":\"Stand in the lower-left third\",\"anchor\":\"left_third\"}]}"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        max_completion_tokens=1800,
    )
    # gpt-5.x nano is a reasoning model: max_completion_tokens covers reasoning + visible
    # output, so a length cutoff (or a refusal) can yield partial or empty content. Degrade to
    # {} on any bad payload so the downstream validators fall back to their defaults rather
    # than 500-ing the whole request (CONTRACT.md §5 — the engine always degrades).
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if (choice and choice.message) else None
    if not content:
        return {}
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def extract_features(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image '{image_path}' could not be loaded.")

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    avg_color = np.mean(img, axis=(0, 1))
    color_ratio = float(avg_color[0] / (avg_color[2] + 1e-5))

    edges = cv2.Canny(gray, 100, 200)
    sharpness = float(np.sum(edges > 0) / (h * w + 1e-6))

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

    left_weight = float(np.mean(saliency_map[:, :w // 2]))
    right_weight = float(np.mean(saliency_map[:, w // 2:]))
    balance = 1.0 - abs(left_weight - right_weight) / (left_weight + right_weight + 1e-5)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    alignment = 1.0
    if lines is not None:
        deviations = [
            abs(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) % 90
            for l in lines
        ]
        deviations = [d if d < 45 else 90 - d for d in deviations]
        critical = [d for d in deviations if 0.5 < d < 20]
        if critical:
            alignment = max(0.0, 1.0 - (float(np.mean(critical)) / 15.0))

    # --- Subject-location signals (normalised, top-left origin) — surfaced for the framing output ---
    sal = np.asarray(saliency_map, dtype=np.float64)
    sal_total = float(sal.sum())
    if sal_total > 1e-9:
        xs = np.arange(w, dtype=np.float64).reshape(1, -1)
        ys = np.arange(h, dtype=np.float64).reshape(-1, 1)
        cx = float((sal * xs).sum() / sal_total) / max(w, 1)
        cy = float((sal * ys).sum() / sal_total) / max(h, 1)
    else:
        cx, cy = 0.5, 0.5
    saliency_centroid = {"x": min(1.0, max(0.0, cx)), "y": min(1.0, max(0.0, cy))}

    best_idx = int(np.argmax(thirds_scores)) if thirds_scores else 0
    third_name, third_x, third_y = _THIRDS[best_idx]
    strongest_third = {"intersection": third_name, "x": third_x, "y": third_y}

    # Signed tilt of the scene's horizon (positive = slopes down toward the right, image space).
    horizon_tilt = 0.0
    if lines is not None:
        near_horizontal = []
        for l in lines:
            ang = float(np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])))
            ang = (ang + 90) % 180 - 90          # fold to (-90, 90]
            if abs(ang) < 20:                     # near-horizontal lines only
                near_horizontal.append(ang)
        if near_horizontal:
            horizon_tilt = float(np.median(near_horizontal))

    return {
        "brightness": brightness,
        "color_ratio": color_ratio,
        "sharpness": sharpness,
        "rule_of_thirds": rule_of_thirds,
        "alignment": alignment,
        "balance": float(balance),
        "width": int(w),
        "height": int(h),
        # --- new: subject-location signals (T1) ---
        "saliency_centroid": saliency_centroid,
        "thirds_scores": [float(s) for s in thirds_scores],
        "strongest_third": strongest_third,
        "horizon_tilt_deg": round(horizon_tilt, 1),
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

    if color_ratio > 0.9:
        tone = "Warm"
    elif color_ratio < 0.7:
        tone = "Cool"
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


def _clamp01(value, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _nearest_intersection(x: float, y: float) -> str:
    """Name the rule-of-thirds point (or centre) closest to (x, y) — keeps target.intersection
    consistent with target.x/y (CONTRACT.md §3.2)."""
    return min(_TARGET_POINTS, key=lambda p: (x - p[1]) ** 2 + (y - p[2]) ** 2)[0]


def _validate_objects(raw) -> list:
    """Coerce GPT 'objects' into clean, clamped, capped entries. Never raises."""
    if not isinstance(raw, list):
        return []
    cleaned = []
    for o in raw:
        if not isinstance(o, dict):
            continue
        box = o.get("box")
        if not isinstance(box, dict):
            continue
        try:
            conf = max(0.0, min(1.0, float(o.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < OBJECT_CONF_FLOOR:
            continue
        cleaned.append({
            "label": str(o.get("label", "object"))[:40],
            "box": {
                "x": _clamp01(box.get("x")), "y": _clamp01(box.get("y")),
                "w": _clamp01(box.get("w")), "h": _clamp01(box.get("h")),
            },
            "confidence": round(conf, 2),
        })
    cleaned.sort(key=lambda c: c["confidence"], reverse=True)
    return cleaned[:MAX_OBJECTS]


def _validate_framing_suggestions(raw) -> list:
    """Coerce GPT 'framing_suggestions' against the closed enums (CONTRACT.md §3.3). Never raises.

    Unknown anchors become null (consumers fall back to showing instruction text); items with no
    instruction are dropped; capped at 3.
    """
    if not isinstance(raw, list):
        return []
    out = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        instruction = str(s.get("instruction", "")).strip()[:120]
        if not instruction:
            continue
        target = s.get("target") if s.get("target") in FRAMING_TARGETS else "subject"
        allowed = SUBJECT_ANCHORS if target == "subject" else CAMERA_ANCHORS
        anchor = s.get("anchor")
        out.append({
            "target": target,
            "instruction": instruction,
            "anchor": anchor if anchor in allowed else None,   # anchor must match its target group
        })
        if len(out) >= 3:
            break
    return out


def _validate_camera_tilt(raw) -> dict:
    """Coerce GPT 'camera_tilt' into the closed enum set + safe degrees. Never raises."""
    if not isinstance(raw, dict):
        return {"direction": "ok", "degrees": 0, "reason": ""}
    direction = raw.get("direction", "ok")
    if direction not in ("up", "down", "ok"):
        direction = "ok"
    try:
        degrees = max(0, min(30, int(float(raw.get("degrees", 0)))))
    except (TypeError, ValueError):
        degrees = 0
    reason = str(raw.get("reason", ""))[:120] if raw.get("reason") else ""
    return {"direction": direction, "degrees": degrees, "reason": reason}


def _build_framing(gpt: dict, features: dict) -> dict:
    """Assemble the authoritative `framing` object (CONTRACT.md §3.2). Never raises.

    On an empty-scene scan there is no live person, so `target` (where to stand) is the
    load-bearing output: GPT's subject_placement first, falling back to the strongest
    saliency rule-of-thirds intersection so a usable point is ALWAYS present. The saliency
    centroid is used as the reference the guidance arrow points *from*; true per-frame
    person tracking is Capture's optional client-side layer (CONTRACT.md §6).
    """
    sp = gpt.get("subject_placement")
    sp = sp if isinstance(sp, dict) else {}

    # target: where the subject should stand (GPT first, saliency rule-of-thirds fallback)
    fallback = features.get("strongest_third") or {"intersection": "center", "x": 0.5, "y": 0.5}
    point = sp.get("point") if isinstance(sp.get("point"), dict) else None
    if point is not None and (point.get("x") is not None or point.get("y") is not None):
        tx = _clamp01(point.get("x"), fallback["x"])
        ty = _clamp01(point.get("y"), fallback["y"])
        # label from the ACTUAL point, not the saliency fallback, so intersection matches x/y
        target = {"intersection": _nearest_intersection(tx, ty), "x": tx, "y": ty}
    else:
        target = {
            "intersection": fallback["intersection"],
            "x": float(fallback["x"]), "y": float(fallback["y"]),
        }

    # subject reference (no live person on an empty scene -> detected:false, saliency centroid)
    centroid = features.get("saliency_centroid") or {"x": 0.5, "y": 0.5}
    size = sp.get("size")
    subject = {
        "detected": False,
        "source": "saliency",
        "label": "salient_region",
        "confidence": 0.0,
        "center": {"x": _clamp01(centroid.get("x"), 0.5), "y": _clamp01(centroid.get("y"), 0.5)},
        "bbox": None,
        "size": _clamp01(size) if size is not None else None,
    }

    # guidance: nudge from the reference toward the target (deadband -> "ok")
    dx = target["x"] - subject["center"]["x"]
    dy = target["y"] - subject["center"]["y"]
    guidance = {
        "move_subject_x": "right" if dx > PLACEMENT_DEADBAND else "left" if dx < -PLACEMENT_DEADBAND else "ok",
        "move_subject_y": "down" if dy > PLACEMENT_DEADBAND else "up" if dy < -PLACEMENT_DEADBAND else "ok",
        "distance": "ok",                        # needs a live subject size; filled by Capture's live layer
        "dx": round(dx, 3), "dy": round(dy, 3),
        "strength": round(min(1.0, (dx * dx + dy * dy) ** 0.5), 3),
    }

    # level: static scene tilt (live device roll stays frontend-owned)
    alignment = float(features.get("alignment", 1.0))
    level = {
        "scene_horizon_tilt_deg": float(features.get("horizon_tilt_deg", 0.0)),
        "needs_straightening": alignment < 0.7,
        "source_alignment": round(alignment, 3),
    }

    reason = sp.get("reason")

    # camera_tilt: derived from subject_placement, using GPT's reason if available.
    # If the subject should stand in the upper third of the frame (low y) the camera
    # tilts down to achieve that framing; if in the lower third (high y) tilt up.
    camera_tilt = {"direction": "ok", "degrees": 0, "reason": ""}
    point = sp.get("point") if isinstance(sp.get("point"), dict) else None
    if point is not None and point.get("y") is not None:
        py = _clamp01(point.get("y"), 0.5)
        if py < 0.35:
            tilt_deg = min(30, round((0.35 - py) * 50))
            camera_tilt = {"direction": "down", "degrees": tilt_deg, "reason": ""}
        elif py > 0.65:
            tilt_deg = min(30, round((py - 0.65) * 50))
            camera_tilt = {"direction": "up", "degrees": tilt_deg, "reason": ""}
    gpt_ct = gpt.get("camera_tilt")
    if isinstance(gpt_ct, dict) and isinstance(gpt_ct.get("reason"), str) and gpt_ct["reason"].strip():
        camera_tilt["reason"] = gpt_ct["reason"].strip()[:120]

    return {
        "subject": subject,
        "target": target,
        "guidance": guidance,
        "level": level,
        "camera_tilt": camera_tilt,
        "reason": str(reason)[:160] if reason else "",
    }


def analyze_scene(image_path: str) -> dict:
    b64 = _encode_image(image_path)
    if not _moderate_image(b64):
        raise InappropriateImageError("Image flagged as inappropriate")
    features = extract_features(image_path)
    gpt = _analyze_with_gpt(b64)

    filter_name = gpt.get("filter", "Vivid")
    if filter_name not in VALID_FILTERS:
        filter_name = "Vivid"

    return {
        "contract_version": CONTRACT_VERSION,
        "coord_space":  "normalized_topleft",
        "scene_type":   gpt.get("scene_type", "Unknown"),
        "blueprint":    build_blueprint(features),
        "lighting":     assess_lighting(features),
        "composition":  assess_composition(features),
        "pose_tips":    gpt.get("pose_tips", []),
        "hashtags":     gpt.get("hashtags", []),
        "filter":       filter_name,
        "objects":      _validate_objects(gpt.get("objects")),
        "framing":      _build_framing(gpt, features),
        "framing_suggestions": _validate_framing_suggestions(gpt.get("framing_suggestions")),
        "scene_yap":    str(gpt.get("scene_yap", "")).strip()[:SCENE_YAP_MAXLEN],
    }
