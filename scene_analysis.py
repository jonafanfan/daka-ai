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
    """Resize to max 768px and encode as base64 JPEG to keep the payload small."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((768, 768), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


VALID_FILTERS = ["Vivid", "Vivid Warm", "Vivid Cool", "Dramatic", "Dramatic Warm", "Dramatic Cool", "Silvertone", "Noir"]

# Steadiness gate: variance-of-Laplacian below this reads as "too blurry to coach against".
# Tunable — measured on the uploaded analysis frame (~1024px longest side).
BLUR_THRESHOLD = 45.0

# Camera-tilt heuristic (dead-space detection from edge density).
_THIN_AIR = 0.04
_TILT_RATIO = 4.0


class InappropriateImageError(ValueError):
    pass


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
    """Ask GPT only for the copy the app renders: scene name, filter, hashtags."""
    response = _get_openai_client().chat.completions.create(
        model="gpt-5.4-nano",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You are analysing an empty scene for a 打卡 (check-in) photography app. "
                    "Return a JSON object with EXACTLY these fields, values in English:\n"
                    "- \"scene_type\": concise scene name (e.g. \"Café\", \"City Street\", \"Beach\").\n"
                    "- \"filter\": one of exactly \"Vivid\", \"Vivid Warm\", \"Vivid Cool\", \"Dramatic\", "
                    "\"Dramatic Warm\", \"Dramatic Cool\", \"Silvertone\", \"Noir\".\n"
                    "- \"hashtags\": array of exactly 3 lowercase hashtags, each starting with #."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        max_completion_tokens=300,
    )
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if (choice and choice.message) else None
    if not content:
        return {}
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def extract_features(image_path: str) -> dict:
    """Cheap OpenCV signals that each drive a concrete user action:
    lighting gate, steadiness gate, where-to-stand (saliency), and tilt (dead space)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image '{image_path}' could not be loaded.")

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lighting
    brightness = float(np.mean(gray))
    avg_color = np.mean(img, axis=(0, 1))
    color_ratio = float(avg_color[0] / (avg_color[2] + 1e-5))   # B/R

    # Steadiness (variance of Laplacian; higher = sharper)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Dead-space detection (edge density by vertical third) — feeds tilt hint
    edges = cv2.Canny(gray, 100, 200)
    top_edge = float(np.sum(edges[:h // 3, :] > 0)) / (max(h // 3, 1) * w + 1e-6)
    mid_edge = float(np.sum(edges[h // 3:2 * h // 3, :] > 0)) / (max(h // 3, 1) * w + 1e-6)
    bot_edge = float(np.sum(edges[2 * h // 3:, :] > 0)) / (max(h - 2 * h // 3, 1) * w + 1e-6)

    # Saliency → where the visual "mass" is, and how empty each rule-of-thirds point is
    saliency_engine = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency_engine.computeSaliency(img)

    intersections = [
        (h // 3, w // 3), (h // 3, 2 * w // 3),          # TL, TR
        (2 * h // 3, w // 3), (2 * h // 3, 2 * w // 3),   # BL, BR
    ]
    roi_h, roi_w = int(h * 0.1), int(w * 0.1)
    thirds_scores = []
    for (y, x) in intersections:
        roi = saliency_map[
            max(0, y - roi_h):min(h, y + roi_h),
            max(0, x - roi_w):min(w, x + roi_w),
        ]
        thirds_scores.append(float(np.mean(roi)))

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

    return {
        "brightness": brightness,
        "color_ratio": color_ratio,
        "sharpness": sharpness,
        "width": int(w),
        "height": int(h),
        "saliency_centroid": saliency_centroid,
        "thirds_scores": [float(s) for s in thirds_scores],
        "edge_density_top": top_edge,
        "edge_density_mid": mid_edge,
        "edge_density_bot": bot_edge,
    }


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


def _compute_target(features: dict) -> dict:
    """Where the subject should stand: the emptiest of the two lower-third intersections
    (cleanest background, natural for a standing/sitting subject). Normalised, top-left origin."""
    scores = features.get("thirds_scores") or [0.0, 0.0, 0.0, 0.0]
    bl, br = scores[2], scores[3]
    x = (1.0 / 3.0) if bl <= br else (2.0 / 3.0)
    return {"x": round(x, 3), "y": round(2.0 / 3.0, 3), "size": 0.6}


def _compute_tilt_hint(features: dict) -> str:
    """From vertical edge density: dead space at top -> tilt down to cut it; dead space at
    bottom -> tilt up. Otherwise 'ok'."""
    e_top = features.get("edge_density_top", 0.5)
    e_mid = features.get("edge_density_mid", 0.5)
    e_bot = features.get("edge_density_bot", 0.5)
    if e_mid > _THIN_AIR * _TILT_RATIO and e_top < _THIN_AIR and e_mid / max(e_top, 1e-9) > _TILT_RATIO:
        return "down"
    if e_mid > _THIN_AIR * _TILT_RATIO and e_bot < _THIN_AIR and e_mid / max(e_bot, 1e-9) > _TILT_RATIO:
        return "up"
    return "ok"


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
        "coord_space": "normalized_topleft",
        "width": features["width"],
        "height": features["height"],
        "lighting": assess_lighting(features),
        "blurry": features["sharpness"] < BLUR_THRESHOLD,
        "scene_type": gpt.get("scene_type", "Unknown"),
        "filter": filter_name,
        "hashtags": gpt.get("hashtags", []),
        "target": _compute_target(features),
        "tilt_hint": _compute_tilt_hint(features),
    }
