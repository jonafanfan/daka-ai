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

# Steadiness gate: variance-of-Laplacian below this reads as "too blurry to use". Tunable.
BLUR_THRESHOLD = 45.0

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
                    "- \"pose_tips\": array of exactly 3 specific pose tips based on what you can see — "
                    "lighting direction, available space, background, furniture, windows, etc. "
                    "Be specific to this exact scene, not generic."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        max_completion_tokens=500,
    )
    return json.loads(response.choices[0].message.content)


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
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())   # higher = sharper

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

    # Suggested subject placement: of the two lower-third intersections (natural for a standing
    # subject), pick the emptiest (lowest saliency = cleanest background). Returned normalized.
    placement_x = (1.0 / 3.0) if thirds_scores[2] <= thirds_scores[3] else (2.0 / 3.0)
    placement = {"x": round(placement_x, 3), "y": round(2.0 / 3.0, 3)}

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
        "rule_of_thirds": rule_of_thirds,
        "alignment": alignment,
        "balance": float(balance),
        "placement": placement,
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
        "scene_type":   gpt.get("scene_type", "Unknown"),
        "blueprint":    build_blueprint(features),
        "lighting":     assess_lighting(features),
        "blurry":       features["blur_var"] < BLUR_THRESHOLD,
        "composition":  assess_composition(features),
        "placement":    features["placement"],
        "pose_tips":    gpt.get("pose_tips", []),
        "hashtags":     gpt.get("hashtags", []),
        "filter":       filter_name,
    }
