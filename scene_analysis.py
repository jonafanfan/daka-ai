import base64
import json
import threading
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

_clip_model = None
_clip_processor = None
_clip_lock = threading.Lock()
_openai_client = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

SCENE_LABELS = [
    ("Café or Restaurant",     "a photo taken inside a cafe or restaurant"),
    ("Bar or Nightclub",       "a photo taken inside a bar or nightclub at night"),
    ("Night Market",           "a photo taken at an outdoor night market with food stalls and lights"),
    ("Shopping Mall",          "a photo taken inside a shopping mall"),
    ("Convenience Store",      "a photo taken inside a convenience store"),
    ("Museum or Gallery",      "a photo taken inside a museum or art gallery"),
    ("Hotel Lobby",            "a photo taken inside a hotel lobby"),
    ("Office or Workspace",    "a photo taken inside an office or co-working space"),
    ("Street",                 "a photo of a city street during the day"),
    ("Rooftop or Balcony",     "a photo taken from a rooftop or balcony overlooking a city"),
    ("Park or Garden",         "a photo of a park or garden with greenery"),
    ("Beach",                  "a photo of a sandy beach near the ocean"),
    ("Mountain or Hiking Trail","a photo on a mountain trail or hiking path in nature"),
    ("Temple or Historic Site","a photo of a temple, pagoda, or historic monument"),
    ("Gym or Sports Venue",    "a photo taken inside a gym or sports facility"),
    ("Airport or Train Station","a photo taken inside an airport terminal or train station"),
]

def _load_clip():
    global _clip_model, _clip_processor
    with _clip_lock:
        if _clip_model is None:
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def extract_features(image_path: str) -> dict:
    """
    Extract core aesthetic features from an image.
    Returns a dict with brightness, color_ratio, sharpness, rule_of_thirds,
    alignment, balance, width, and height.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image '{image_path}' could not be loaded.")

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Brightness & color balance
    brightness = float(np.mean(gray))
    avg_color = np.mean(img, axis=(0, 1))
    color_ratio = float(avg_color[0] / (avg_color[2] + 1e-5))  # blue / red

    # Sharpness via edge density
    edges = cv2.Canny(gray, 100, 200)
    sharpness = float(np.sum(edges > 0) / (h * w + 1e-6))

    # Saliency map for composition metrics
    saliency_engine = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency_engine.computeSaliency(img)

    # Rule of Thirds metric
    intersections = [
        (h // 3, w // 3),
        (h // 3, 2 * w // 3),
        (2 * h // 3, w // 3),
        (2 * h // 3, 2 * w // 3),
    ]
    roi_h, roi_w = int(h * 0.1), int(w * 0.1)
    thirds_scores = []
    for (y, x) in intersections:
        roi = saliency_map[
            max(0, y - roi_h) : min(h, y + roi_h),
            max(0, x - roi_w) : min(w, x + roi_w),
        ]
        thirds_scores.append(float(np.mean(roi)))

    rule_of_thirds = max(thirds_scores) if thirds_scores else 0.0

    # Visual balance
    left_weight = float(np.mean(saliency_map[:, : w // 2]))
    right_weight = float(np.mean(saliency_map[:, w // 2 :]))
    balance = 1.0 - abs(left_weight - right_weight) / (left_weight + right_weight + 1e-5)

    # Alignment (geometry / horizon)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        100,
        minLineLength=80,
        maxLineGap=10,
    )
    alignment = 1.0
    if lines is not None:
        deviations = [
            abs(
                np.degrees(
                    np.arctan2(
                        l[0][3] - l[0][1],
                        l[0][2] - l[0][0],
                    )
                )
            )
            % 90
            for l in lines
        ]
        deviations = [d if d < 45 else 90 - d for d in deviations]
        critical = [d for d in deviations if 0.5 < d < 20]
        if critical:
            alignment = max(0.0, 1.0 - (float(np.mean(critical)) / 15.0))

    return {
        "brightness": brightness,
        "color_ratio": color_ratio,
        "sharpness": sharpness,
        "rule_of_thirds": rule_of_thirds,
        "alignment": alignment,
        "balance": float(balance),
        "width": int(w),
        "height": int(h),
    }


CONFIDENCE_THRESHOLD = 0.15

def classify_scene(image: Image.Image) -> tuple[str, float]:
    """Classify scene using CLIP zero-shot classification."""
    model, processor = _load_clip()
    labels = [s[0] for s in SCENE_LABELS]
    prompts = [s[1] for s in SCENE_LABELS]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    top_prob = probs.max().item()
    if top_prob < CONFIDENCE_THRESHOLD:
        return "Unknown", top_prob
    return labels[probs.argmax().item()], top_prob


def build_blueprint(features: dict) -> dict:
    """Build an AR blueprint description from features."""
    h = features["height"]
    w = features["width"]
    orientation = "portrait" if h >= w else "landscape"

    notes = []
    if features["alignment"] < 0.7:
        notes.append("tilted horizon — consider straightening")
    if features["balance"] < 0.6:
        notes.append("unbalanced composition — subject may be off-centre")
    if features["rule_of_thirds"] > 0.5:
        notes.append("strong rule-of-thirds alignment")

    return {
        "orientation": orientation,
        "grid": "rule_of_thirds",
        "notes": notes,
    }


def assess_lighting(features: dict) -> dict:
    """Assess lighting quality and tone from image features."""
    brightness = features["brightness"]
    color_ratio = features["color_ratio"]

    # Quality
    if 100 < brightness < 200:
        quality = "Good"
    elif 60 < brightness <= 100 or 200 <= brightness < 230:
        quality = "Fair"
    else:
        quality = "Poor"

    # Tone (blue/red ratio: high = cool, low = warm)
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
    tip = tips.get((quality, tone), "Adjust your position for better light")

    return {"quality": quality, "tone": tone, "tip": tip}


def assess_composition(features: dict) -> dict:
    """Assess composition quality from image features."""
    sharpness = features["sharpness"]
    alignment = features["alignment"]
    balance = features["balance"]

    if sharpness > 0.1:
        focus = "Sharp"
    elif sharpness > 0.05:
        focus = "Soft"
    else:
        focus = "Blurry"

    if alignment > 0.8:
        horizon = "Level"
    elif alignment > 0.6:
        horizon = "Slightly tilted"
    else:
        horizon = "Tilted"

    if balance > 0.8:
        symmetry = "Balanced"
    elif balance > 0.6:
        symmetry = "Slightly off"
    else:
        symmetry = "Unbalanced"

    return {"focus": focus, "horizon": horizon, "balance": symmetry}


SCENE_CATEGORIES = {
    "indoor_dining":  ["Café or Restaurant"],
    "night_out":      ["Bar or Nightclub", "Night Market"],
    "indoor_public":  ["Shopping Mall", "Convenience Store", "Museum or Gallery", "Hotel Lobby"],
    "work":           ["Office or Workspace"],
    "outdoor_urban":  ["Street", "Rooftop or Balcony"],
    "outdoor_nature": ["Park or Garden", "Beach", "Mountain or Hiking Trail"],
    "cultural":       ["Temple or Historic Site"],
    "active":         ["Gym or Sports Venue"],
    "transit":        ["Airport or Train Station"],
}

def _get_category(label: str) -> str:
    for cat, labels in SCENE_CATEGORIES.items():
        if label in labels:
            return cat
    return "default"

CATEGORY_POSE_TIPS = {
    "indoor_dining": [
        "Sit side-on, look towards the window",
        "Hold a drink with both hands, smile looking down",
        "Face away from camera, look into the distance",
    ],
    "night_out": [
        "Lean on the bar or stall counter, look over shoulder",
        "Hold food or drink up, smile candid",
        "Stand in the crowd, face half-turned towards the light",
    ],
    "indoor_public": [
        "Stand centred, arms relaxed, chin slightly up",
        "Lean against the wall with one shoulder",
        "Walk towards the camera with a natural stride",
    ],
    "work": [
        "Sit at desk, look up from work naturally",
        "Stand by a window with laptop or phone in hand",
        "Lean on a glass wall or partition, arms loosely crossed",
    ],
    "outdoor_urban": [
        "Look over your shoulder into the light",
        "Hands in pockets, gaze slightly off-camera",
        "Stand along leading lines for depth",
    ],
    "outdoor_nature": [
        "Walk naturally along a path, mid-stride",
        "Sit on the ground, knees up, looking away",
        "Stand facing the horizon, arms relaxed",
    ],
    "cultural": [
        "Stand respectfully beside architecture, look up",
        "Sit on steps, legs crossed, natural expression",
        "Frame yourself in a doorway or archway",
    ],
    "active": [
        "Strike a confident stance with equipment in frame",
        "Action shot: mid-exercise with good form visible",
        "Lean on equipment casually, look off-camera",
    ],
    "transit": [
        "Walk with luggage, look ahead confidently",
        "Sit with bag on lap, look out the window",
        "Stand under the departures board, look up",
    ],
}

DEFAULT_POSE_TIPS = [
    "Face the light source at a slight angle",
    "Keep your posture relaxed and natural",
]

def generate_pose_tips(image_path: str, category: str) -> list[str]:
    try:
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        response = _get_openai_client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": (
                        "You are a photography coach helping someone take a 打卡 photo. "
                        "Analyse this scene and give 3 specific pose tips based on what "
                        "you can see — the lighting direction, available space, background "
                        "elements, and any notable features like windows, walls, or furniture. "
                        "Be specific to this exact scene, not generic. "
                        "Return exactly a JSON array of 3 short strings, no other text."
                    )}
                ]
            }],
            max_tokens=300,
        )

        tips = json.loads(response.choices[0].message.content.strip())
        if isinstance(tips, list) and len(tips) >= 3:
            return [str(t) for t in tips[:3]]
    except Exception:
        pass

    return CATEGORY_POSE_TIPS.get(category, DEFAULT_POSE_TIPS)

CATEGORY_HASHTAGS = {
    "indoor_dining":  ["#cafe", "#foodie", "#citywalk"],
    "night_out":      ["#nightout", "#nightmarket", "#vibes"],
    "indoor_public":  ["#shopping", "#architecture", "#citywalk"],
    "work":           ["#worklife", "#office", "#hustle"],
    "outdoor_urban":  ["#streetphoto", "#cityvibes", "#citywalk"],
    "outdoor_nature": ["#nature", "#outdoors", "#explore"],
    "cultural":       ["#heritage", "#architecture", "#travel"],
    "active":         ["#fitness", "#active", "#gym"],
    "transit":        ["#travel", "#onthemove", "#departure"],
}

CATEGORY_FILTERS = {
    "indoor_dining":  "Warm film",
    "night_out":      "Warm vintage",
    "indoor_public":  "Cool minimal",
    "work":           "Cool minimal",
    "outdoor_urban":  "Desaturated urban",
    "outdoor_nature": "Soft natural",
    "cultural":       "Warm vintage",
    "active":         "High contrast",
    "transit":        "Desaturated urban",
}

def analyze_scene(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    features = extract_features(image_path)
    scene_type, scene_confidence = classify_scene(image)
    category = _get_category(scene_type)

    return {
        "scene_type": scene_type,
        "scene_confidence": round(scene_confidence, 3),
        "blueprint": build_blueprint(features),
        "lighting": assess_lighting(features),
        "composition": assess_composition(features),
        "pose_tips": generate_pose_tips(image_path, category),
        "hashtags": CATEGORY_HASHTAGS.get(category, ["#citywalk"]),
        "filter": CATEGORY_FILTERS.get(category, "Soft natural"),
    }
