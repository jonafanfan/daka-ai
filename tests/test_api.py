"""Tests for the /analyze endpoint's guards.

analyze_scene is monkeypatched throughout — these cover the HTTP contract (limits, status codes,
error shapes), not the vision pipeline, and must never reach the OpenAI API.
"""
import io

import pytest
from fastapi.testclient import TestClient

import api_server
import scene_analysis
from scene_analysis import InappropriateImageError

JPEG = "image/jpeg"


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """The limiter is module-global; without this, tests leak into each other."""
    api_server._hits.clear()
    yield
    api_server._hits.clear()


@pytest.fixture
def client():
    return TestClient(api_server.app)


@pytest.fixture
def ok_analysis(monkeypatch):
    result = {"scene_type": "Cafe", "lighting": {"quality": "Good"}, "blurry": False}
    monkeypatch.setattr(api_server, "analyze_scene", lambda path, lang="en": result)
    return result


@pytest.fixture
def real_jpeg():
    """Bytes that cv2 and PIL can genuinely decode.

    Most tests here never reach the engine, so fake bytes are enough. /measure does reach it: it
    runs the whole OpenCV pipeline, and undecodable bytes come back as a 400 that would look like
    a passing test.
    """
    import cv2
    import numpy as np

    h, w, cell = 240, 320, 16
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            img[y:y + cell, x:x + cell] = 225 if ((y // cell) + (x // cell)) % 2 == 0 else 30
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def upload(content=b"\xff\xd8\xff\xe0fake jpeg bytes", content_type=JPEG, name="photo.jpg"):
    return {"file": (name, io.BytesIO(content), content_type)}


# ── health ──

def test_health_is_open(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── the language the caller asks for ──

def langs_seen(client, monkeypatch, **post):
    """What the engine was told to write in, for a given request."""
    seen = []
    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p, lang="en": seen.append(lang) or {"lighting": {}}
    )
    client.post("/analyze", files=upload(), **post)
    return seen


def test_the_requested_language_reaches_the_engine(client, monkeypatch):
    assert langs_seen(client, monkeypatch, data={"lang": "zh"}) == ["zh"]


def test_a_request_with_no_language_still_works(client, monkeypatch):
    """Cached copies of the old page send no lang at all, and a scan from one must not 422."""
    assert langs_seen(client, monkeypatch) == ["en"]


def test_a_junk_language_is_passed_through_rather_than_rejected(client, monkeypatch):
    """Deliberate: the engine falls back on anything it does not recognise, and refusing the
    request would throw away a scan over a field that only chooses wording."""
    assert langs_seen(client, monkeypatch, data={"lang": "../etc/passwd"}) == ["../etc/passwd"]


# ── happy path ──

def test_valid_upload_returns_the_analysis(client, ok_analysis):
    response = client.post("/analyze", files=upload())
    assert response.status_code == 200
    assert response.json() == ok_analysis


def test_temp_file_is_cleaned_up(client, monkeypatch, tmp_path):
    seen = {}

    def capture(path, lang="en"):
        seen["path"] = path
        assert __import__("os").path.exists(path), "engine must receive a real file"
        return {"lighting": {}}

    monkeypatch.setattr(api_server, "analyze_scene", capture)
    client.post("/analyze", files=upload())
    import os
    assert not os.path.exists(seen["path"]), "temp file must be removed after the request"


def test_temp_file_is_cleaned_up_even_on_failure(client, monkeypatch):
    seen = {}

    def boom(path, lang="en"):
        seen["path"] = path
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(api_server, "analyze_scene", boom)
    assert client.post("/analyze", files=upload()).status_code == 500
    import os
    assert not os.path.exists(seen["path"])


# ── the split scan ──
#
# /measure is the half the user waits on, so the thing worth pinning is what it does NOT do: no
# moderation call, no vision call, nothing that leaves the machine. /describe is where both live.

def test_measure_makes_no_openai_calls(client, monkeypatch, real_jpeg):
    """The whole point of the split. If moderation or the model creep back in here, the marker is
    behind a network round trip again and the user is holding their arm up for it."""
    calls = []
    monkeypatch.setattr(scene_analysis, "_moderate_image", lambda b64: calls.append("moderate") or True)
    monkeypatch.setattr(scene_analysis, "_analyze_with_gpt", lambda *a, **k: calls.append("vision") or {})
    response = client.post("/measure", files=upload(content=real_jpeg))
    assert response.status_code == 200, response.json()
    assert calls == [], f"/measure called OpenAI: {calls}"


def test_measure_returns_the_marker_and_the_gates(client, real_jpeg):
    body = client.post("/measure", files=upload(content=real_jpeg)).json()
    assert set(body) == {
        "blueprint", "lighting", "blurry", "blur_var", "edge_sharpness",
        "composition", "placement", "camera_tilt",
    }
    assert body["placement"]["x"] in (0.333, 0.667)


def test_describe_moderates_before_the_model(client, monkeypatch, real_jpeg):
    """Moderation follows the image, and /describe is the only route that sends it anywhere."""
    order = []
    monkeypatch.setattr(scene_analysis, "_moderate_image", lambda b64: order.append("moderate") or True)
    monkeypatch.setattr(scene_analysis, "_analyze_with_gpt", lambda *a, **k: order.append("vision") or {})
    client.post("/describe", files=upload(content=real_jpeg), data={"side": "right"})
    assert order == ["moderate", "vision"]


def test_describe_refuses_a_flagged_image_without_calling_the_model(client, monkeypatch, real_jpeg):
    called = []
    monkeypatch.setattr(scene_analysis, "_moderate_image", lambda b64: False)
    monkeypatch.setattr(scene_analysis, "_analyze_with_gpt", lambda *a, **k: called.append(1) or {})
    response = client.post("/describe", files=upload(content=real_jpeg))
    assert response.status_code == 400
    assert called == [], "a flagged image still reached the vision model"


@pytest.mark.parametrize("side, expected_x", [("left", 0.333), ("right", 0.667)])
def test_describe_passes_the_side_through_to_the_prompt(client, monkeypatch, real_jpeg, side, expected_x):
    """The model is told which side the geometry picked so its sentence cannot contradict the
    marker. In one call that was free; split apart, the client has to hand it back."""
    seen = {}
    monkeypatch.setattr(scene_analysis, "_moderate_image", lambda b64: True)
    monkeypatch.setattr(
        scene_analysis, "_analyze_with_gpt",
        lambda b64, placement=None, lang="en": seen.update(placement or {}) or {},
    )
    client.post("/describe", files=upload(content=real_jpeg), data={"side": side})
    assert seen.get("x") == expected_x


def test_an_unknown_side_is_ignored_rather_than_guessed(client, monkeypatch, real_jpeg):
    """A wrong side is worse than none: the prompt has its own default, and a contradicted marker
    reads as the app being confused."""
    seen = []
    monkeypatch.setattr(scene_analysis, "_moderate_image", lambda b64: True)
    monkeypatch.setattr(
        scene_analysis, "_analyze_with_gpt",
        lambda b64, placement=None, lang="en": seen.append(placement) or {},
    )
    client.post("/describe", files=upload(content=real_jpeg), data={"side": "sideways"})
    assert seen == [None]


def test_the_two_halves_cover_exactly_what_analyze_returns(client, monkeypatch, real_jpeg):
    """Nothing gained, nothing dropped. A field that exists in neither half would silently vanish
    for the split client while still working for the old one."""
    monkeypatch.setattr(scene_analysis, "_moderate_image", lambda b64: True)
    monkeypatch.setattr(
        scene_analysis, "_analyze_with_gpt",
        lambda *a, **k: {"scene_type": "Cafe", "filter": "Vivid", "hashtags": ["#a"],
                         "placement_hint": "Stand by the wall"},
    )
    whole = set(client.post("/analyze", files=upload(content=real_jpeg)).json())
    measured = set(client.post("/measure", files=upload(content=real_jpeg)).json())
    described = set(client.post("/describe", files=upload(content=real_jpeg)).json())
    assert measured | described == whole, (
        f"missing from the split: {whole - (measured | described)}; "
        f"invented by it: {(measured | described) - whole}"
    )
    assert not (measured & described), f"both halves claim {measured & described}"


# ── content type ──

@pytest.mark.parametrize("content_type", ["text/plain", "application/pdf", "application/json"])
def test_non_image_content_type_is_rejected(client, ok_analysis, content_type):
    response = client.post("/analyze", files=upload(content_type=content_type))
    assert response.status_code == 415
    assert "error" in response.json()


@pytest.mark.parametrize("content_type", ["image/jpeg", "image/png", "image/webp", "image/heic"])
def test_supported_image_types_are_accepted(client, ok_analysis, content_type):
    assert client.post("/analyze", files=upload(content_type=content_type)).status_code == 200


# ── size cap ──

def test_oversized_upload_is_rejected(client, ok_analysis):
    too_big = b"x" * (api_server.MAX_UPLOAD_BYTES + 1)
    response = client.post("/analyze", files=upload(content=too_big))
    assert response.status_code == 413
    assert "8 MB" in response.json()["error"]


def test_upload_at_the_cap_is_accepted(client, ok_analysis):
    at_cap = b"x" * api_server.MAX_UPLOAD_BYTES
    assert client.post("/analyze", files=upload(content=at_cap)).status_code == 200


def test_empty_upload_is_rejected(client, ok_analysis):
    response = client.post("/analyze", files=upload(content=b""))
    assert response.status_code == 400
    assert "empty" in response.json()["error"].lower()


# ── rate limit ──

def test_rate_limit_blocks_after_the_quota(client, ok_analysis):
    for _ in range(api_server.RATE_LIMIT_REQUESTS):
        assert client.post("/analyze", files=upload()).status_code == 200
    response = client.post("/analyze", files=upload())
    assert response.status_code == 429
    assert "error" in response.json()


def test_rate_limit_is_per_ip(client, ok_analysis):
    for _ in range(api_server.RATE_LIMIT_REQUESTS):
        client.post("/analyze", files=upload(), headers={"x-forwarded-for": "1.1.1.1"})
    assert client.post(
        "/analyze", files=upload(), headers={"x-forwarded-for": "1.1.1.1"}
    ).status_code == 429
    assert client.post(
        "/analyze", files=upload(), headers={"x-forwarded-for": "2.2.2.2"}
    ).status_code == 200


def test_rate_limit_precedes_the_engine(client, monkeypatch):
    """A throttled request must not cost an OpenAI call — that is the whole point."""
    calls = []
    monkeypatch.setattr(api_server, "analyze_scene", lambda p, lang="en": calls.append(p) or {"lighting": {}})
    for _ in range(api_server.RATE_LIMIT_REQUESTS + 5):
        client.post("/analyze", files=upload())
    assert len(calls) == api_server.RATE_LIMIT_REQUESTS


# ── error mapping ──

def test_moderation_rejection_maps_to_400(client, monkeypatch):
    def flagged(path, lang="en"):
        raise InappropriateImageError("nope")

    monkeypatch.setattr(api_server, "analyze_scene", flagged)
    response = client.post("/analyze", files=upload())
    assert response.status_code == 400
    assert response.json()["error"] == "Image not suitable for analysis"


def test_moderation_is_distinguished_from_an_unreadable_image(client, monkeypatch):
    """InappropriateImageError subclasses ValueError, so catch order is load-bearing.

    If the ValueError branch ran first, a moderation rejection would be reported to the user as
    "that image couldn't be read", which is both wrong and confusing.
    """
    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p, lang="en": (_ for _ in ()).throw(InappropriateImageError())
    )
    moderation = client.post("/analyze", files=upload()).json()["error"]

    api_server._hits.clear()
    monkeypatch.setattr(api_server, "analyze_scene", lambda p, lang="en": (_ for _ in ()).throw(ValueError()))
    unreadable = client.post("/analyze", files=upload()).json()["error"]

    assert moderation != unreadable
    assert moderation == "Image not suitable for analysis"


def test_undecodable_image_maps_to_400(client, monkeypatch):
    from PIL import UnidentifiedImageError

    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p, lang="en": (_ for _ in ()).throw(UnidentifiedImageError())
    )
    response = client.post("/analyze", files=upload())
    assert response.status_code == 400
    assert "couldn't be read" in response.json()["error"]


def test_unexpected_failure_maps_to_500(client, monkeypatch):
    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p, lang="en": (_ for _ in ()).throw(RuntimeError("boom"))
    )
    response = client.post("/analyze", files=upload())
    assert response.status_code == 500


def test_500_does_not_leak_internals(client, monkeypatch):
    """Regression: the old handler returned str(e) and type(e).__name__ to the client."""
    secret = "postgres://user:hunter2@db.internal:5432/prod"

    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p, lang="en": (_ for _ in ()).throw(RuntimeError(secret))
    )
    response = client.post("/analyze", files=upload())
    body = response.text
    assert secret not in body
    assert "hunter2" not in body
    assert "RuntimeError" not in body
    assert response.json() == {"error": "Analysis failed on the server — try again in a moment."}


def test_every_error_body_has_the_same_shape(client, monkeypatch):
    """The client reads data.error unconditionally, so the key must always be there."""
    monkeypatch.setattr(api_server, "analyze_scene", lambda p, lang="en": {"lighting": {}})
    cases = [
        client.post("/analyze", files=upload(content_type="text/plain")),
        client.post("/analyze", files=upload(content=b"")),
        client.post("/analyze", files=upload(content=b"x" * (api_server.MAX_UPLOAD_BYTES + 1))),
    ]
    for response in cases:
        assert response.status_code >= 400
        assert list(response.json()) == ["error"]
        assert isinstance(response.json()["error"], str) and response.json()["error"]


# ── CORS ──

def test_allowed_origin_gets_cors_headers(client, ok_analysis):
    origin = api_server.ALLOWED_ORIGINS[0]
    response = client.post("/analyze", files=upload(), headers={"Origin": origin})
    assert response.headers.get("access-control-allow-origin") == origin


def test_disallowed_origin_gets_no_cors_headers(client, ok_analysis):
    response = client.post(
        "/analyze", files=upload(), headers={"Origin": "https://evil.example.com"}
    )
    assert "access-control-allow-origin" not in response.headers


def test_the_shipped_frontends_are_allowed_by_default():
    """All three must work with no env var set in Render: the web frontend, and the native shell,
    which serves the same page from capacitor://localhost on iOS and http://localhost on Android.

    Pinned as an exact list rather than a membership check, because the failure that matters is an
    origin creeping IN. This endpoint is unauthenticated and every call spends money.
    """
    assert api_server.ALLOWED_ORIGINS == [
        "https://dakaba.pages.dev",
        "capacitor://localhost",
        "http://localhost",
    ], f"unexpected default origin set: {api_server.ALLOWED_ORIGINS}"


def test_a_default_origin_actually_reaches_cors(client, ok_analysis):
    """The list is only half of it: this is the header the browser actually checks."""
    response = client.post(
        "/analyze", files=upload(), headers={"origin": "https://dakaba.pages.dev"}
    )
    assert response.headers.get("access-control-allow-origin") == "https://dakaba.pages.dev"


def test_wildcard_origin_is_not_configured():
    """Regression: allow_origins was "*" before hardening."""
    assert "*" not in api_server.ALLOWED_ORIGINS
