"""Tests for the /analyze endpoint's guards.

analyze_scene is monkeypatched throughout — these cover the HTTP contract (limits, status codes,
error shapes), not the vision pipeline, and must never reach the OpenAI API.
"""
import io

import pytest
from fastapi.testclient import TestClient

import api_server
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
    monkeypatch.setattr(api_server, "analyze_scene", lambda path: result)
    return result


def upload(content=b"\xff\xd8\xff\xe0fake jpeg bytes", content_type=JPEG, name="photo.jpg"):
    return {"file": (name, io.BytesIO(content), content_type)}


# ── health ──

def test_health_is_open(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── happy path ──

def test_valid_upload_returns_the_analysis(client, ok_analysis):
    response = client.post("/analyze", files=upload())
    assert response.status_code == 200
    assert response.json() == ok_analysis


def test_temp_file_is_cleaned_up(client, monkeypatch, tmp_path):
    seen = {}

    def capture(path):
        seen["path"] = path
        assert __import__("os").path.exists(path), "engine must receive a real file"
        return {"lighting": {}}

    monkeypatch.setattr(api_server, "analyze_scene", capture)
    client.post("/analyze", files=upload())
    import os
    assert not os.path.exists(seen["path"]), "temp file must be removed after the request"


def test_temp_file_is_cleaned_up_even_on_failure(client, monkeypatch):
    seen = {}

    def boom(path):
        seen["path"] = path
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(api_server, "analyze_scene", boom)
    assert client.post("/analyze", files=upload()).status_code == 500
    import os
    assert not os.path.exists(seen["path"])


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
    monkeypatch.setattr(api_server, "analyze_scene", lambda p: calls.append(p) or {"lighting": {}})
    for _ in range(api_server.RATE_LIMIT_REQUESTS + 5):
        client.post("/analyze", files=upload())
    assert len(calls) == api_server.RATE_LIMIT_REQUESTS


# ── error mapping ──

def test_moderation_rejection_maps_to_400(client, monkeypatch):
    def flagged(path):
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
        api_server, "analyze_scene", lambda p: (_ for _ in ()).throw(InappropriateImageError())
    )
    moderation = client.post("/analyze", files=upload()).json()["error"]

    api_server._hits.clear()
    monkeypatch.setattr(api_server, "analyze_scene", lambda p: (_ for _ in ()).throw(ValueError()))
    unreadable = client.post("/analyze", files=upload()).json()["error"]

    assert moderation != unreadable
    assert moderation == "Image not suitable for analysis"


def test_undecodable_image_maps_to_400(client, monkeypatch):
    from PIL import UnidentifiedImageError

    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p: (_ for _ in ()).throw(UnidentifiedImageError())
    )
    response = client.post("/analyze", files=upload())
    assert response.status_code == 400
    assert "couldn't be read" in response.json()["error"]


def test_unexpected_failure_maps_to_500(client, monkeypatch):
    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    response = client.post("/analyze", files=upload())
    assert response.status_code == 500


def test_500_does_not_leak_internals(client, monkeypatch):
    """Regression: the old handler returned str(e) and type(e).__name__ to the client."""
    secret = "postgres://user:hunter2@db.internal:5432/prod"

    monkeypatch.setattr(
        api_server, "analyze_scene", lambda p: (_ for _ in ()).throw(RuntimeError(secret))
    )
    response = client.post("/analyze", files=upload())
    body = response.text
    assert secret not in body
    assert "hunter2" not in body
    assert "RuntimeError" not in body
    assert response.json() == {"error": "Analysis failed on the server — try again in a moment."}


def test_every_error_body_has_the_same_shape(client, monkeypatch):
    """The client reads data.error unconditionally, so the key must always be there."""
    monkeypatch.setattr(api_server, "analyze_scene", lambda p: {"lighting": {}})
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


def test_netlify_production_origin_is_allowed_by_default():
    """The deployed frontend must work with no env var set in Render."""
    assert "https://dakaba.netlify.app" in api_server.ALLOWED_ORIGINS


def test_wildcard_origin_is_not_configured():
    """Regression: allow_origins was "*" before hardening."""
    assert "*" not in api_server.ALLOWED_ORIGINS
