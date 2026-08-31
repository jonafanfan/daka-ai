"""Tests for the two OpenAI-facing functions and the degradation paths behind them.

The client is faked throughout — nothing here touches the network, and the autouse fixture makes
constructing a real client an outright test failure rather than a silent API call.

What matters here is failure behaviour. The engine's stated design is that a bad completion
degrades to safe defaults instead of failing the scan, because the OpenCV half (placement,
framing, lighting, blur) is the part carrying the product; the model only supplies the scene name,
hashtags and filter choice. These tests pin down where that promise holds and where it does not.
"""
import base64
import io
import json
import types

import cv2
import numpy as np
import pytest
from PIL import Image

import scene_analysis
from scene_analysis import (
    VALID_FILTERS,
    InappropriateImageError,
    _analyze_with_gpt,
    _encode_image,
    _moderate_image,
    analyze_scene,
)


# ── fake client ───────────────────────────────────────────────────────────────

class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeModerations:
    def __init__(self, flagged=False, error=None):
        self.flagged, self.error, self.calls = flagged, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return types.SimpleNamespace(results=[types.SimpleNamespace(flagged=self.flagged)])


class FakeClient:
    def __init__(self, completion=None, completion_error=None, flagged=False, moderation_error=None):
        self.moderations = FakeModerations(flagged, moderation_error)
        self.chat = types.SimpleNamespace(completions=FakeCompletions(completion, completion_error))


NO_MESSAGE = object()


def completion(content):
    """Shape a chat-completion response: response.choices[0].message.content."""
    message = None if content is NO_MESSAGE else types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


NO_CHOICES = types.SimpleNamespace(choices=[])


class RealClientForbidden(BaseException):
    """Deliberately a BaseException, not an Exception.

    Both engine functions wrap their API call in `except Exception` to degrade, which would happily
    swallow an AssertionError raised here — a test that forgot to install a fake would then pass
    quietly with {} instead of failing. Inheriting BaseException makes the guard punch through the
    degradation handlers. (Verified: with an Exception subclass, the escape is silent.)
    """


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    """Reset the cached client and make constructing a real one fail loudly.

    _get_openai_client caches into a module global, so without the reset a fake would leak into
    later tests — and without the OpenAI guard, a test that forgets to install a fake would try to
    reach the real API.
    """
    monkeypatch.setattr(scene_analysis, "_openai_client", None)

    def forbidden(*args, **kwargs):
        raise RealClientForbidden("a real OpenAI client was constructed — install a fake")

    monkeypatch.setattr(scene_analysis, "OpenAI", forbidden)
    yield
    scene_analysis._openai_client = None


def test_the_no_real_client_guard_actually_fires():
    """Meta-test: proves the safety net above survives the degradation handlers.

    Without this, a regression that made the guard swallowable would go unnoticed — and every
    subsequent test could be silently exercising the fallback path instead of what it claims to.
    """
    with pytest.raises(RealClientForbidden):
        _analyze_with_gpt("Zm9v")     # no fake installed on purpose

    with pytest.raises(RealClientForbidden):
        _moderate_image("Zm9v")


def install(client, monkeypatch):
    monkeypatch.setattr(scene_analysis, "_openai_client", client)
    return client


def prompt_text(client):
    return client.chat.completions.calls[0]["messages"][0]["content"][0]["text"]


def decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ── _moderate_image ───────────────────────────────────────────────────────────

def test_clean_image_is_safe(monkeypatch):
    install(FakeClient(flagged=False), monkeypatch)
    assert _moderate_image("Zm9v") is True


def test_flagged_image_is_unsafe(monkeypatch):
    install(FakeClient(flagged=True), monkeypatch)
    assert _moderate_image("Zm9v") is False


def test_moderation_fails_open(monkeypatch):
    """Deliberate, and a bypass worth stating out loud.

    If the moderation call itself errors, the image is treated as safe. The alternative — refusing
    every scan during an OpenAI incident — was judged worse. Also documented in CONTRACT.md.
    """
    install(FakeClient(moderation_error=RuntimeError("service unavailable")), monkeypatch)
    assert _moderate_image("Zm9v") is True


def test_moderation_request_shape(monkeypatch):
    client = install(FakeClient(), monkeypatch)
    _moderate_image("QUJD")
    (call,) = client.moderations.calls
    assert call["model"] == "omni-moderation-latest"
    assert call["input"][0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


# ── _analyze_with_gpt: happy path and the request it sends ────────────────────

def test_valid_json_is_parsed(monkeypatch):
    payload = {"scene_type": "Cafe", "filter": "Vivid Warm", "hashtags": ["#a", "#b", "#c"]}
    install(FakeClient(completion=completion(json.dumps(payload))), monkeypatch)
    assert _analyze_with_gpt("Zm9v") == payload


def test_completion_request_shape(monkeypatch):
    client = install(FakeClient(completion=completion("{}")), monkeypatch)
    _analyze_with_gpt("QUJD")
    (call,) = client.chat.completions.calls
    assert call["response_format"] == {"type": "json_object"}
    assert call["max_completion_tokens"] == 500
    content = call["messages"][0]["content"]
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_prompt_asks_for_exactly_the_three_live_fields(monkeypatch):
    """Regression on the pose_tips removal — it must not creep back into the prompt."""
    client = install(FakeClient(completion=completion("{}")), monkeypatch)
    _analyze_with_gpt("Zm9v")
    prompt = prompt_text(client)
    for field in ("scene_type", "filter", "hashtags"):
        assert field in prompt
    assert "pose_tips" not in prompt, "pose_tips was removed in contract 0.10"


@pytest.mark.parametrize("filter_name", VALID_FILTERS)
def test_prompt_offers_every_valid_filter(monkeypatch, filter_name):
    """The engine validates against VALID_FILTERS, so the prompt must offer the same set."""
    client = install(FakeClient(completion=completion("{}")), monkeypatch)
    _analyze_with_gpt("Zm9v")
    assert filter_name in prompt_text(client)


# ── _analyze_with_gpt: degradation ───────────────────────────────────────────

@pytest.mark.parametrize(
    "content, label",
    [
        (None, "refused / content-filtered"),
        ("", "empty string"),
        ("   ", "whitespace only"),
        ("{'scene_type': 'Cafe'}", "single quotes, not JSON"),
        ('{"scene_type": "Cafe"', "truncated mid-object"),
        ("Here is the JSON: {}", "prose wrapper"),
    ],
)
def test_bad_completions_degrade_to_empty(monkeypatch, content, label):
    """Note: the `if not content: return {}` guard upstream of the parse is redundant.

    Mutation testing showed removing it changes nothing, because the fall-through is already
    covered: json.loads(None) raises TypeError and json.loads("") raises JSONDecodeError, and both
    are in the parse guard's except clause. Harmless belt-and-braces — recorded so nobody assumes
    it is load-bearing, and so the two None-ish cases above stay covered either way.
    """
    install(FakeClient(completion=completion(content)), monkeypatch)
    assert _analyze_with_gpt("Zm9v") == {}, label


def test_no_choices_degrades(monkeypatch):
    install(FakeClient(completion=NO_CHOICES), monkeypatch)
    assert _analyze_with_gpt("Zm9v") == {}


def test_missing_message_degrades(monkeypatch):
    """choices[0].message is None — seen with some refusal shapes."""
    install(FakeClient(completion=completion(NO_MESSAGE)), monkeypatch)
    assert _analyze_with_gpt("Zm9v") == {}


# ── API-level failures degrade too ──────────────────────────────────────────

@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("upstream timeout"),
        ConnectionError("connection reset"),
        ValueError("invalid api key"),
        TimeoutError("deadline exceeded"),
    ],
    ids=["runtime", "connection", "value", "timeout"],
)
def test_api_errors_degrade_instead_of_raising(monkeypatch, error):
    """A dead API must cost the scene label, not the whole scan.

    The OpenCV features are already computed by the time this runs, so raising here would discard
    work that succeeded and break placement and framing — neither of which needs the model.
    """
    install(FakeClient(completion_error=error), monkeypatch)
    assert _analyze_with_gpt("Zm9v") == {}


def test_api_failure_is_logged(monkeypatch, caplog):
    """Degrading silently would make an outage indistinguishable from bad model output."""
    install(FakeClient(completion_error=RuntimeError("upstream timeout")), monkeypatch)
    with caplog.at_level("WARNING", logger="daka.engine"):
        _analyze_with_gpt("Zm9v")
    assert any("vision call failed" in r.message for r in caplog.records)


def test_moderation_failure_is_logged(monkeypatch, caplog):
    """Fail-open is a safety bypass, so it must leave a trace."""
    install(FakeClient(moderation_error=RuntimeError("service unavailable")), monkeypatch)
    with caplog.at_level("WARNING", logger="daka.engine"):
        assert _moderate_image("Zm9v") is True
    assert any("moderation call failed" in r.message for r in caplog.records)


def test_api_error_still_yields_a_usable_scan(monkeypatch, scene_image):
    """End to end: the half of the product that does not need the model survives an outage."""
    install(FakeClient(completion_error=RuntimeError("upstream timeout")), monkeypatch)
    result = analyze_scene(scene_image)
    assert result["scene_type"] == "Unknown"
    assert result["hashtags"] == []
    assert result["filter"] == "Vivid"
    assert result["placement"]["x"] in (round(1 / 3, 3), round(2 / 3, 3))
    assert result["lighting"]["quality"] in ("Good", "Fair", "Poor")
    assert isinstance(result["blurry"], bool)


# ── non-object JSON degrades too ────────────────────────────────────────────

@pytest.mark.parametrize(
    "content", ["[1, 2, 3]", '"just a string"', "null", "42", "true"],
    ids=["array", "string", "null", "number", "bool"],
)
def test_non_object_json_degrades(monkeypatch, content):
    """Valid JSON that is not an object must not reach analyze_scene's gpt.get(...) calls.

    response_format={"type": "json_object"} should make this unreachable in practice, so the
    isinstance check is defensive depth — but without it the one function contracted never to 500
    hands back a list, and .get() raises AttributeError.
    """
    install(FakeClient(completion=completion(content)), monkeypatch)
    assert _analyze_with_gpt("Zm9v") == {}


def test_non_object_json_still_yields_a_usable_scan(monkeypatch, scene_image):
    """Previously raised AttributeError from analyze_scene; now falls back like any bad completion."""
    install(FakeClient(completion=completion("[1, 2, 3]")), monkeypatch)
    result = analyze_scene(scene_image)
    assert result["scene_type"] == "Unknown"
    assert result["filter"] == "Vivid"


# ── _encode_image ────────────────────────────────────────────────────────────

def test_encode_produces_decodable_jpeg(scene_image):
    raw = base64.b64decode(_encode_image(scene_image), validate=True)
    assert raw.startswith(b"\xff\xd8"), "JPEG SOI marker"
    assert decode(base64.b64encode(raw).decode()).format == "JPEG"


def test_encode_downscales_large_images(tmp_path):
    big = tmp_path / "big.png"
    assert cv2.imwrite(str(big), np.full((2000, 3000, 3), 128, dtype=np.uint8))
    decoded = decode(_encode_image(str(big)))
    assert max(decoded.size) == 768, "long edge should be capped at 768px"
    assert decoded.size == (768, 512), "aspect ratio must be preserved"


def test_encode_leaves_small_images_alone(tmp_path):
    small = tmp_path / "small.png"
    assert cv2.imwrite(str(small), np.full((100, 150, 3), 128, dtype=np.uint8))
    assert decode(_encode_image(str(small))).size == (150, 100)


def test_encode_converts_to_rgb(tmp_path):
    """A PNG with alpha must not blow up the JPEG save — hence the .convert("RGB")."""
    rgba = tmp_path / "rgba.png"
    Image.new("RGBA", (120, 90), (200, 100, 50, 128)).save(rgba)
    assert decode(_encode_image(str(rgba))).mode == "RGB"


# ── analyze_scene: how the two halves combine ────────────────────────────────

def test_flagged_image_raises(monkeypatch, scene_image):
    install(FakeClient(flagged=True), monkeypatch)
    with pytest.raises(InappropriateImageError):
        analyze_scene(scene_image)


def test_flagged_image_never_reaches_the_vision_call(monkeypatch, scene_image):
    """Moderation gates the expensive call — a rejected image must cost only the cheap one."""
    client = install(FakeClient(flagged=True), monkeypatch)
    with pytest.raises(InappropriateImageError):
        analyze_scene(scene_image)
    assert client.chat.completions.calls == [], "vision call should not have been made"


def test_empty_gpt_result_falls_back_to_safe_defaults(monkeypatch, scene_image):
    """The whole point of the degradation path: a useless completion still yields a usable scan."""
    install(FakeClient(completion=completion("{}")), monkeypatch)
    result = analyze_scene(scene_image)
    assert result["scene_type"] == "Unknown"
    assert result["hashtags"] == []
    assert result["filter"] == "Vivid"
    # and the OpenCV half — the part that actually carries the product — is intact
    assert result["placement"]["x"] in (round(1 / 3, 3), round(2 / 3, 3))
    assert result["lighting"]["quality"] in ("Good", "Fair", "Poor")
    assert isinstance(result["blurry"], bool)


@pytest.mark.parametrize("filter_name", VALID_FILTERS)
def test_valid_filters_pass_through(monkeypatch, scene_image, filter_name):
    install(FakeClient(completion=completion(json.dumps({"filter": filter_name}))), monkeypatch)
    assert analyze_scene(scene_image)["filter"] == filter_name


@pytest.mark.parametrize(
    "bogus", ["Sepia", "vivid", "VIVID WARM", "", None, 42, "Vivid Warm ", "Noir!"]
)
def test_invalid_filters_are_coerced(monkeypatch, scene_image, bogus):
    """Server-side validation matters: the client maps filter names to CSS strings, so an unknown
    value would silently render unfiltered. Note the check is exact-match — case and stray
    whitespace both fail it, which is why "vivid" and "Vivid Warm " appear here.
    """
    install(FakeClient(completion=completion(json.dumps({"filter": bogus}))), monkeypatch)
    assert analyze_scene(scene_image)["filter"] == "Vivid"


def test_response_keys_match_the_contract(monkeypatch, scene_image):
    install(FakeClient(completion=completion("{}")), monkeypatch)
    assert set(analyze_scene(scene_image)) == {
        "scene_type", "blueprint", "lighting", "blurry", "blur_var", "edge_sharpness",
        "composition", "placement", "hashtags", "filter",
    }


def test_pose_tips_is_not_in_the_response(monkeypatch, scene_image):
    """Regression on the 0.10 removal, including when the model volunteers the field anyway."""
    volunteered = json.dumps({"scene_type": "Cafe", "pose_tips": ["lean on the wall"]})
    install(FakeClient(completion=completion(volunteered)), monkeypatch)
    assert "pose_tips" not in analyze_scene(scene_image)


def test_unknown_model_fields_are_ignored(monkeypatch, scene_image):
    noise = json.dumps({"scene_type": "Cafe", "mood": "wistful", "iso": 400})
    install(FakeClient(completion=completion(noise)), monkeypatch)
    result = analyze_scene(scene_image)
    assert result["scene_type"] == "Cafe"
    assert "mood" not in result and "iso" not in result


def test_hashtags_pass_through_without_length_enforcement(monkeypatch, scene_image):
    """The prompt asks for exactly 3; the engine does not enforce it. The contract tells consumers
    to treat the length defensively — this documents that they genuinely have to."""
    install(FakeClient(completion=completion(json.dumps({"hashtags": ["#one"]}))), monkeypatch)
    assert analyze_scene(scene_image)["hashtags"] == ["#one"]


def test_moderation_runs_before_the_opencv_work(monkeypatch, tmp_path):
    """A flagged image should not pay for feature extraction either.

    Uses a path that is not a readable image: if extract_features ran, it would raise ValueError
    instead of InappropriateImageError. _encode_image runs first regardless, so the file still has
    to be openable by PIL — a 1x1 PNG is enough.
    """
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (1, 1), (10, 20, 30)).save(tiny)
    install(FakeClient(flagged=True), monkeypatch)
    with pytest.raises(InappropriateImageError):
        analyze_scene(str(tiny))
