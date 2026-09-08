"""Tests for the two guidance signals: why the marker is where it is, and dead-space tilt.

Both exist so the app can explain itself rather than showing an unexplained dot. The placement
reason is derived from work the engine was already doing and throwing away; the dead-space check
reuses the saliency map placement has already built.
"""
import numpy as np
import pytest

from scene_analysis import PLACEMENT_REASONS, _compute_placement, _detect_dead_space
from conftest import LEFT_THIRD, RIGHT_THIRD, gray, saliency


def reason(g, s):
    return _compute_placement(g, s)["reason"]


# ── the reason names the signal that actually decided ────────────────────────

def test_backlight_veto_is_named_as_the_reason(flat_saliency):
    """The veto is a hard constraint, so it owns the explanation whatever else voted."""
    assert reason(gray(left=120, right=255), flat_saliency) == "backlight"
    assert reason(gray(left=255, right=120), flat_saliency) == "backlight"


def test_veto_beats_a_strong_opposing_vote_in_the_reason_too():
    """Saliency votes right twice; the blown-out right vetoes it. The veto must be credited."""
    interest_left = saliency(0.0, box=(0.0, 1.0, 0.0, 0.33))
    result = _compute_placement(gray(left=120, right=255), interest_left)
    assert result["x"] == LEFT_THIRD
    assert result["reason"] == "backlight"


def test_light_is_named_when_only_light_votes(flat_saliency):
    """flat_saliency is deliberately unreliable, so light is the only signal in play."""
    assert reason(gray(left=210, right=90), flat_saliency) == "light"
    assert reason(gray(left=90, right=210), flat_saliency) == "light"


def test_a_saliency_signal_is_named_when_only_saliency_votes(flat_gray):
    """Uniform grey casts no light vote, so the reason must come from the saliency pair."""
    interest_left = saliency(0.0, box=(0.0, 1.0, 0.0, 0.33))
    assert reason(flat_gray, interest_left) in ("balance", "clean_background")


def test_default_when_nothing_votes(flat_saliency):
    """No usable saliency, no light asymmetry, no blowout — the tie-break decides, so no signal
    can honestly be credited."""
    assert reason(gray(left=128, right=130), flat_saliency) == "default"


def test_a_losing_signal_is_never_credited():
    """The reason must be a signal that pointed the way we actually went.

    Saliency (2.0 combined) sends the subject right; light (1.2) wants left and loses. Crediting
    light would be actively misleading — the marker would explain itself with the one reason that
    argued against where it is.
    """
    interest_left = saliency(0.0, box=(0.0, 1.0, 0.0, 0.33))   # -> right
    bright_right = gray(left=90, right=210)                     # -> left, and loses
    result = _compute_placement(bright_right, interest_left)
    assert result["x"] == RIGHT_THIRD
    assert result["reason"] != "light"


# ── the reason is always present and displayable ─────────────────────────────

def test_every_reason_code_has_text():
    for code, text in PLACEMENT_REASONS.items():
        assert text and isinstance(text, str), code


def test_reason_text_is_short_enough_to_draw_under_the_marker():
    """These are rendered on the viewfinder on a phone. Long strings run off the frame."""
    for code, text in PLACEMENT_REASONS.items():
        assert len(text) <= 34, f"{code} is {len(text)} chars: {text!r}"


def test_reason_and_text_always_agree():
    rng = np.random.default_rng(7)
    for _ in range(40):
        g = gray(rng.uniform(20, 250), rng.uniform(0, 255), rng.uniform(0, 255))
        s = saliency(rng.uniform(0, 0.5), (0, rng.uniform(0.2, 1.0), 0, rng.uniform(0.2, 1.0)))
        result = _compute_placement(g, s)
        assert result["reason"] in PLACEMENT_REASONS
        assert result["reason_text"] == PLACEMENT_REASONS[result["reason"]]


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4, 3), dtype=np.float32), None, np.zeros((0, 0), dtype=np.float32)],
    ids=["3d", "none", "empty"],
)
def test_fallback_still_carries_a_reason(flat_gray, bad):
    result = _compute_placement(flat_gray, bad)
    assert result["reason"] == "default"
    assert result["reason_text"] == PLACEMENT_REASONS["default"]


# ── dead space: aim off the empty third ──────────────────────────────────────

def band(top=0.0, mid=0.0, bottom=0.0, h=120, w=160):
    """Saliency map with independently set thirds."""
    s = np.zeros((h, w), dtype=np.float32)
    s[: h // 3] = top
    s[h // 3 : 2 * h // 3] = mid
    s[2 * h // 3 :] = bottom
    return s


def test_blank_top_says_aim_lower():
    """Empty ceiling or featureless sky above -> tilt down to cut it out."""
    result = _detect_dead_space(band(top=0.02, mid=0.9, bottom=0.8))
    assert result["direction"] == "down"
    assert "lower" in result["reason"]


def test_blank_bottom_says_aim_higher():
    result = _detect_dead_space(band(top=0.8, mid=0.9, bottom=0.02))
    assert result["direction"] == "up"
    assert "higher" in result["reason"]


def test_an_evenly_interesting_frame_is_left_alone():
    assert _detect_dead_space(band(top=0.7, mid=0.8, bottom=0.75))["direction"] == "ok"


def test_a_mildly_quieter_third_is_not_dead():
    """The threshold must not fire on ordinary variation, or it nags on every scan.

    Top carries 60% of the rest's interest — quieter, but not dead. Only well under half counts.
    """
    assert _detect_dead_space(band(top=0.48, mid=0.8, bottom=0.8))["direction"] == "ok"


def test_a_flat_map_is_not_judged():
    """Same reliability gate as placement: on a uniform map the thirds are all noise."""
    assert _detect_dead_space(np.full((120, 160), 0.5, dtype=np.float32))["direction"] == "ok"


def test_a_near_flat_map_is_not_judged_either():
    """The case a perfectly uniform map does not actually test.

    On a uniform map the ratio comparison already returns "ok" on its own, so it proves nothing
    about the reliability gate. Here the bands differ by a wide *ratio* (0.001 vs 0.01) but a
    trivial absolute amount, so the ratio test alone would confidently call the top dead. Only the
    gate stops it, and without one the app would nag about dead space on featureless scenes.
    """
    faint = band(top=0.001, mid=0.01, bottom=0.01)
    assert faint.std() < 0.010, "must be under the gate for this test to mean anything"
    assert _detect_dead_space(faint)["direction"] == "ok"


def test_the_deader_of_two_quiet_bands_wins():
    """When both outer thirds are quiet, the emptier one decides which way to aim.

    Top (0.02) is quiet enough to trip the threshold on its own, but the bottom (0.01) is emptier
    still — so the advice must be to aim up and cut the floor, not down. Without the comparison
    against the opposite band, whichever branch is written first would always win.
    """
    result = _detect_dead_space(band(top=0.02, mid=0.9, bottom=0.01))
    assert result["direction"] == "up"
    assert "higher" in result["reason"]


def test_ok_carries_no_reason_text():
    """The client shows reason verbatim, so 'ok' must not produce a cue."""
    assert _detect_dead_space(band(top=0.7, mid=0.8, bottom=0.75))["reason"] == ""


def test_only_one_direction_can_win():
    """Both thirds quiet relative to a busy middle: pick the deader one, never both."""
    result = _detect_dead_space(band(top=0.02, mid=0.9, bottom=0.10))
    assert result["direction"] == "down", "top is emptier, so the top is what to cut"


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros((0, 0), dtype=np.float32),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((5,), dtype=np.float32),
        np.zeros((2, 10), dtype=np.float32),      # too short to have thirds
    ],
    ids=["none", "empty", "3d", "1d", "two-rows"],
)
def test_malformed_input_degrades_to_ok(bad):
    assert _detect_dead_space(bad) == {"direction": "ok", "reason": ""}


def test_direction_is_always_from_the_closed_set():
    rng = np.random.default_rng(11)
    for _ in range(40):
        s = band(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1))
        assert _detect_dead_space(s)["direction"] in ("up", "down", "ok")
