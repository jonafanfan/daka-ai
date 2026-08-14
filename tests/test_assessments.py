"""Tests for the pure feature -> description functions."""
import itertools

import pytest

from scene_analysis import assess_composition, assess_lighting, build_blueprint


def lighting(brightness=150.0, color_ratio=0.8):
    return assess_lighting({"brightness": brightness, "color_ratio": color_ratio})


# ── tone: the regression that motivated this suite ──
#
# color_ratio is blue/red over a BGR image, so a warm scene scores LOW. This was inverted for
# months: golden-hour scenes were labelled Cool and told "Nice cool tones, use them for a clean
# aesthetic". Named by real-world scene so a future reader can tell which direction is correct
# without re-deriving the channel order.

@pytest.mark.parametrize(
    "scene, blue, red, expected",
    [
        ("golden-hour cafe",  120, 190, "Warm"),
        ("tungsten interior", 100, 175, "Warm"),
        ("candlelit room",     70, 200, "Warm"),
        ("overcast street",   150, 140, "Cool"),
        ("blue-hour dusk",    170, 120, "Cool"),
        ("shade under sky",   200, 150, "Cool"),
        ("neutral indoor",    130, 160, "Neutral"),
    ],
)
def test_tone_follows_the_red_blue_balance(scene, blue, red, expected):
    assert lighting(color_ratio=blue / red)["tone"] == expected, scene


def test_tone_is_monotonic_in_the_ratio():
    """The real inversion guard: more blue must never read warmer.

    Stated as monotonicity rather than by example, so it holds regardless of where the thresholds
    sit — any swap of the two comparisons breaks it immediately.
    """
    order = {"Warm": 0, "Neutral": 1, "Cool": 2}
    sequence = [order[lighting(color_ratio=r / 100)["tone"]] for r in range(1, 300)]
    assert sequence == sorted(sequence), "tone must not move warmer as the blue/red ratio rises"
    assert set(sequence) == {0, 1, 2}, "all three tones should be reachable"


def test_strongly_red_dominant_is_warm():
    """Only *strongly* red-dominant frames are guaranteed Warm.

    Note the asymmetry: "any red dominance implies Warm" is too strong and does not hold. The
    neutral band sits at 0.7-0.9 rather than centred on 1.0 because most real scenes carry a mild
    red bias, so a ratio of 0.95 is red-dominant in absolute terms yet relatively blue for a
    photograph — and correctly reads Cool. The guarantee is about the calibrated band, not about
    the raw channel comparison.
    """
    for red in range(150, 256, 10):
        assert lighting(color_ratio=(red * 0.5) / red)["tone"] == "Warm"


def test_strongly_blue_dominant_is_cool():
    for blue in range(150, 256, 10):
        assert lighting(color_ratio=blue / (blue * 0.6))["tone"] == "Cool"


# ── quality bands ──

@pytest.mark.parametrize(
    "brightness, expected",
    [
        (0, "Poor"), (30, "Poor"), (60, "Poor"),      # 60 is exclusive lower bound of Fair
        (61, "Fair"), (100, "Fair"),                   # 100 is inclusive upper bound of Fair
        (101, "Good"), (150, "Good"), (199, "Good"),
        (200, "Fair"), (229, "Fair"),
        (230, "Poor"), (255, "Poor"),
    ],
)
def test_quality_bands(brightness, expected):
    assert lighting(brightness=brightness)["quality"] == expected


def test_poor_quality_is_reachable_from_both_ends():
    """Poor must mean 'too dark OR too bright' — the client gates capture on it."""
    assert lighting(brightness=10)["quality"] == "Poor"
    assert lighting(brightness=250)["quality"] == "Poor"


# ── tips ──

def test_every_quality_tone_pair_has_a_tip():
    """No combination may fall through to the generic default."""
    generic = "Adjust your position for better light"
    brightness_for = {"Poor": 10.0, "Fair": 80.0, "Good": 150.0}
    ratio_for = {"Warm": 0.5, "Neutral": 0.8, "Cool": 1.2}
    for quality, tone in itertools.product(brightness_for, ratio_for):
        result = lighting(brightness=brightness_for[quality], color_ratio=ratio_for[tone])
        assert (result["quality"], result["tone"]) == (quality, tone)
        assert result["tip"] != generic, f"({quality}, {tone}) fell through to the default"


def test_cool_tips_are_not_given_to_warm_scenes():
    """Direct regression on the user-visible symptom of the inversion."""
    assert "cool tones" not in lighting(brightness=150, color_ratio=120 / 190)["tip"]


# ── composition ──

@pytest.mark.parametrize(
    "sharpness, expected", [(0.2, "Sharp"), (0.11, "Sharp"), (0.08, "Soft"), (0.01, "Blurry")]
)
def test_focus_thresholds(sharpness, expected):
    features = {"sharpness": sharpness, "alignment": 1.0, "balance": 1.0}
    assert assess_composition(features)["focus"] == expected


@pytest.mark.parametrize(
    "alignment, expected", [(1.0, "Level"), (0.85, "Level"), (0.7, "Slightly tilted"), (0.3, "Tilted")]
)
def test_horizon_thresholds(alignment, expected):
    features = {"sharpness": 0.2, "alignment": alignment, "balance": 1.0}
    assert assess_composition(features)["horizon"] == expected


@pytest.mark.parametrize(
    "balance, expected", [(1.0, "Balanced"), (0.7, "Slightly off"), (0.2, "Unbalanced")]
)
def test_balance_thresholds(balance, expected):
    features = {"sharpness": 0.2, "alignment": 1.0, "balance": balance}
    assert assess_composition(features)["balance"] == expected


# ── blueprint ──

def test_orientation_follows_the_aspect_ratio():
    base = {"alignment": 1.0, "balance": 1.0, "rule_of_thirds": 0.0}
    assert build_blueprint({**base, "width": 100, "height": 200})["orientation"] == "portrait"
    assert build_blueprint({**base, "width": 200, "height": 100})["orientation"] == "landscape"
    # square counts as portrait (h >= w)
    assert build_blueprint({**base, "width": 100, "height": 100})["orientation"] == "portrait"


def test_notes_are_empty_for_a_clean_frame():
    clean = {"width": 100, "height": 100, "alignment": 1.0, "balance": 1.0, "rule_of_thirds": 0.0}
    assert build_blueprint(clean)["notes"] == []


def test_notes_report_each_problem_independently():
    base = {"width": 100, "height": 100, "alignment": 1.0, "balance": 1.0, "rule_of_thirds": 0.0}
    tilted = build_blueprint({**base, "alignment": 0.5})["notes"]
    unbalanced = build_blueprint({**base, "balance": 0.4})["notes"]
    thirds = build_blueprint({**base, "rule_of_thirds": 0.8})["notes"]
    assert any("tilted" in n for n in tilted)
    assert any("unbalanced" in n for n in unbalanced)
    assert any("rule-of-thirds" in n for n in thirds)
    # and all three can co-occur
    everything = build_blueprint(
        {**base, "alignment": 0.5, "balance": 0.4, "rule_of_thirds": 0.8}
    )["notes"]
    assert len(everything) == 3


def test_grid_is_always_rule_of_thirds():
    """The client's overlay is hardcoded to thirds; the engine must not claim otherwise."""
    base = {"width": 100, "height": 100, "alignment": 1.0, "balance": 1.0, "rule_of_thirds": 0.0}
    assert build_blueprint(base)["grid"] == "rule_of_thirds"
