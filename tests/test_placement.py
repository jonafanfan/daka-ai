"""Tests for _compute_placement — where the app tells the subject to stand.

This is the product's core output: it drives the standing marker, and it is ~70 lines of tuned
heuristics whose claims are all *directional* ("stand opposite the focal mass", "stand on the
dimmer side", "never stand in front of a blowout"). A sign error in any of them would look
entirely plausible while giving exactly backwards advice — which is what happened to
assess_lighting's tone mapping. These tests pin each direction down independently.
"""
import numpy as np
import pytest

from scene_analysis import _compute_placement
from conftest import LEFT_THIRD, RIGHT_THIRD, gray, saliency

FALLBACK = {"x": RIGHT_THIRD, "y": RIGHT_THIRD}


# ── the output shape is a contract: x snaps to a thirds line, y stays in a usable band ──

def test_x_always_snaps_to_a_thirds_line():
    rng = np.random.default_rng(1)
    for _ in range(40):
        g = gray(rng.uniform(20, 250), rng.uniform(0, 255), rng.uniform(0, 255))
        s = saliency(rng.uniform(0, 0.5), (0, rng.uniform(0.2, 1.0), 0, rng.uniform(0.2, 1.0)))
        assert _compute_placement(g, s)["x"] in (LEFT_THIRD, RIGHT_THIRD)


def test_y_only_ever_takes_one_of_three_values(flat_gray):
    """y is not continuous — the three headroom branches are the whole range.

    Asserting the exact set rather than the [0.60, 0.72] band matters: the band assertion is a
    tautology, because every branch already sits inside it. Which also means the
    `min(0.72, max(0.60, y))` clamp in _compute_placement is currently unreachable — harmless
    belt-and-braces, but it protects nothing today and shouldn't be mistaken for live logic.
    """
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(40):
        s = saliency(rng.uniform(0, 0.5), (rng.uniform(0, 0.6), 1.0, 0, 1.0))
        seen.add(_compute_placement(flat_gray, s)["y"])
    assert seen <= {0.62, 0.667, 0.70}, f"unexpected y values: {seen - {0.62, 0.667, 0.70}}"
    assert len(seen) > 1, "inputs should have exercised more than one headroom branch"


# ── balance: stand opposite the scene's focal mass ──

def test_focal_mass_on_the_left_sends_the_subject_right(flat_gray):
    interest_left = saliency(0.0, box=(0.0, 1.0, 0.0, 0.33))
    assert _compute_placement(flat_gray, interest_left)["x"] == RIGHT_THIRD


def test_focal_mass_on_the_right_sends_the_subject_left(flat_gray):
    interest_right = saliency(0.0, box=(0.0, 1.0, 0.67, 1.0))
    assert _compute_placement(flat_gray, interest_right)["x"] == LEFT_THIRD


# ── light: stand on the dimmer side, so the bright side falls on the face ──

def test_bright_left_sends_the_subject_right(flat_saliency):
    """A window on the left means standing right, not standing in the window."""
    assert _compute_placement(gray(left=210, right=90), flat_saliency)["x"] == RIGHT_THIRD


def test_bright_right_sends_the_subject_left(flat_saliency):
    assert _compute_placement(gray(left=90, right=210), flat_saliency)["x"] == LEFT_THIRD


def test_symmetric_light_casts_no_vote(flat_saliency):
    """Below the 4% asymmetry floor, light abstains and the documented tie-break decides.

    With no signal at all and no hot region either, the tie-break is `not (rhot > lhot)` — both
    zero — so it resolves right. Asserting the exact value (rather than "either third") is what
    makes this test able to fail if the abstention floor is ever removed: a light vote here would
    read left as brighter and also pick right, so only the *absence* of asymmetry keeps this
    stable. Paired with the discriminating test below.
    """
    assert _compute_placement(gray(left=128, right=130), flat_saliency)["x"] == RIGHT_THIRD


def test_dark_scene_disables_the_light_vote(flat_saliency):
    """Under mean brightness 40 the light reading is noise and must not vote.

    Constructed so the two outcomes differ: the right half is brighter, so an *enabled* light
    vote would send the subject left. Because the frame is too dark to trust, no vote is cast and
    the tie-break resolves right instead. If the reliability gate were dropped this flips.
    """
    too_dark = gray(left=5, right=30)          # mean 17.5, well under the 40 floor
    assert too_dark.mean() < 40
    assert _compute_placement(too_dark, flat_saliency)["x"] == RIGHT_THIRD
    # Same asymmetry, lifted above the floor, resolves the other way — proving the gate is load-bearing.
    bright_enough = gray(left=60, right=200)
    assert bright_enough.mean() > 40
    assert _compute_placement(bright_enough, flat_saliency)["x"] == LEFT_THIRD


# ── backlight veto: never silhouette the subject ──

def test_blown_out_right_forbids_standing_right(flat_saliency):
    assert _compute_placement(gray(left=120, right=255), flat_saliency)["x"] == LEFT_THIRD


def test_blown_out_left_forbids_standing_left(flat_saliency):
    assert _compute_placement(gray(left=255, right=120), flat_saliency)["x"] == RIGHT_THIRD


def test_veto_overrides_a_strong_opposite_vote():
    """The veto is a hard constraint: saliency wanting the right cannot beat a blown-out right."""
    interest_left = saliency(0.0, box=(0.0, 1.0, 0.0, 0.33))   # votes right, twice
    blown_right = gray(left=120, right=255)                     # forbids right
    assert _compute_placement(blown_right, interest_left)["x"] == LEFT_THIRD


def test_a_small_bright_patch_is_not_a_backlight(flat_saliency):
    """Below the 6% hot-fraction floor a specular highlight must not trigger the *veto*.

    It does still tip the no-signal tie-break, which reads "avoid the hotter half" — so the
    subject goes left. That is the tie-break doing its job, not the veto firing: the distinction
    matters because the veto is a hard constraint that would override a strong opposite vote,
    whereas this is the weakest possible input. `test_veto_overrides_a_strong_opposite_vote`
    covers the hard-constraint case.
    """
    g = gray(120)
    g[:2, -4:] = 255                                       # ~0.08% of the frame
    hot_fraction = float((g[:, g.shape[1] // 2:] > 245).mean())
    assert hot_fraction < 0.06, "patch must stay under the veto floor for this test to mean anything"
    assert _compute_placement(g, flat_saliency)["x"] == LEFT_THIRD


# ── headroom: y follows where the visual mass sits ──

def test_top_heavy_scene_lowers_the_subject(flat_gray):
    top = saliency(0.0, box=(0.0, 0.33, 0.0, 1.0))
    assert _compute_placement(flat_gray, top)["y"] == 0.70


def test_bottom_heavy_scene_raises_the_subject(flat_gray):
    bottom = saliency(0.0, box=(0.67, 1.0, 0.0, 1.0))
    assert _compute_placement(flat_gray, bottom)["y"] == 0.62


# ── degradation: never raise, always return a usable point ──

@pytest.mark.parametrize(
    "bad_saliency",
    [
        np.zeros((10, 10, 3), dtype=np.float32),   # colour, not a 2-D map
        np.zeros((0, 0), dtype=np.float32),        # empty
        np.zeros((5,), dtype=np.float32),          # 1-D
        None,
    ],
    ids=["3d", "empty", "1d", "none"],
)
def test_malformed_saliency_falls_back(flat_gray, bad_saliency):
    assert _compute_placement(flat_gray, bad_saliency) == FALLBACK


@pytest.mark.parametrize(
    "bad_gray",
    [np.zeros((10, 10, 3), dtype=np.float32), None],
    ids=["3d", "none"],
)
def test_malformed_gray_falls_back(bad_gray, flat_saliency):
    assert _compute_placement(bad_gray, flat_saliency) == FALLBACK


def test_mismatched_shapes_do_not_raise():
    """gray and saliency disagreeing on size must degrade, not explode."""
    result = _compute_placement(gray(128)[:50], saliency(0.5))
    assert result["x"] in (LEFT_THIRD, RIGHT_THIRD)
    assert 0.60 <= result["y"] <= 0.72


# ── characterisation: how the three signals are actually weighted ──

def test_saliency_outweighs_light_when_they_disagree():
    """Documents current behaviour, and a design smell worth revisiting.

    The docstring calls this a fusion of three signals, but saliency casts TWO votes (balance
    +1.0, cleanliness +1.0) against light's single 1.2 — so light can never overturn a saliency
    decision, only soften it. Both saliency votes also derive from the same map, so they tend to
    agree, compounding the imbalance. If light is meant to be able to win, its weight has to
    exceed 2.0.
    """
    interest_left = saliency(0.0, box=(0.0, 1.0, 0.0, 0.33))   # both saliency votes -> right
    bright_right = gray(left=90, right=210)                     # light -> left
    assert _compute_placement(bright_right, interest_left)["x"] == RIGHT_THIRD
