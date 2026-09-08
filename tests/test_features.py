"""Tests for extract_features — the OpenCV half of the engine.

Needs a real image on disk, so these write synthetic scenes to a tmp path.
"""
import cv2
import numpy as np
import pytest

from scene_analysis import EDGE_SHARPNESS_MIN, MIN_EDGE_DENSITY, extract_features


def write(tmp_path, img, name="scene.jpg"):
    path = tmp_path / name
    # PNG for lossless fidelity: JPEG artefacts would perturb the very edge metrics under test.
    path = path.with_suffix(".png")
    assert cv2.imwrite(str(path), img)
    return str(path)


def checkerboard(h=240, w=320, cell=16, lo=30, hi=225):
    """High-contrast texture: plenty of crisp edges."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((y // cell) + (x // cell)) % 2 == 0:
                img[y : y + cell, x : x + cell] = hi
            else:
                img[y : y + cell, x : x + cell] = lo
    return img


def plain_wall(h=240, w=320, value=150):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ── the blur gate: the fix here was to stop rejecting low-texture scenes ──

def test_a_sharp_textured_scene_is_not_blurry(tmp_path):
    f = extract_features(write(tmp_path, checkerboard()))
    assert f["blurry"] is False
    assert f["sharpness"] > MIN_EDGE_DENSITY
    assert f["edge_sharpness"] > EDGE_SHARPNESS_MIN


def test_a_blurred_textured_scene_is_blurry(tmp_path):
    """Camera-shake blur: edges still exist but are soft, which is the signal the gate reads.

    Kernel size is load-bearing. Up to ~9 the edges stay crisp enough to pass; at 13 the mean
    Laplacian at edges drops under EDGE_SHARPNESS_MIN and the frame is correctly rejected. See
    test_extreme_blur_escapes_the_gate for what happens beyond that.
    """
    blurred = cv2.GaussianBlur(checkerboard(), (13, 13), 0)
    f = extract_features(write(tmp_path, blurred))
    assert f["sharpness"] > MIN_EDGE_DENSITY, "edges must survive for the gate to judge them"
    assert f["edge_sharpness"] < EDGE_SHARPNESS_MIN
    assert f["blurry"] is True


def test_extreme_blur_escapes_the_gate(tmp_path):
    """Characterisation of a real gap, not an endorsement of it.

    Past a certain blur every edge is smeared below Canny's threshold, so edge density hits zero
    and the "too plain to judge" escape hatch fires — a severely defocused frame passes. The two
    conditions are indistinguishable by edge density alone: a plain wall and a destroyed image
    both have no edges.

    How reachable this is on real photos is unclear and probably narrower than this test implies.
    A checkerboard carries one spatial frequency, so once it is gone nothing remains; real scenes
    are broadband and their large-scale structure (wall lines, furniture silhouettes) tends to
    survive blur and keep the gate judging. Recorded so the limitation is known rather than
    discovered later, and so this test starts failing if the gate is ever tightened.
    """
    destroyed = cv2.GaussianBlur(checkerboard(), (31, 31), 0)
    f = extract_features(write(tmp_path, destroyed))
    assert f["sharpness"] < MIN_EDGE_DENSITY
    assert f["edge_sharpness"] == -1.0
    assert f["blurry"] is False, "currently passes — see docstring"


def test_a_plain_wall_gets_the_benefit_of_the_doubt(tmp_path):
    """The regression the content-robust gate exists for.

    Variance-of-Laplacian alone flags any low-texture scene as blurry, which blocked the whole
    flow on a plain wall or a minimalist cafe. A frame with nothing to be blurry must pass, and
    signal that it was unjudgeable via the sentinel rather than a real measurement.
    """
    f = extract_features(write(tmp_path, plain_wall()))
    assert f["sharpness"] < MIN_EDGE_DENSITY, "wall should be below the edge-density floor"
    assert f["blurry"] is False
    assert f["edge_sharpness"] == -1.0, "sentinel for 'too plain to judge'"


def test_blur_var_is_reported_even_when_unjudgeable(tmp_path):
    """blur_var is the tuning diagnostic — it must be real regardless of the gate's verdict."""
    f = extract_features(write(tmp_path, plain_wall()))
    assert f["edge_sharpness"] == -1.0
    assert f["blur_var"] >= 0.0


# ── geometry ──

def test_orientation_dimensions_are_reported_truthfully(tmp_path):
    f = extract_features(write(tmp_path, checkerboard(h=300, w=200)))
    assert (f["height"], f["width"]) == (300, 200)


def test_level_lines_score_as_aligned(tmp_path):
    """Horizontal edges only: nothing in the 0.5-20 degree 'critical' band, so alignment stays 1."""
    img = plain_wall(value=40)
    for y in range(20, 240, 20):
        img[y : y + 3, :] = 220
    f = extract_features(write(tmp_path, img))
    assert f["alignment"] == pytest.approx(1.0)


def test_tilted_lines_reduce_alignment(tmp_path):
    """A few degrees of tilt is exactly what the horizon warning is for."""
    img = plain_wall(value=40)
    h, w = img.shape[:2]
    for offset in range(-200, 400, 40):
        cv2.line(img, (0, offset), (w, offset + int(w * 0.12)), (220, 220, 220), 3)  # ~7 degrees
    f = extract_features(write(tmp_path, img))
    assert f["alignment"] < 1.0
    assert 0.0 <= f["alignment"] <= 1.0


# ── placement is wired through ──

def test_placement_is_included_and_well_formed(tmp_path):
    f = extract_features(write(tmp_path, checkerboard()))
    assert set(f["placement"]) == {"x", "y", "reason", "reason_text"}
    assert f["placement"]["x"] in (round(1 / 3, 3), round(2 / 3, 3))
    assert 0.60 <= f["placement"]["y"] <= 0.72


# ── invariants every caller relies on ──

def test_all_documented_keys_are_present(tmp_path):
    f = extract_features(write(tmp_path, checkerboard()))
    assert set(f) == {
        "brightness", "color_ratio", "sharpness", "blur_var", "edge_sharpness", "blurry",
        "rule_of_thirds", "alignment", "balance", "placement", "camera_tilt", "width", "height",
    }


def test_color_ratio_is_blue_over_red(tmp_path):
    """Direction matters and is easy to invert — assert it against an unambiguous image.

    A strongly red frame must score LOW. cv2 uses BGR, so channel order is (blue, green, red).
    """
    red_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 220        # red channel
    red_frame[:, :, 0] = 40         # blue channel
    assert extract_features(write(tmp_path, red_frame))["color_ratio"] < 0.7

    blue_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    blue_frame[:, :, 0] = 220
    blue_frame[:, :, 2] = 40
    assert extract_features(write(tmp_path, blue_frame))["color_ratio"] > 0.9


def test_brightness_tracks_the_image(tmp_path):
    dark = extract_features(write(tmp_path, plain_wall(value=20), "dark"))["brightness"]
    bright = extract_features(write(tmp_path, plain_wall(value=230), "bright"))["brightness"]
    assert dark < 40 < bright


def test_unreadable_file_raises_valueerror(tmp_path):
    """api_server maps ValueError to a 400, so this is the contract for a corrupt upload."""
    junk = tmp_path / "not-an-image.jpg"
    junk.write_bytes(b"definitely not a JPEG")
    with pytest.raises(ValueError):
        extract_features(str(junk))


def test_missing_file_raises_valueerror(tmp_path):
    with pytest.raises(ValueError):
        extract_features(str(tmp_path / "absent.jpg"))
