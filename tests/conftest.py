"""Shared helpers for building synthetic scenes.

The engine's inputs are just arrays, so a "scene" here is a handful of rectangles: a bright
region for a window, a high-saliency region for visual interest, a blown-out region for a
backlight. That is enough to pin down every directional claim the placement heuristic makes.
"""
import numpy as np
import pytest

H, W = 120, 160  # small but not square, so x/y mix-ups surface


def gray(fill=128.0, left=None, right=None):
    """Grayscale brightness plane. `left`/`right` override each half."""
    g = np.full((H, W), float(fill), dtype=np.float32)
    if left is not None:
        g[:, : W // 2] = float(left)
    if right is not None:
        g[:, W // 2 :] = float(right)
    return g


def saliency(base=0.0, box=None, value=1.0):
    """Saliency map, optionally with one hot box. `box` is (y0, y1, x0, x1) in fractions."""
    s = np.full((H, W), float(base), dtype=np.float32)
    if box is not None:
        y0, y1, x0, x1 = box
        s[int(y0 * H) : int(y1 * H), int(x0 * W) : int(x1 * W)] = float(value)
    return s


LEFT_THIRD = round(1 / 3, 3)
RIGHT_THIRD = round(2 / 3, 3)


@pytest.fixture
def flat_gray():
    """Uniform mid-grey: bright enough to be 'reliable', with no left/right asymmetry."""
    return gray(128.0)


@pytest.fixture
def flat_saliency():
    """Uniform saliency — deliberately *unreliable*, so only the light signal votes."""
    return saliency(0.5)
