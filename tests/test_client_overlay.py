"""Runtime checks on the viewfinder overlay drawing code.

The client had no tests at all, and that let a real bug ship: a refactor changed
`drawStandMarker(tf, ctx, ...)` to `drawStandMarker(W, H, ctx, ...)` but left one `tf.H` behind in
the caption's font line. `tf` was no longer in scope, so the function threw a ReferenceError on
every frame — after drawing the marker but before drawing its caption. The exception escaped the
whole coaching block, so `setCue()` never ran and every cue froze on the placeholder text baked
into the HTML. Silently, and only on a device.

`node --check` cannot catch that: it is valid syntax. Only *running* the function does. These
tests extract the real function from the shipped page and execute it against a stub canvas, so an
out-of-scope identifier fails here instead of on someone's phone.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def function_source(name):
    """Pull one top-level function out of the page, verbatim."""
    html = PAGE.read_text(encoding="utf-8")
    match = re.search(r"^    (?:async )?function " + name + r"\(.*?^    \}$", html, re.S | re.M)
    assert match, f"{name} not found in {PAGE.name}"
    return match.group(0)


def run_js(script):
    """Run a snippet in node. Returns whatever it prints as JSON on the last line."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        # encoding matters: text=True alone uses the platform locale, mangling non-ASCII.
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, (
        f"node exited {result.returncode}\n--- stderr ---\n{result.stderr.strip()}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


STUB_CTX = """
const called = [];
const ctx = {
  save(){}, restore(){}, beginPath(){}, ellipse(){}, fill(){}, stroke(){},
  moveTo(){}, lineTo(){}, setLineDash(){},
  measureText(s){ return { width: s.length * 7 }; },
  fillText(t, x, y){ called.push({ t, x, y }); },
};
"""


def test_marker_draws_without_throwing():
    """The regression itself: any out-of-scope identifier surfaces as a non-zero exit."""
    out = run_js(function_source("drawStandMarker") + STUB_CTX + """
      drawStandMarker(390, 844, ctx, 0.667, 0.667, false, 'Light falls on your face');
      console.log(JSON.stringify({ captions: called.length }));
    """)
    assert out["captions"] == 1, "the caption should have been drawn exactly once"


def test_marker_draws_with_no_caption():
    """No reason text is a normal state — older responses carry none."""
    out = run_js(function_source("drawStandMarker") + STUB_CTX + """
      drawStandMarker(390, 844, ctx, 0.333, 0.667, true, '');
      console.log(JSON.stringify({ captions: called.length }));
    """)
    assert out["captions"] == 0


def test_caption_stays_inside_the_frame():
    """A marker near the edge must not push its own caption off screen."""
    out = run_js(function_source("drawStandMarker") + STUB_CTX + """
      const W = 390;
      const seen = [];
      for (const x of [0.02, 0.333, 0.667, 0.98]) {
        called.length = 0;
        drawStandMarker(W, 844, ctx, x, 0.667, false, 'Stand in front of the blue door');
        const c = called[0];
        seen.push({ x, left: c.x - ctx.measureText('Stand in front of the blue door').width / 2,
                       right: c.x + ctx.measureText('Stand in front of the blue door').width / 2 });
      }
      console.log(JSON.stringify({ seen, W }));
    """)
    for row in out["seen"]:
        assert row["left"] >= -1, f"caption runs off the left at x={row['x']}"
        assert row["right"] <= out["W"] + 1, f"caption runs off the right at x={row['x']}"


def test_marker_maps_normalised_coords_straight_to_the_canvas():
    """No cover transform here — engine coords are already relative to the visible crop.

    Re-introducing one is exactly what put the marker off screen before, so this pins the mapping.
    """
    out = run_js(function_source("drawStandMarker") + STUB_CTX + """
      drawStandMarker(400, 800, ctx, 0.5, 0.5, false, 'x');
      console.log(JSON.stringify({ y: called[0].y }));
    """)
    # caption sits just under the footprint: fy + rh + 10, with fy = 0.5*800 and rh = 0.028*800
    assert out["y"] == pytest.approx(0.5 * 800 + 0.028 * 800 + 10)


def test_visible_crop_matches_the_screen_aspect():
    """The crop fed to the engine must be exactly what the viewfinder shows, for any track shape.

    When these disagreed, the engine was analysing scenery the user could not see and the marker
    was drawn off screen.
    """
    out = run_js(function_source("visibleCrop") + """
      const rows = [];
      for (const [vw, vh] of [[1920,1080],[1080,1920],[1440,1440],[640,480]]) {
        const c = visibleCrop({ videoWidth: vw, videoHeight: vh, clientWidth: 390, clientHeight: 844 });
        rows.push({ vw, vh, ...c });
      }
      console.log(JSON.stringify(rows));
    """)
    for row in out:
        assert row["sw"] <= row["vw"] and row["sh"] <= row["vh"], "crop must fit inside the track"
        assert row["sx"] >= 0 and row["sy"] >= 0
        assert row["sw"] / row["sh"] == pytest.approx(390 / 844, rel=0.01), (
            f"crop aspect must match the screen for a {row['vw']}x{row['vh']} track"
        )


# ── caption placement vs the cue pill ────────────────────────────────────────

def test_caption_flips_above_the_marker_when_the_cue_pill_would_cover_it():
    """placement.y tops out at 0.72, and at that height the caption lands under the cue pill.

    The pill is anchored 160px off the bottom, so a caption below roughly H-205 is unreadable.
    Rather than let it hide, it flips above the footprint.
    """
    out = run_js(function_source("drawStandMarker") + STUB_CTX + """
      const H = 844, rows = [];
      for (const y of [0.60, 0.667, 0.72]) {
        called.length = 0;
        drawStandMarker(390, H, ctx, 0.667, y, false, 'Light falls on your face');
        rows.push({ y, capY: called[0].y, footY: y * H });
      }
      console.log(JSON.stringify(rows));
    """)
    for row in out:
        clear_of_pill = row["capY"] <= 844 - 205
        above_marker = row["capY"] < row["footY"]
        assert clear_of_pill or above_marker, (
            f"at y={row['y']} the caption sits at {row['capY']}, under the cue pill"
        )


def test_a_high_marker_still_captions_below():
    """Flipping is a last resort — below the footprint reads better, so keep it where it fits."""
    out = run_js(function_source("drawStandMarker") + STUB_CTX + """
      drawStandMarker(390, 844, ctx, 0.667, 0.60, false, 'Cleaner background here');
      console.log(JSON.stringify({ capY: called[0].y, footY: 0.60 * 844 }));
    """)
    assert out["capY"] > out["footY"], "should still sit below when there is room"
