"""Runs the real render loop against a stubbed DOM.

Two bugs shipped to a phone in consecutive PRs, both the same shape: an identifier that was not in
scope, inside the overlay code. `tf.H` after a signature change, then `ts` after adding subject
detection to a loop that takes no timestamp parameter. Both are valid JavaScript, so `node --check`
passed and CI was green. Both only failed when the function actually ran.

`test_client_overlay.py` executes individual drawing functions. This file executes the *loop* —
the thing that decides whether anything gets drawn at all — because that is where both bugs lived
and neither would have been caught by testing the drawing functions alone.

The stub is deliberately dumb: it records calls and returns plausible values. It is not a browser
and cannot tell you the marker looks right. What it can tell you is that a frame completes without
throwing, and that the marker and cue are actually reached — which is exactly what was broken.
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

# Everything the page touches at load time and during a frame. Anything missing here shows up as a
# thrown error rather than a silent pass, which is the point.
DOM_STUB = r"""
const drawn = [];
const cues = [];
function el(id) {
  return {
    id,
    style: new Proxy({}, { set: () => true, get: () => '' }),
    classList: { add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    addEventListener(){}, removeEventListener(){}, click(){},
    querySelectorAll(){ return []; }, appendChild(){}, remove(){},
    set textContent(v) { if (id === 'coachText') cues.push(v); },
    get textContent() { return ''; },
    innerHTML: '', value: '', files: [],
    clientWidth: 390, clientHeight: 844, width: 390, height: 844,
    videoWidth: 1080, videoHeight: 1920, srcObject: null,
    getContext() {
      return {
        save(){}, restore(){}, beginPath(){}, ellipse(){ drawn.push('marker'); },
        fill(){}, stroke(){}, moveTo(){}, lineTo(){}, setLineDash(){},
        clearRect(){}, drawImage(){}, putImageData(){},
        fillText(t){ drawn.push('caption:' + t); },
        measureText(s){ return { width: s.length * 7 }; },
        getImageData(w, h){ return { data: new Uint8ClampedArray(64 * 64 * 4) }; },
      };
    },
    toBlob(cb){ cb({}); }, toDataURL(){ return 'data:image/jpeg;base64,x'; },
    play(){ return Promise.resolve(); },
  };
}
const document = {
  getElementById: el,
  createElement: el,
  querySelectorAll(){ return []; },
  addEventListener(){},
};
let rafCount = 0;
const window = {
  addEventListener(){},
  DeviceMotionEvent: undefined,
  DeviceOrientationEvent: undefined,
  innerWidth: 390, innerHeight: 844,
};
const navigator = { mediaDevices: { getUserMedia(){ return Promise.reject(new Error('no cam')); },
                                    enumerateDevices(){ return Promise.resolve([]); } } };
const requestAnimationFrame = () => { rafCount++; return 1; };
const cancelAnimationFrame = () => {};
const performance = { now: () => 1000 };
const fetch = () => Promise.reject(new Error('offline'));
"""


def page_script():
    """The page's inline script, minus the top-level wiring that needs a live DOM."""
    html = PAGE.read_text(encoding="utf-8")
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    # The trailing $('id').addEventListener(...) wiring runs fine against the stub, but the two
    # IIFEs and the dynamic import of MediaPipe do not belong in a unit test.
    js = js.replace("import(MP)", "Promise.reject(new Error('no cdn'))")
    return js


def run(extra):
    """Write the harness to a file and run it. Passing it via `node -e` blows the Windows
    command-line length limit, which fails as a confusing FileNotFoundError."""
    script = DOM_STUB + page_script() + extra
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "harness.mjs"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            ["node", str(path)], capture_output=True, text=True, timeout=30,
        )
    assert result.returncode == 0, (
        f"the page threw while running a frame\n--- stderr ---\n{result.stderr.strip()[:2000]}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


SCAN = """
  standPos = { x: 0.667, y: 0.667 };
  standReason = 'Light falls on your face';
  coachingActive = true;
  liveActive = true;
"""


def test_a_frame_completes_without_throwing():
    """The regression both bugs would have failed. Neither was a syntax error."""
    out = run(SCAN + """
      liveLoop(1234);
      console.log(JSON.stringify({ drawn, cues }));
    """)
    assert out["drawn"], "nothing was drawn — the loop threw before reaching the marker"


def test_the_marker_is_drawn():
    out = run(SCAN + """
      liveLoop(1234);
      console.log(JSON.stringify({ drawn }));
    """)
    assert "marker" in out["drawn"], f"marker missing; drew {out['drawn']}"


def test_the_caption_is_drawn():
    """This is what the `tf` bug broke: the marker drew, the caption did not."""
    out = run(SCAN + """
      liveLoop(1234);
      console.log(JSON.stringify({ drawn }));
    """)
    captions = [d for d in out["drawn"] if d.startswith("caption:")]
    assert captions == ["caption:Light falls on your face"], f"got {captions}"


def test_a_cue_is_set_every_frame():
    """This is what BOTH bugs broke: the exception escaped before setCue ran, so the cue froze on
    whatever text the HTML shipped with."""
    out = run(SCAN + """
      liveLoop(1234);
      console.log(JSON.stringify({ cues }));
    """)
    assert out["cues"], "no cue was set — the loop threw before reaching setCue"


def test_the_loop_keeps_scheduling_itself():
    out = run(SCAN + """
      liveLoop(1234);
      console.log(JSON.stringify({ rafCount }));
    """)
    assert out["rafCount"] == 1, "the loop must re-arm even on a frame that draws nothing"


def test_it_survives_a_frame_with_no_scan_yet():
    """Before any scan there is no standPos. The loop still has to run and re-arm."""
    out = run("""
      liveActive = true;
      liveLoop(1234);
      console.log(JSON.stringify({ rafCount, drawn }));
    """)
    assert out["rafCount"] == 1
    assert out["drawn"] == [], "nothing should be drawn before a scan"


def test_it_survives_the_tracker_never_loading():
    """The CDN is blocked in this harness, so this is the real degradation path."""
    out = run(SCAN + """
      tracker.failed = true;
      liveLoop(1234);
      console.log(JSON.stringify({ drawn, cues }));
    """)
    assert "marker" in out["drawn"], "the marker must still draw without subject detection"
    assert out["cues"], "a fallback cue must still be set"
