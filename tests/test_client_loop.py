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
const drawn = [];      // every marker / caption the overlay draws
const played = [];     // every play() call, by element id — a suspended <video> needs one
// styles and toggles are separate arrays, so an index into one says nothing about the other.
// A shared counter stamped on both is what makes "did A happen before B" answerable.
let seq = 0;
let frameTranslate = 0;   // px the frame is currently transformed by, reflected in its rect
const hiddenSet = [];     // every `el.hidden = v`, in order, by element id
const cues = [];      // every #coachText assignment, in order
const hints = [];     // every #placeHint assignment, in order
const styles = [];    // every style property assignment, tagged with the element id
const toggles = [];   // every classList.toggle(cls, val) call, tagged with the element id
function el(id) {
  return {
    id,
    style: new Proxy({}, {
      set: (t, k, val) => {
        t[k] = val; styles.push({ id, prop: k, val, seq: seq++ });
        // A real transform moves the element's rect. Modelling that is what lets a test see the
        // frame drift when placeFrame forgets how far it has already been pushed.
        if (id === 'camFrame' && k === 'transform') {
          const m = /translateY\((-?[\d.]+)px\)/.exec(val || '');
          frameTranslate = m ? parseFloat(m[1]) : 0;
        }
        return true;
      },
      get: (t, k) => t[k] || '',
    }),
    classList: {
      // add/remove are recorded as toggles too, so a test can read the resulting state whichever
      // call the page happened to use to get there.
      add(...cs){ cs.forEach(cls => toggles.push({ id, cls, val: true, seq: seq++ })); },
      remove(...cs){ cs.forEach(cls => toggles.push({ id, cls, val: false, seq: seq++ })); },
      toggle(cls, val){ toggles.push({ id, cls, val, seq: seq++ }); },
      contains(){ return false; },
    },
    // getElementById hands back a fresh object each call, so a plain property would not survive
    // being read back. Recorded instead, which is also what lets a test see the ORDER things were
    // shown in.
    set hidden(v) { hiddenSet.push({ id, v }); },
    get hidden() { const l = hiddenSet.filter(h => h.id === id); return l.length ? l[l.length - 1].v : true; },
    addEventListener(){}, removeEventListener(){}, click(){},
    querySelectorAll(){ return []; }, appendChild(){}, remove(){},
    set textContent(v) {
      if (id === 'coachText') cues.push(v);
      if (id === 'placeHint') hints.push(v);
    },
    get textContent() { return ''; },
    innerHTML: '', value: '', files: [],
    clientWidth: 390, clientHeight: 844, width: 390, height: 844,
    // Real enough for the two things that measure themselves: the controls bar, which animates
    // its own height, and the frame, which is placed against the height of the screen. A single
    // rect for every element made the frame's natural position equal to the centred one, so any
    // assertion about where it travels to would have passed on arithmetic that never ran.
    getBoundingClientRect() {
      if (id === 'camera')   return { width: 390, height: 844, top: 0, left: 0, right: 390, bottom: 844 };
      // 3:4 at full width, sitting below the status area — the top-middle resting position.
      if (id === 'camFrame') return { width: 390, height: 520, top: 80 + frameTranslate, left: 0, right: 390, bottom: 600 + frameTranslate };
      return { width: 390, height: 140, top: 0, left: 0, right: 390, bottom: 140 };
    },
    offsetHeight: 140,
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
    play(){ played.push(id); return Promise.resolve(); },
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
// `let`, not `const`: a test that needs to answer a request replaces this binding, and the
// page resolves the identifier here rather than on globalThis, so a globalThis assignment
// silently does nothing.
let fetch = () => Promise.reject(new Error('offline'));
"""


def page_script():
    """The page's inline script, minus the top-level wiring that needs a live DOM."""
    html = PAGE.read_text(encoding="utf-8")
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    # The trailing $('id').addEventListener(...) wiring runs fine against the stub, but the two
    # IIFEs and the dynamic import of MediaPipe do not belong in a unit test.
    return js.replace("import(MP)", "Promise.reject(new Error('no cdn'))")


def run(extra):
    """Write the harness to a file and run it. Passing it via `node -e` blows the Windows
    command-line length limit, which fails as a confusing FileNotFoundError."""
    script = DOM_STUB + page_script() + extra
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "harness.mjs"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            # encoding is not optional: text=True alone decodes with the platform locale, which on
            # Windows is cp1252 and mangles every non-ASCII character the cues contain.
            ["node", str(path)], capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
    assert result.returncode == 0, (
        f"the page threw while running a frame\n--- stderr ---\n{result.stderr.strip()[:2000]}"
    )
    out = result.stdout.strip().splitlines()
    assert out, (
        "the page ran but printed nothing. Usually a promise that never settles, so the .then() "
        "holding the console.log never fires and node exits quietly — which is how a hung "
        "presentPaywall looked on a device: a button that did nothing at all.\n"
        f"--- stderr ---\n{result.stderr.strip()[:1000]}"
    )
    return json.loads(out[-1])


# 0.88 is where the engine actually puts the feet. The old fixtures used 0.667, a value
# _compute_placement can no longer produce — they passed while describing a frame nobody sees.
SCAN = """
  standPos = { x: 0.667, y: 0.88 };
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


# ── the coaching flow: one cue at a time, in the right order ─────────────────
#
# The requested order is subject -> position -> camera. These pin it down, because the order is a
# product decision rather than an implementation detail: a crooked frame nobody is standing in is
# not worth complaining about yet.

def cue_for(setup):
    out = run(SCAN + """
      tracker.landmarker = {};              // pretend the model loaded
      streak = CONFIRM_FRAMES;              // and that detection has been confirmed
      """ + setup + """
      liveLoop(1234);
      console.log(JSON.stringify({ cues }));
    """)
    return out["cues"][-1]


def test_no_subject_asks_them_into_frame():
    assert cue_for("subject.seen = false;") == "Get them in frame"


def test_subject_off_to_one_side_is_told_which_way():
    assert cue_for("subject.seen = true; subject.x = 0.30; subject.y = 0.88;") == "Move them right"
    assert cue_for("subject.seen = true; subject.x = 0.95; subject.y = 0.88;") == "Move them left"


def test_subject_too_far_or_too_near_is_told_so():
    """Feet higher in frame than the marker means further away."""
    assert cue_for("subject.seen = true; subject.x = 0.667; subject.y = 0.40;") == "Bring them closer"
    assert cue_for("subject.seen = true; subject.x = 0.667; subject.y = 0.99;") == "Send them back"


def test_the_depth_tolerance_is_tighter_below_the_marker_than_above():
    """The frame is asymmetric, so the tolerance has to be.

    With the feet at 0.88 there is only ~0.12 of picture below the marker and 0.88 above it. One
    tolerance both ways would need feet past the bottom edge before "send them back" could fire,
    which is to say never — and someone standing too close is exactly who needs telling, because
    a few steps more and the tracker loses their legs off the bottom of the frame entirely.
    """
    near = "subject.seen = true; subject.x = 0.667; subject.y = 0.96;"    # 0.08 too close
    far = "subject.seen = true; subject.x = 0.667; subject.y = 0.80;"     # 0.08 too far
    assert cue_for(near) == "Send them back", "0.08 too close should be corrected"
    assert cue_for(far) == "Perfect — take the photo", "0.08 too far is within tolerance"


def test_the_marker_is_not_green_while_a_depth_cue_is_showing():
    """Otherwise the app contradicts itself: a green marker saying you are in position, over a cue
    telling you to move. They read the same tolerance for exactly this reason."""
    out = run(SCAN + """
      tracker.landmarker = {}; streak = CONFIRM_FRAMES;
      subject.seen = true; subject.x = 0.667; subject.y = 0.99;
      liveLoop(1234);
      console.log(JSON.stringify({ onMarker, cue: cues[cues.length - 1] }));
    """)
    assert out["cue"] == "Send them back"
    assert out["onMarker"] is False, "green marker under a move cue"


def test_position_is_settled_before_the_camera_is_mentioned():
    """Someone still walking into place must not also be told to straighten the camera."""
    cue = cue_for("""
      subject.seen = true; subject.x = 0.30; subject.y = 0.88;
      needsStraightening = true; liveLean = 20; tiltDirection = 'down';
    """)
    assert cue == "Move them right", f"camera cue jumped the queue: {cue!r}"


def test_dead_space_is_cued_when_the_horizon_is_fine():
    """Phrased at render time from the direction, not stored as a finished sentence — see
    test_client_i18n for why that distinction is load-bearing."""
    assert cue_for("subject.seen = true; subject.x = 0.667; subject.y = 0.88; tiltDirection = 'down'; tiltFallback = 'engine wording';") == \
        "Empty space above — aim a little lower"


def test_an_unphraseable_direction_falls_back_to_the_engines_wording():
    assert cue_for("subject.seen = true; subject.x = 0.667; subject.y = 0.88; tiltDirection = 'sideways'; tiltFallback = 'Engine wording';") == \
        "Engine wording"


def test_straightening_takes_priority_over_the_tilt_hint():
    cue = cue_for("subject.seen = true; subject.x = 0.667; subject.y = 0.88; needsStraightening = true; liveLean = 20; tiltDirection = 'down';")
    assert cue == "Straighten the camera", f"tilt hint jumped the queue: {cue!r}"


def test_the_needs_fix_class_follows_the_straighten_cue():
    out = run(SCAN + """
      tracker.landmarker = {}; streak = CONFIRM_FRAMES;
      subject.seen = true; subject.x = 0.667; subject.y = 0.88;
      needsStraightening = true; liveLean = 20;
      liveLoop(1234);
      needsStraightening = false;
      liveLoop(1235);
      const fixToggles = toggles.filter(t => t.id === 'camLevel' && t.cls === 'needs-fix');
      console.log(JSON.stringify({ fixToggles }));
    """)
    vals = [t["val"] for t in out["fixToggles"]]
    assert vals == [True, False], f"needs-fix should track the straighten cue, got {vals}"


def test_camera_cues_come_once_they_are_on_the_marker():
    on_marker = "subject.seen = true; subject.x = 0.667; subject.y = 0.88;"
    assert cue_for(on_marker + "needsStraightening = true; liveLean = 20;") == "Straighten the camera"
    assert cue_for(on_marker + "tiltDirection = 'down'; tiltFallback = 'engine wording';") == \
        "Empty space above — aim a little lower"


def test_everything_satisfied_says_take_the_photo():
    assert cue_for("subject.seen = true; subject.x = 0.667; subject.y = 0.88;") \
        == "Perfect — take the photo"


def test_the_marker_is_green_only_when_someone_is_on_it():
    """Orange means 'not yet' at a glance — the whole point of the colour."""
    out = run(SCAN + """
      tracker.landmarker = {};
      subject.seen = true; subject.x = 0.667; subject.y = 0.88;
      liveLoop(1234);
      const wasOn = onMarker;
      subject.x = 0.20;
      liveLoop(1235);
      console.log(JSON.stringify({ wasOn, nowOn: onMarker }));
    """)
    assert out["wasOn"] is True and out["nowOn"] is False


# ── false-positive resistance ────────────────────────────────────────────────

def test_a_single_detection_frame_is_not_believed():
    """A person-shaped object produces one good frame. A person produces many.

    Without this the marker turned green the instant the model guessed, which it will do on an
    empty room because it is trained to find someone rather than to decide whether anyone is there.
    """
    out = run(SCAN + """
      tracker.landmarker = {};
      streak = 1;                      // one confirming frame so far
      subject.seen = false;
      liveLoop(1234);
      console.log(JSON.stringify({ cues, onMarker }));
    """)
    assert out["cues"][-1] == "Get them in frame"
    assert out["onMarker"] is False, "must not go green on a single frame"


def test_confirmation_is_required_before_the_marker_can_go_green():
    out = run(SCAN + """
      tracker.landmarker = {};
      subject.seen = true; subject.x = 0.667; subject.y = 0.88;
      streak = CONFIRM_FRAMES;
      liveLoop(1234);
      console.log(JSON.stringify({ onMarker }));
    """)
    assert out["onMarker"] is True


def test_the_marker_stays_orange_when_nobody_has_been_seen():
    """Found by mutation testing: dropping `subject.seen` from the onMarker test left every other
    test green, because they all set it true before checking the distance.

    The stale coordinates are the trap. subject.x/.y keep their last values after someone walks
    out of shot, so position alone still reads as "on the spot" — and a marker that goes green at
    an empty room is the exact failure the confirmation logic exists to prevent."""
    out = run(SCAN + """
      tracker.landmarker = {}; streak = CONFIRM_FRAMES;
      subject.seen = false;             // nobody in shot ...
      subject.x = 0.667; subject.y = 0.88;   // ... but the last sighting was right on the marker
      liveLoop(1234);
      console.log(JSON.stringify({ onMarker, cue: cues[cues.length - 1] }));
    """)
    assert out["onMarker"] is False, "marker went green with nobody in frame"
    assert out["cue"] == "Get them in frame"


# The <video> is shown cropped, so landmarks normalised to the full track have to be converted
# before they can be compared with a marker that lives in the visible crop. Skipping the
# conversion is what once drew the marker hundreds of pixels off screen.
POSE_LANDMARKS = """
function landmarksAt(x, footY, headY) {
  const lm = [];
  for (let i = 0; i < 33; i++) lm.push({ x, y: footY, visibility: 0.9 });
  lm[0]  = { x, y: headY, visibility: 0.9 };   // nose
  lm[11] = { x, y: headY + 0.1, visibility: 0.9 };
  lm[12] = { x, y: headY + 0.1, visibility: 0.9 };
  lm[27] = { x, y: footY, visibility: 0.9 };   // ankles
  lm[28] = { x, y: footY, visibility: 0.9 };
  return lm;
}
"""


def test_landmarks_are_converted_into_the_visible_crop():
    """The track is 1080x1920 shown in a 390x844 box, so `cover` keeps a centre slice about 887px
    wide. An ankle at x=0.8 of the full frame is therefore further right than 0.8 of what is on
    screen — reporting the raw value would put the subject somewhere nobody can see."""
    out = run(POSE_LANDMARKS + SCAN + """
      tracker.landmarker = { detectForVideo: () => ({ landmarks: [landmarksAt(0.8, 0.9, 0.2)] }) };
      streak = CONFIRM_FRAMES - 1;      // this frame is the confirming one
      subject.seen = false;
      detectSubject($('video'), 1234);
      console.log(JSON.stringify({ x: subject.x, seen: subject.seen }));
    """)
    assert out["seen"] is True, "a confirmed pose was not accepted"
    assert abs(out["x"] - 0.865) < 0.01, (
        f"landmark used raw instead of crop-relative: {out['x']:.3f} (raw would be 0.800)"
    )


def all_cue_strings():
    """Every cue the page can show, in English, including the ones inside ternaries.

    Matching only `setCue('...')` misses `setCue(a ? 'x' : 'y')` — which is where the movement
    cues live, i.e. exactly the ones this file cares about. That version of the check passed while
    testing nothing.

    Cues are now table keys rather than literals, so the keys are collected from the call sites and
    resolved through the page's own English table. Reading the table instead of restating it means
    a cue that is reworded here is still checked; a cue that is *deleted* disappears from both, so
    the guard test below keeps this from silently returning nothing.
    """
    page = PAGE.read_text(encoding="utf-8")
    keys = set()
    for call in re.findall(r"setCue\(([^;]*?)\);", page):
        keys |= set(re.findall(r"tr\('([^']+)'\)", call))
    english = run("console.log(JSON.stringify(STRINGS.en));")
    return {english[k] for k in keys if k in english}


def test_the_cue_scan_finds_the_movement_cues():
    """Guard on the guard: if this ever returns nothing, the test below is vacuous."""
    cues = all_cue_strings()
    assert len(cues) >= 6, f"expected the full cue set, found {sorted(cues)}"
    assert any(c.startswith("Move") for c in cues), "movement cues not found — check the regex"


def test_every_movement_cue_says_who_moves():
    """The reader is holding the phone. A cue telling *them* to move left, when it is the subject
    who should move — and screen-left at that — sends everyone the wrong way."""
    for cue in all_cue_strings():
        if cue.split()[0] in ("Move", "Bring", "Send", "Step", "Come"):
            assert " them" in cue, f"ambiguous about who moves: {cue!r}"


# ── the confirmation logic itself, driven by a fake pose ─────────────────────
#
# The flow tests above pre-set `subject.seen` and never enter detectSubject, so they say nothing
# about whether a detection is *believed*. These feed real landmark data through it.

FAKE_POSE = """
function pose(opts) {
  const o = Object.assign({ vis: 0.9, noseY: 0.25, ankleY: 0.85 }, opts || {});
  const lm = [];
  for (let i = 0; i < 33; i++) lm.push({ x: 0.5, y: 0.5, z: 0, visibility: o.vis });
  lm[0]  = { x: 0.5, y: o.noseY,  z: 0, visibility: o.vis };   // nose
  lm[11] = { x: 0.45, y: 0.40, z: 0, visibility: o.vis };      // shoulders
  lm[12] = { x: 0.55, y: 0.40, z: 0, visibility: o.vis };
  lm[27] = { x: 0.48, y: o.ankleY, z: 0, visibility: o.vis };  // ankles
  lm[28] = { x: 0.52, y: o.ankleY, z: 0, visibility: o.vis };
  return lm;
}
function fakeTracker(opts) {
  return { detectForVideo: () => ({ landmarks: [pose(opts)] }) };
}
"""


def detect_frames(n, opts="{}", start=1000):
    """Run n detections, spaced past the 100ms throttle, and report what was believed."""
    return run(SCAN + FAKE_POSE + f"""
      tracker.landmarker = fakeTracker({opts});
      let t = {start};
      for (let i = 0; i < {n}; i++) {{ t += 150; detectSubject($('video'), t); }}
      console.log(JSON.stringify({{ seen: subject.seen, streak }}));
    """)


def test_one_good_frame_is_not_enough():
    """A person-shaped object gives you one. This is the false-green fix."""
    out = detect_frames(1)
    assert out["seen"] is False, "believed a single frame"
    assert out["streak"] == 1


def test_three_good_frames_are_believed():
    out = detect_frames(3)
    assert out["seen"] is True, f"still not believed after 3 frames: {out}"


def test_a_low_confidence_pose_is_never_believed():
    """Landmark visibility below the floor means the model is guessing at background clutter."""
    out = detect_frames(6, opts="{ vis: 0.3 }")
    assert out["seen"] is False
    assert out["streak"] == 0, "a rejected frame must reset the streak, not bank it"


def test_a_pose_too_short_to_be_a_standing_person_is_rejected():
    """Head barely above the feet — the model stretched a pose over something that is not a person."""
    out = detect_frames(6, opts="{ noseY: 0.70, ankleY: 0.85 }")
    assert out["seen"] is False


def test_a_realistic_standing_height_is_accepted():
    out = detect_frames(3, opts="{ noseY: 0.30, ankleY: 0.88 }")
    assert out["seen"] is True


def test_a_pose_with_hidden_legs_is_rejected():
    """Legs and torso are checked separately, and this is what the leg check is for.

    A pose with a clear torso but invisible legs is the model guessing where someone's feet are
    behind furniture. Since the marker is a footprint, an invented foot position is exactly the
    wrong thing to trust. Written with a visible torso on purpose: a test that dims *everything*
    passes on the torso check alone and says nothing about the leg check.
    """
    out = run(SCAN + FAKE_POSE + """
      tracker.landmarker = { detectForVideo: () => {
        const lm = pose({ vis: 0.9 });
        lm[27].visibility = 0.2; lm[28].visibility = 0.2;   // ankles hidden
        lm[25].visibility = 0.2; lm[26].visibility = 0.2;   // knees hidden
        return { landmarks: [lm] };
      } };
      let t = 1000;
      for (let i = 0; i < 6; i++) { t += 150; detectSubject($('video'), t); }
      console.log(JSON.stringify({ seen: subject.seen, streak }));
    """)
    assert out["seen"] is False, "trusted a foot position the model could not actually see"


def test_detection_runs_on_a_downscaled_frame_not_the_camera_track():
    """The bug: MediaPipe uploads whatever it is handed as a GPU texture, and the capture track
    went to 4096 wide — which is the maximum texture size on a great many mobile GPUs. On those the
    call throws rather than runs slowly, the catch swallowed it, and detection silently never
    happened. No subject, no green marker, no cues, and nothing on screen saying why.
    """
    out = run(SCAN + FAKE_POSE + """
      let seen = null;
      tracker.landmarker = { detectForVideo: (src) => { seen = src; return { landmarks: [pose()] }; } };
      detectSubject($('video'), 2000);
      // Not an identity check: the DOM stub hands back a fresh object per getElementById, so
      // `seen === $('video')` is false however the page behaves. The widths differ, which is the
      // thing that actually distinguishes them — the stub's video is 390 wide, the canvas is
      // whatever DETECT_WIDTH says.
      console.log(JSON.stringify({
        width: seen && seen.width, expected: DETECT_WIDTH, videoWidth: $('video').width,
      }));
    """)
    assert out["expected"] != out["videoWidth"], (
        "the stub's video happens to be DETECT_WIDTH wide — this test can no longer tell them apart"
    )
    assert out["width"] == out["expected"], (
        f"model was handed a {out['width']}px source, not the {out['expected']}px detection canvas"
    )
    assert out["expected"] <= 640, f"detection frame is {out['expected']}px — wider than pose needs"


def test_a_throwing_model_is_reported_rather_than_swallowed():
    """Silence is what cost us the bug above: there was no way to tell a model that had found
    nobody from one that was failing on every single call."""
    out = run(SCAN + FAKE_POSE + """
      const errors = [];
      console.error = (...a) => errors.push(a.join(' '));
      tracker.landmarker = { detectForVideo: () => { throw new Error('texture too large'); } };
      detectSubject($('video'), 2000);
      detectSubject($('video'), 3000);
      console.log(JSON.stringify({ errors: errors.length, text: errors[0] || '' }));
    """)
    assert out["errors"] == 1, "should log once, not on every frame — this runs 10x a second"
    assert "texture too large" in out["text"]


def test_the_model_is_configured_with_raised_confidence_floors():
    """Asserted statically, not behaviourally: the fake tracker replaces the model entirely, so
    nothing here can exercise MediaPipe's own thresholds. The defaults are 0.5, which is tuned for
    finding a person rather than deciding whether one is present."""
    page = PAGE.read_text(encoding="utf-8")
    for option in ("minPoseDetectionConfidence", "minPosePresenceConfidence", "minTrackingConfidence"):
        match = re.search(option + r":\s*([0-9.]+)", page)
        assert match, f"{option} is not set — the 0.5 default is too eager for this use"
        assert float(match.group(1)) >= 0.7, f"{option} dropped to {match.group(1)}"


# ── the horizon level, driven by real gravity vectors ───────────────────────
#
# onMotion had no tests at all: every cue test sets `liveLean` by hand, which walks straight past
# the function that computes it. That is how it shipped reading pitch as though it were roll, and
# how the sign stayed wrong on iOS.

POSE = """
function pose(rollDeg, pitchDeg) {
  // Gravity as a non-iOS device reports it: a vector pointing along world-up, in device axes.
  // Rotating the phone CLOCKWISE, as the person holding it sees it, swings that vector toward -x.
  const G = 9.81, rad = d => d * Math.PI / 180;
  return { x: -G * Math.sin(rad(rollDeg)) * Math.cos(rad(pitchDeg)),
           y:  G * Math.cos(rad(rollDeg)) * Math.cos(rad(pitchDeg)),
           z:  G * Math.sin(rad(pitchDeg)) };
}
function hold(rollDeg, pitchDeg) {
  onMotion({ accelerationIncludingGravity: pose(rollDeg, pitchDeg) });
  const rot = styles.filter(s => s.id === 'camLevelLine' && s.prop === 'transform');
  const op  = styles.filter(s => s.id === 'camLevelLine' && s.prop === 'opacity');
  const lvl = toggles.filter(t => t.id === 'camLevel' && t.cls === 'level');
  return { angle: rot.length ? parseFloat(rot[rot.length - 1].val.match(/-?[0-9.]+/)[0]) : null,
           opacity: op.length ? op[op.length - 1].val : null,
           level: lvl.length ? lvl[lvl.length - 1].val : null,
           liveLean: Math.round(liveLean) };
}
"""


def test_held_upright_the_line_is_flat_and_green():
    out = run(POSE + """
      console.log(JSON.stringify(hold(0, 0)));
    """)
    assert out["angle"] == 0
    assert out["level"] is True


def test_the_line_lies_along_the_horizon_not_mirrored_against_it():
    """Rotate the phone clockwise and the horizon in the frame swings anticlockwise, so the line
    must too. Get this backwards and the indicator mirrors the real horizon — still zero when level,
    so it looks plausible, but it leans the wrong way exactly when you need it."""
    out = run(POSE + """
      console.log(JSON.stringify({ cw: hold(20, 0), ccw: hold(-20, 0) }));
    """)
    assert out["cw"]["angle"] == pytest.approx(-20, abs=0.2), "clockwise phone, anticlockwise line"
    assert out["ccw"]["angle"] == pytest.approx(20, abs=0.2)
    assert out["cw"]["level"] is False and out["ccw"]["level"] is False


def test_aiming_up_or_down_leaves_the_level_alone():
    """The regression that started this. A level answers one question — is the horizon straight —
    and pitch is not part of it. The reading used to be the angle from world-vertical, so aiming
    down to frame a shot climbed the number and killed the green. camera_tilt actively asks the user
    to aim lower, so the app broke its own indicator by being obeyed."""
    out = run(POSE + """
      console.log(JSON.stringify({ down25: hold(0, 25), down45: hold(0, 45), up30: hold(0, -30) }));
    """)
    for name, row in out.items():
        assert row["angle"] == pytest.approx(0, abs=0.2), f"{name} leaked pitch into the level"
        assert row["level"] is True, f"{name} lost the green while perfectly level"


def test_the_level_band_is_the_one_the_straighten_cue_clears_at():
    """Both sit at 3 degrees on purpose: the line should go green exactly as the cue goes away,
    not a couple of degrees either side of it."""
    out = run(POSE + """
      console.log(JSON.stringify({ inside: hold(2, 0), outside: hold(4, 0) }));
    """)
    assert out["inside"]["level"] is True
    assert out["outside"]["level"] is False


def test_a_phone_lying_flat_hides_the_line_rather_than_claiming_level():
    """Face up, almost no gravity is left in the screen plane, so roll is two noisy numbers handed
    to atan2 — it swings and can read a confident 0. It must also not touch liveLean, or a phone put
    down on a table would satisfy the straighten cue's latch."""
    out = run(POSE + """
      liveLean = 999;                       // sentinel: a flat reading must not overwrite it
      const flat = hold(0, 90);
      console.log(JSON.stringify({ flat, liveLeanAfter: liveLean }));
    """)
    assert out["flat"]["opacity"] == "0", "a meaningless angle must not be drawn as a horizon"
    assert out["flat"]["level"] is False
    assert out["liveLeanAfter"] == 999, "a flat phone quietly satisfied the straighten latch"


@pytest.mark.parametrize("roll", [0, 12, -12, 35])
def test_the_line_is_the_same_whichever_sign_convention_the_phone_uses(roll):
    """iOS reports accelerationIncludingGravity negated compared with everyone else — face-up on a
    table it reads z = -9.8 where Android reads +9.8 — which shifts the computed angle by exactly
    180 degrees. A line drawn at t and at t+180 looks the same, so folding the angle into (-90, 90]
    makes the difference vanish rather than trying to detect the platform.

    Detection was tried first and does not work: DeviceMotionEvent.requestPermission is defined by
    Chromium as well as by Safari, so the feature test that supposedly means "iOS" reports iOS on a
    Windows desktop. This asserts the property instead — same pose, either convention, same line.
    """
    out = run(POSE + f"""
      const p = pose({roll}, 0);
      const angle = () => {{
        const rot = styles.filter(s => s.id === 'camLevelLine' && s.prop === 'transform');
        return parseFloat(rot[rot.length - 1].val.match(/-?[0-9.]+/)[0]);
      }};
      onMotion({{ accelerationIncludingGravity: p }});
      const android = angle();
      onMotion({{ accelerationIncludingGravity: {{ x: -p.x, y: -p.y, z: -p.z }} }});
      const ios = angle();
      console.log(JSON.stringify({{ android, ios }}));
    """)
    assert out["android"] == pytest.approx(out["ios"], abs=0.2), (
        f"the line leans differently on the two conventions: {out}"
    )
    assert out["android"] == pytest.approx(-roll, abs=0.2), "clockwise phone, anticlockwise line"


# ── retake: another shot of the same setup ───────────────────────────────────

ANALYSIS = """
  analysisResult = {
    scene_type: 'Cafe',
    placement: { x: 0.333, y: 0.70, reason: 'light', reason_text: 'Light falls on your face' },
    camera_tilt: { direction: 'down', reason: 'Empty space above — aim a little lower' },
    composition: { horizon: 'Tilted' },
    placement_hint: 'Stand in front of the blue door',
    lighting: { quality: 'Good' },
    hashtags: [],
    filter: 'Vivid',
  };
"""

# Retake is only reachable from the results screen, which is only reachable after the camera has
# started — so the state it actually runs against has a live track. Without this the page is right
# to try to re-acquire the camera, and these tests would be measuring the no-camera path instead.
LIVE_STREAM = """
  hasCamera = true;
  stream = { getVideoTracks: () => [{ readyState: 'live' }] };
"""


def test_retake_restores_the_coaching_state():
    """The whole point: the scene has not changed, so the same marker should come straight back."""
    out = run(ANALYSIS + LIVE_STREAM + """
      motionGranted = true;
      resetCoaching();                 // as stopLive() does on the way to the results screen
      await retake();
      console.log(JSON.stringify({
        coachingActive, standPos, standReason, tiltDirection, tiltFallback,
        needsStraightening, hints,
      }));
    """)
    assert out["coachingActive"] is True
    assert out["standPos"] == {"x": 0.333, "y": 0.70}
    assert out["standReason"] == "Light falls on your face"
    # The direction is kept, not a finished sentence; the cue is phrased from it each frame, which
    # is what lets a language switch after the scan reach it. Read before running a frame, because
    # a frame with the phone held level would latch needsStraightening back off.
    assert out["tiltDirection"] == "down"
    assert out["tiltFallback"] == "Empty space above — aim a little lower"
    assert out["needsStraightening"] is True


def test_retake_costs_no_api_call():
    """The reason it exists. Re-scanning an unchanged scene costs a wait and two OpenAI calls to
    put the marker in exactly the same place."""
    out = run(ANALYSIS + LIVE_STREAM + """
      let fetches = 0;
      globalThis.fetch = () => { fetches++; return Promise.reject(new Error('x')); };
      resetCoaching();
      await retake();
      console.log(JSON.stringify({ fetches }));
    """)
    assert out["fetches"] == 0, "retake must not re-analyse the scene"


# ── the split scan ────────────────────────────────────────────────────────────
#
# The marker comes from /measure and the words from /describe, which took between 3.7 and 11.1
# seconds for the same image. These drive the real scanScene with /describe held open, which is
# the state the user actually stands in.

SPLIT_STUB = """
  const MEASURED = {
    lighting: { quality: 'Good' },
    blurry: false,
    placement: { x: 0.667, y: 0.88, reason: 'light', reason_text: 'Light falls on your face' },
    camera_tilt: { direction: 'ok', reason: '' },
    composition: {},
  };
  const urls = [];
  let settleDescribe, failDescribe;
  globalThis.FormData = class { append(k, v) { (this.sent ||= []).push([k, String(v)]); } };
  fetch = (url) => {
    urls.push(url);
    if (String(url).includes('/measure')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(MEASURED) });
    }
    if (String(url).includes('/describe')) {
      return new Promise((resolve, reject) => {
        settleDescribe = (body) => resolve({ ok: true, json: () => Promise.resolve(body) });
        failDescribe = () => reject(new Error('describe failed'));
      });
    }
    return Promise.reject(new Error('unexpected call to ' + url));
  };
  const later = (fn) => setTimeout(fn, 10);
  // /describe is deliberately left open in some of these, and postScan's 60s abort
  // timer is only cleared when a request settles — so say the answer and go.
  const done = (o) => process.stdout.write(JSON.stringify(o) + '\\n', () => process.exit(0));
"""


def test_the_marker_is_up_before_the_model_answers():
    """The whole reason for the split. /describe is still open when this asserts."""
    out = run(SPLIT_STUB + """
      scanScene().then(() => later(() => done({
        coaching: coachingActive,
        standPos,
        reason: standReason,
        sceneType: analysisResult.scene_type ?? null,
        measured: urls.filter(u => String(u).includes('/measure')).length,
      })));
    """)
    assert out["coaching"] is True, "coaching did not start from /measure alone"
    assert out["standPos"] == {"x": 0.667, "y": 0.88}
    assert out["reason"] == "Light falls on your face"
    assert out["measured"] == 1
    assert out["sceneType"] is None, "the scene name arrived before /describe answered"


def test_the_words_land_when_describe_answers():
    out = run(SPLIT_STUB + """
      scanScene().then(() => later(() => {
        settleDescribe({ scene_type: 'Cafe', placement_hint: 'Stand by the wall',
                         filter: 'Vivid Warm', hashtags: ['#a'] });
        later(() => done({
          scene: analysisResult.scene_type,
          hint: analysisResult.placement_hint,
          filter: analysisResult.filter,
          stillCoaching: coachingActive,
        }));
      }));
    """)
    assert out["scene"] == "Cafe"
    assert out["hint"] == "Stand by the wall"
    assert out["filter"] == "Vivid Warm"
    assert out["stillCoaching"] is True


def test_a_describe_that_fails_leaves_the_coaching_alone():
    """A failed vision call costs the scene name and the hint. It must not cost the marker, which
    never needed the model in the first place."""
    out = run(SPLIT_STUB + """
      scanScene().then(() => later(() => {
        failDescribe();
        later(() => done({
          coaching: coachingActive,
          standPos,
          scene: analysisResult.scene_type ?? null,
        }));
      }));
    """)
    assert out["coaching"] is True
    assert out["standPos"] == {"x": 0.667, "y": 0.88}
    assert out["scene"] is None


def test_a_late_describe_cannot_paint_over_a_newer_scan():
    """Rescan while the first /describe is in flight and let it answer afterwards. Without the
    token it writes the old scene's hint over the one you are now pointing at."""
    out = run(SPLIT_STUB + """
      scanScene().then(() => later(() => {
        scanToken += 1;                       // as a rescan does
        settleDescribe({ scene_type: 'Stale', placement_hint: 'somewhere else' });
        later(() => done({
          scene: analysisResult.scene_type ?? null,
          hint: analysisResult.placement_hint ?? null,
        }));
      }));
    """)
    assert out["scene"] is None, "a stale response overwrote the current scan"
    assert out["hint"] is None


def test_describe_is_asked_for_exactly_once_per_scan():
    out = run(SPLIT_STUB + """
      scanScene().then(() => later(() => done({
        described: urls.filter(u => String(u).includes('/describe')).length,
      })));
    """)
    assert out["described"] == 1


def test_the_scene_badge_stays_empty_until_the_name_arrives():
    """It fell back to the word "Scene", which was fine when the field always arrived with
    everything else. Split apart it would read "Scene" for seconds and then swap."""
    page = PAGE.read_text(encoding="utf-8")
    badge = re.search(r"\$\('sceneBadgeTop'\)\.textContent = ([^;]+);", page)
    assert badge, "the scene badge assignment moved"
    assert "res.scene" not in badge.group(1), (
        f"the badge still falls back to a placeholder: {badge.group(1)}"
    )


# ── share vs save ────────────────────────────────────────────────────────────

SAVE_STUB = """
  capturedImg = { naturalWidth: 100, naturalHeight: 100 };
  analysisResult = { hashtags: ['#a', '#b'] };
  const calls = [];
  let downloaded = 0, toasts = [];
  globalThis.File = function (parts, name, opts) { return { name, type: opts && opts.type }; };
  globalThis.URL = { createObjectURL: () => 'blob:x', revokeObjectURL(){} };
  document.createElement = (t) => ({
    style: {}, classList: { add(){}, remove(){}, toggle(){} },
    getContext: () => ({ drawImage(){}, getImageData: () => ({ data: [] }), putImageData(){} }),
    toBlob: (cb) => cb({}), click: () => { if (t === 'a') downloaded++; },
    remove(){}, appendChild(){}, addEventListener(){},
  });
  document.body = { appendChild(){} };
  toast = (m) => toasts.push(m);
  // Share writes the hashtags here instead of attaching them to the photo.
  const copied = [];
  navigator.clipboard = { writeText: (t) => { copied.push(t); return Promise.resolve(); } };
"""

WITH_SHEET = """
  // The harness binds `navigator` as a const, so it is extended rather than replaced.
  navigator.canShare = () => true;
  navigator.share = (o) => { calls.push(o); return Promise.resolve(); };
"""

NO_SHEET = """
  navigator.canShare = undefined;
  navigator.share = undefined;
"""


def test_neither_button_attaches_text_to_the_photo():
    """Both buttons open the system sheet on a phone, because no web page can write to the camera
    roll on either mobile OS — the sheet's "Save Image" is the only route there.

    Neither sends text with the file. Attaching `text` as well makes iOS treat the words as the
    item being shared: the sheet shows a generic file placeholder instead of the picture, and some
    targets take the text and drop the image. Sharing a photo should look like sharing a photo.
    """
    out = run(SAVE_STUB + WITH_SHEET + """
      savePhoto()
        .then(() => sharePhoto())
        .then(() => console.log(JSON.stringify({
          save: { files: calls[0].files.length, text: calls[0].text || null },
          share: { files: calls[1].files.length, text: calls[1].text || null },
        })));
    """)
    assert out["save"]["files"] == 1 and out["save"]["text"] is None, "Save should attach no text"
    assert out["share"]["files"] == 1, "Share should attach the photo"
    assert out["share"]["text"] is None, (
        "Share attached text again — the sheet will fall back to a file placeholder"
    )


def test_share_puts_the_hashtags_on_the_clipboard_instead():
    """They are not dropped, just moved off the share payload: one paste into the caption field
    wherever the photo lands. Written before share() is awaited, because the clipboard needs the
    user gesture that is still live at that point — spending it on the sheet first loses it."""
    out = run(SAVE_STUB + WITH_SHEET + """
      sharePhoto().then(() => console.log(JSON.stringify({ copied, toasts })));
    """)
    assert out["copied"] == ["#a #b"], f"hashtags were not copied: {out['copied']}"
    assert out["toasts"] == ["Hashtags copied — paste them into the caption"], (
        f"nothing told the user where the hashtags went: {out['toasts']}"
    )


def test_save_does_not_touch_the_clipboard():
    """Save is for keeping, not posting. Overwriting the clipboard of someone who only wanted the
    photo is a side effect they did not ask for."""
    out = run(SAVE_STUB + WITH_SHEET + """
      savePhoto().then(() => console.log(JSON.stringify({ copied })));
    """)
    assert out["copied"] == [], f"Save clobbered the clipboard with {out['copied']}"


def test_save_falls_back_to_a_real_download_without_a_share_sheet():
    """Desktop. This is the only path where a file genuinely lands on disk."""
    out = run(SAVE_STUB + NO_SHEET + """
      savePhoto().then(() => console.log(JSON.stringify({ downloaded, toasts })));
    """)
    assert out["downloaded"] == 1, "nothing was written"
    assert out["toasts"] == ["Saved"]


def test_it_does_not_claim_to_have_saved_when_it_handed_off_to_the_sheet():
    """The bug this fixes said "Saved" unconditionally, including on a phone where the download
    attribute silently does nothing. Claiming success we cannot verify is worse than staying quiet:
    the OS gives its own confirmation when the sheet completes."""
    out = run(SAVE_STUB + WITH_SHEET + """
      savePhoto().then(() => console.log(JSON.stringify({ toasts })));
    """)
    assert out["toasts"] == [], f"claimed {out['toasts']} without writing anything"


def test_the_saved_file_is_named_per_shot():
    """A folder of daka.jpg, daka(1).jpg is nobody's idea of a keeper — but a run-together
    timestamp reads as a serial number, so it is spaced and punctuated like a screenshot while
    still sorting chronologically.

    Only Files and desktop downloads ever see this: saving through the sheet to Photos hands the
    name to iOS, which discards it.
    """
    out = run("console.log(JSON.stringify({ a: filename() }));")
    name = out["a"]
    assert re.fullmatch(r"DaKaBa \d{4}-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2}\.jpg", name), name
    # A colon is illegal in a filename on every platform that matters, and the OS would silently
    # rewrite it — so the time is separated by dots.
    assert ":" not in name


def test_retake_does_nothing_without_an_analysis():
    """Defensive: the button is only reachable from the results screen, but a stale tap or a
    restored session should not put the app into coaching with no marker to show."""
    out = run("""
      analysisResult = null;
      await retake();
      console.log(JSON.stringify({ coachingActive, standPos }));
    """)
    assert out["coachingActive"] is False
    assert out["standPos"] is None


def test_retake_restarts_a_suspended_video_element():
    """A <video> inside a display:none screen is suspended, and autoplay does not re-fire when the
    screen comes back. Nothing detaches the stream, so this looked like it should work — but the
    element stays paused, videoWidth stays 0, and liveLoop returns on its first line every frame.
    The viewfinder never starts and nothing is logged."""
    out = run(ANALYSIS + LIVE_STREAM + """
      await retake();
      console.log(JSON.stringify({ played, coachingActive }));
    """)
    assert "video" in out["played"], "the viewfinder was never told to play"
    assert "videoBackdrop" in out["played"], "the blurred surround was left frozen"
    assert out["coachingActive"] is True


def test_retake_reacquires_a_camera_ios_released():
    """The "sometimes". If the page loses visibility while the camera screen is hidden, iOS ends
    the track outright — readyState "ended", never recovering on its own. Playing the element
    again achieves nothing; the only fix is asking for the camera a second time."""
    out = run(ANALYSIS + """
      hasCamera = true;
      stream = { getVideoTracks: () => [{ readyState: 'ended' }] };
      let asked = 0;
      navigator.mediaDevices.getUserMedia = () => {
        asked++;
        return Promise.resolve({ getVideoTracks: () => [{ readyState: 'live' }] });
      };
      await retake();
      console.log(JSON.stringify({ asked, coachingActive }));
    """)
    assert out["asked"] == 1, "an ended track must be re-acquired, not replayed"
    assert out["coachingActive"] is True, "coaching should resume on the new stream"


def test_the_camera_screen_is_shown_before_the_controls_are_measured():
    """setCamState ends in resizeControls, which measures the controls bar to animate its height.
    A measurement taken while the screen is still display:none comes back 0, resizeControls bails,
    and the bar keeps `height: auto` — so the first state change afterwards has no pinned height to
    animate FROM. That is why the frame only started gliding from the second scan onwards.

    Ordering is the whole fix, so ordering is what this pins."""
    out = run("""
      navigator.mediaDevices.getUserMedia = () =>
        Promise.resolve({ getVideoTracks: () => [{ readyState: 'live' }] });
      // The page writes to both recorders as it loads; only this call's ordering is in question.
      toggles.length = 0; styles.length = 0;
      await startCamera();
      const shown = (toggles.find(t => t.id === 'camera' && t.cls === 'active' && t.val) || {}).seq;
      const measured = (styles.find(s => s.id === 'camIdle' && s.prop === 'display') || {}).seq;
      console.log(JSON.stringify({ shown, measured }));
    """)
    assert out["shown"] is not None and out["measured"] is not None, (
        "startCamera did not reach either step"
    )
    assert out["shown"] < out["measured"], (
        "setCamState ran while the camera screen was still hidden, so the controls bar was "
        "measured at zero height and the first transition cannot animate"
    )


def frame_transforms(states):
    """The transform written to the frame after each of the given camera states, in order."""
    steps = "".join(
        f"setCamState('{s}'); out.push(styles.filter(x => x.id === 'camFrame' "
        "&& x.prop === 'transform').pop()); "
        for s in states
    )
    out = run("const out = [];" + steps + """
      console.log(JSON.stringify({ t: out.map(s => (s && s.val) || '') }));
    """)
    return out["t"]


def test_the_frame_travels_to_the_centre_while_analysing():
    """Top-middle to compose, centre of the screen while the scan is in flight, top-middle again to
    coach. With a 520px frame in an 844px screen resting at top 80, centred means top 162 — a shift
    of 82."""
    idle, loading, tips = frame_transforms(["idle", "loading", "tips"])
    assert idle == "", "the composing state should sit where layout puts it"
    assert re.fullmatch(r"translateY\(82px\)", loading), (
        f"analysing did not centre the frame: {loading!r}"
    )
    assert tips == "", f"coaching should return to the top-middle position: {tips!r}"


def test_the_frame_does_not_drift_across_repeated_transitions():
    """placeFrame measures where the frame is now, which already includes however far it has been
    pushed. Subtracting the current shift is what recovers the natural position — without it a
    second centring measures from the centred position, decides it is already there, and drops the
    frame back to the top mid-analysis.

    Found by mutation testing: the straight idle/loading/tips path cannot see this, because the
    shift is only non-zero on the one step that sets it.
    """
    t = frame_transforms(["idle", "loading", "loading", "tips", "loading"])
    assert t[1] == "translateY(82px)", f"first centring: {t[1]!r}"
    assert t[2] == "translateY(82px)", (
        f"the frame moved on a second centring instead of staying put: {t[2]!r}"
    )
    assert t[3] == "", f"coaching should return to the top: {t[3]!r}"
    assert t[4] == "translateY(82px)", f"centring after a round trip drifted: {t[4]!r}"


def test_the_frame_moves_by_transform_not_by_layout():
    """A live <video> re-rasterises when its box changes, which is what made the earlier
    height-driven move stutter and flash black. Transform is composited and touches nothing
    inside the frame, so it must stay the mechanism."""
    html = PAGE.read_text(encoding="utf-8")
    rule = re.search(r"\.cam-frame\s*\{([^}]*)\}", html).group(1)
    assert re.search(r"transition:[^;]*transform", rule), "the frame's travel is not animated"
    for banned in ("transition: top", "transition: margin"):
        assert banned not in rule, f"{banned} moves the frame by relayout"
    placed = re.search(r"function placeFrame\(centred\) \{(.*?)\n    \}", html, re.S).group(1)
    assert "translateY" in placed, "placeFrame no longer moves the frame by transform"


# ── tip jar ─────────────────────────────────────────────────────────────────
#
# In-app purchase is a store transaction, so this is native-only by necessity — the same wall the
# camera roll runs into. The web build must be completely untouched by it.

RC_STUB = """
  const rc = { configured: [], purchased: [] };
  window.Capacitor = {
    isNativePlatform: () => true,
    Plugins: { Purchases: {
      configure: (o) => { rc.configured.push(o); return Promise.resolve(); },
      getOfferings: () => Promise.resolve({ current: { availablePackages: [
        { identifier: 'tip_small', product: { title: 'Small tip', priceString: '£1.99' } },
        { identifier: 'tip_large', product: { title: 'Large tip', priceString: '£4.99' } },
      ] } }),
      purchasePackage: (o) => { rc.purchased.push(o); return Promise.resolve({}); },
    } },
  };
"""


def test_the_tip_jar_stays_hidden_on_the_web():
    """dakaba.pages.dev has no native bridge and cannot make a store purchase. Offering a tip that
    cannot complete is worse than not asking, so the button never appears and none of this runs."""
    out = run("""
      initTipJar().then(() => console.log(JSON.stringify({
        shown: hiddenSet.filter(h => h.id === 'tipBtn'),
      })));
    """)
    assert out["shown"] == [], "the tip button was touched on a platform that cannot buy anything"


def test_the_tip_jar_stays_hidden_without_a_key():
    """A checkout of this repo has no RevenueCat key in it. That must degrade to no button, not to
    a configure() call that throws."""
    out = run(RC_STUB + """
      RC_API_KEY = '';          // as a fork or a fresh checkout would have it
      initTipJar().then(() => console.log(JSON.stringify({
        configured: rc.configured.length,
        shown: hiddenSet.filter(h => h.id === 'tipBtn'),
      })));
    """)
    assert out["configured"] == 0, "configured RevenueCat with an empty key"
    assert out["shown"] == []


def test_the_tip_button_appears_only_once_there_is_something_to_buy():
    """Shown after RevenueCat answers with real packages — not on configure, and not optimistically
    before the offerings come back."""
    out = run(RC_STUB + """
      RC_API_KEY = 'test_fake';
      initTipJar().then(() => console.log(JSON.stringify({
        key: rc.configured[0].apiKey,
        shown: hiddenSet.filter(h => h.id === 'tipBtn').map(h => h.v),
      })));
    """)
    assert out["key"] == "test_fake"
    assert out["shown"] == [False], f"tip button visibility: {out['shown']}"


def test_an_empty_offering_leaves_the_button_hidden():
    out = run("""
      RC_API_KEY = 'test_fake';
      window.Capacitor = { isNativePlatform: () => true, Plugins: { Purchases: {
        configure: () => Promise.resolve(),
        getOfferings: () => Promise.resolve({ current: { availablePackages: [] } }),
      } } };
      initTipJar().then(() => console.log(JSON.stringify({
        shown: hiddenSet.filter(h => h.id === 'tipBtn'),
      })));
    """)
    assert out["shown"] == [], "offered a tip with no packages behind it"


def test_a_failing_revenuecat_never_breaks_the_app():
    """A tip jar that takes the camera down with it would be a poor trade."""
    out = run("""
      // Without a key initTipJar returns before it reaches configure(), which made an earlier
      // version of this test pass against code that had no error handling at all.
      RC_API_KEY = 'test_fake';
      window.Capacitor = { isNativePlatform: () => true, Plugins: { Purchases: {
        configure: () => Promise.reject(new Error('network')),
      } } };
      initTipJar()
        .then(() => console.log(JSON.stringify({ ok: true, shown: hiddenSet.filter(h => h.id === 'tipBtn') })))
        .catch(e => console.log(JSON.stringify({ ok: false, err: String(e) })));
    """)
    assert out["ok"] is True, f"initTipJar rejected instead of degrading: {out.get('err')}"
    assert out["shown"] == []


def paywall_returning(result, extra=""):
    """Drive openTip() with a paywall that answers `result`, and report what happened after."""
    return run(RC_STUB + f"""
      toast = (m) => {{ (globalThis.toasts = globalThis.toasts || []).push(m); }};
      let presented = 0, sheetShown = 0;
      window.Capacitor.Plugins.RevenueCatUI = {{
        presentPaywall: () => {{ presented++; return Promise.resolve({{ result: '{result}' }}); }},
      }};
      tipOffering = {{ identifier: 'default', paywall: {{ template: 'x' }} }};
      const realSheet = showTipSheet;
      showTipSheet = () => {{ sheetShown++; realSheet(); }};
      {extra}
      openTip().then(() => console.log(JSON.stringify({{
        presented, sheetShown, toasts: globalThis.toasts || [],
      }})));
    """)


@pytest.mark.parametrize("result", ["PURCHASED", "RESTORED"])
def test_a_paywall_purchase_is_thanked_and_stops_there(result):
    out = paywall_returning(result)
    assert out["presented"] == 1
    assert out["sheetShown"] == 0, "showed our own sheet on top of a completed purchase"
    assert out["toasts"] == ["Thank you so much"]


def test_a_cancelled_paywall_says_nothing_and_shows_no_sheet():
    """Cancelling is a decision. Answering it by immediately presenting a second way to pay is
    the app arguing with them."""
    out = paywall_returning("CANCELLED")
    assert out["toasts"] == [], f"complained about a cancellation: {out['toasts']}"
    assert out["sheetShown"] == 0, "re-asked after they said no"


@pytest.mark.parametrize("result", ["NOT_PRESENTED", "ERROR"])
def test_no_paywall_falls_back_to_our_own_sheet(result):
    """NOT_PRESENTED is the state this project is in until a paywall is designed in the dashboard,
    and may be permanent on a Test Store offering. Treating it as a failure would leave the button
    dead; the sheet is built from the same offering and needs nothing configured."""
    out = paywall_returning(result)
    assert out["sheetShown"] == 1, f"{result} left the tip button doing nothing"


def test_a_throwing_paywall_still_reaches_the_sheet():
    out = run(RC_STUB + """
      let sheetShown = 0;
      window.Capacitor.Plugins.RevenueCatUI = {
        presentPaywall: () => Promise.reject(new Error('no ui')),
      };
      tipOffering = { identifier: 'default', paywall: { template: 'x' } };
      const realSheet = showTipSheet;
      showTipSheet = () => { sheetShown++; realSheet(); };
      openTip().then(() => console.log(JSON.stringify({ sheetShown })));
    """)
    assert out["sheetShown"] == 1, "a broken paywall took the tip jar down with it"


def test_an_offering_with_no_paywall_goes_straight_to_the_sheet():
    """presentPaywall on an offering with no paywall designed does not return NOT_PRESENTED — it
    never settles, so awaiting it hangs and the fallback is never reached. Tapping the tip button
    did nothing at all on a device, silently, with a green suite.

    So the paywall is only ever asked for when the offering actually carries one.
    """
    out = run(RC_STUB + """
      let presented = 0, sheetShown = 0;
      window.Capacitor.Plugins.RevenueCatUI = {
        presentPaywall: () => { presented++; return new Promise(() => {}); },   // never settles
      };
      tipOffering = { identifier: 'default' };        // no paywall attached
      const realSheet = showTipSheet;
      showTipSheet = () => { sheetShown++; realSheet(); };
      openTip().then(() => console.log(JSON.stringify({ presented, sheetShown })));
    """)
    assert out["presented"] == 0, "asked for a paywall that does not exist"
    assert out["sheetShown"] == 1, "the tip button did nothing"


def test_the_sheet_is_unhidden_before_it_is_opened():
    """An element at display:none cannot transition, so clearing `hidden` and adding the open class
    in the same task makes the card appear in place instead of rising. The order, and the reflow
    between them, is the animation."""
    out = run(RC_STUB + """
      showTipSheet();
      const unhide = hiddenSet.find(h => h.id === 'tipSheet' && h.v === false);
      const opened = toggles.find(t => t.id === 'tipSheet' && t.cls === 'open' && t.val);
      console.log(JSON.stringify({ unhide: !!unhide, opened: !!opened }));
    """)
    assert out["unhide"], "the sheet was never unhidden"
    assert out["opened"], "the sheet was never opened, so it cannot have animated"


def test_closing_animates_out_before_hiding():
    """Hiding immediately would make the card vanish rather than slide away. The class comes off
    first; `hidden` follows once the transition has had time to run."""
    out = run(RC_STUB + """
      showTipSheet();
      closeTipSheet();
      const closedNow = hiddenSet.filter(h => h.id === 'tipSheet' && h.v === true).length;
      await new Promise(r => setTimeout(r, 400));
      const closedAfter = hiddenSet.filter(h => h.id === 'tipSheet' && h.v === true).length;
      const unopened = toggles.filter(t => t.id === 'tipSheet' && t.cls === 'open' && !t.val).length;
      console.log(JSON.stringify({ closedNow, closedAfter, unopened }));
    """)
    assert out["unopened"] == 1, "the open class was never removed, so nothing animates out"
    assert out["closedNow"] == 0, "hidden immediately — the card vanishes instead of sliding"
    assert out["closedAfter"] == 1, "the sheet never actually hid after the animation"


def test_the_close_is_not_left_to_an_event_reduced_motion_suppresses():
    """transitionend never fires when the transition has been removed, which is exactly what
    prefers-reduced-motion does here — the sheet would stay on screen permanently for the people
    least able to dismiss it comfortably."""
    html = PAGE.read_text(encoding="utf-8")
    fn = re.search(r"function closeTipSheet\(\) \{(.*?)\n    \}", html, re.S).group(1)
    assert "setTimeout" in fn, "the close depends on an event reduced motion suppresses"
    assert "transitionend" not in fn


def test_the_tip_package_prefers_the_product_named_tip():
    """Matching the product id first lets the dashboard rename the package without breaking this;
    the lifetime fallback means a default offering works before anyone has renamed anything."""
    out = run("""
      const byId = pickTipPackage([
        { identifier: 'x', packageType: 'LIFETIME', product: { identifier: 'other' } },
        { identifier: 'y', product: { identifier: 'tip' } },
      ]);
      const byType = pickTipPackage([
        { identifier: 'a', product: { identifier: 'nope' } },
        { identifier: 'b', packageType: 'LIFETIME', product: { identifier: 'whatever' } },
      ]);
      const empty = pickTipPackage([]);
      console.log(JSON.stringify({ byId: byId.identifier, byType: byType.identifier, empty }));
    """)
    assert out["byId"] == "y", "product id 'tip' should win over package type"
    assert out["byType"] == "b", "no 'tip' product should fall back to the lifetime package"
    assert out["empty"] is None, "an empty offering must yield nothing to sell"


def test_nothing_anywhere_reads_an_entitlement():
    """The app is free and stays free. A tip buys a thank you and nothing else — no gate, no
    dakaba_pro, no branch that asks whether you have paid."""
    # Comments stripped first: this is a claim about what the code does, and the comments
    # explaining why nothing is gated necessarily use the words being banned.
    code = re.sub(r"/\*.*?\*/", "", page_script(), flags=re.S)
    code = re.sub(r"(?m)^\s*//.*$", "", code)
    code = re.sub(r"\s//[^\n\"']*$", "", code, flags=re.M)
    for banned in ("entitlements", "dakaba_pro", "isPro", "hasActive"):
        assert banned not in code, f"{banned!r} is read in code — something is being gated"


def test_a_cancelled_tip_says_nothing():
    """Cancelling is a choice, not a fault. Scolding someone for deciding not to tip is the fastest
    way to make them regret opening it."""
    out = run(RC_STUB + """
      toast = (m) => { (globalThis.toasts = globalThis.toasts || []).push(m); };
      window.Capacitor.Plugins.Purchases.purchasePackage =
        () => Promise.reject({ code: 'PURCHASE_CANCELLED' });
      buyTip({ identifier: 'tip_small' }, null)
        .then(() => console.log(JSON.stringify({ toasts: globalThis.toasts || [] })));
    """)
    assert out["toasts"] == [], f"complained about a cancelled purchase: {out['toasts']}"


def test_a_completed_tip_thanks_them():
    out = run(RC_STUB + """
      toast = (m) => { (globalThis.toasts = globalThis.toasts || []).push(m); };
      buyTip({ identifier: 'tip_small' }, null)
        .then(() => console.log(JSON.stringify({
          toasts: globalThis.toasts || [], purchased: rc.purchased.length,
        })));
    """)
    assert out["purchased"] == 1
    assert out["toasts"] == ["Thank you so much"]


def test_a_scan_and_a_retake_produce_the_same_coaching_state():
    """Both go through beginCoaching, so they cannot drift apart. This is why it was extracted."""
    out = run(ANALYSIS + LIVE_STREAM + """
      motionGranted = true;
      beginCoaching(analysisResult);
      const afterScan = { standPos: { ...standPos }, standReason, tiltDirection, tiltFallback, needsStraightening, hint: hints[hints.length - 1] };
      resetCoaching();
      await retake();
      const afterRetake = { standPos: { ...standPos }, standReason, tiltDirection, tiltFallback, needsStraightening, hint: hints[hints.length - 1] };
      console.log(JSON.stringify({ afterScan, afterRetake }));
    """)
    assert out["afterScan"] == out["afterRetake"]
