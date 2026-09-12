"""The colour grades, measured rather than eyeballed.

Warm and cool are a white-balance change: red up and blue down, or the reverse. CSS filter
functions cannot express one — `hue-rotate` leaves a neutral completely untouched, because grey has
no hue to rotate, and `sepia` only ever goes warm while desaturating as it goes. The grades used
both, so the cool ones tinted nothing at all and 'Vivid Warm' came out *less* warm than 'Vivid'.

The tint is now a channel gain, applied by an SVG colour matrix in the preview and by `tint()` when
baking the saved file. These tests run the page's own baking maths over real colours, so the
numbers here are exactly what a saved photo gets.
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

PIPELINE = ["applyFilters", "tint", "brightness", "contrast", "saturate", "sepia",
            "grayscale", "hueRotate"]


def _page():
    return PAGE.read_text(encoding="utf-8")


def filter_defs():
    block = re.search(r"const FILTER_CSS = \{(.*?)\n    \};", _page(), re.S)
    assert block, "FILTER_CSS not found"
    return dict(re.findall(r"'([^']+)':\s*'([^']+)'", block.group(1)))


def function_source(js, name):
    """Extract one function by matching braces.

    Not by regex: several of these are written on a single line, so a pattern that stops at the
    next line consisting of `    }` runs straight past them and swallows the five that follow.
    That duplicates declarations, which an ES module rejects outright.
    """
    start = js.index("function " + name + "(")
    depth, i = 0, js.index("{", start)
    while i < len(js):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[start:i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces reading {name}")


def grade(colours):
    """Run the page's real baking pipeline over `colours`, returning the graded results."""
    page = _page()
    js = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    src = "\n".join(function_source(js, name) for name in PIPELINE)
    tint_map = re.search(r"const TINT = \{[^}]*\};", page).group(0)

    harness = f"""
{tint_map}
{src}
const defs = {json.dumps(filter_defs())};
const colours = {json.dumps(colours)};
const out = {{}};
for (const [name, css] of Object.entries(defs)) {{
  out[name] = {{}};
  for (const [label, rgb] of Object.entries(colours)) {{
    const d = new Uint8ClampedArray([rgb[0], rgb[1], rgb[2], 255]);
    applyFilters(d, css);
    out[name][label] = [d[0], d[1], d[2]];
  }}
}}
console.log(JSON.stringify(out));
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "grade.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run(["node", str(path)], capture_output=True, text=True,
                                encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr.strip()[:1500]
    return json.loads(result.stdout.strip().splitlines()[-1])


NEUTRAL = {"grey": [128, 128, 128], "wall": [200, 200, 200]}
SKIN = {"light": [232, 190, 172], "deep": [140, 100, 78]}

WARM = ["Vivid Warm", "Dramatic Warm"]
COOL = ["Vivid Cool", "Dramatic Cool"]
NEUTRAL_GRADES = ["Vivid", "Dramatic", "Noir"]


def tint_of(rgb):
    """Red minus blue on a neutral: the honest measure of white balance. 0 means no tint."""
    return rgb[0] - rgb[2]


@pytest.mark.parametrize("name", WARM)
def test_warm_grades_actually_tint_warm(name):
    """The bug this replaced: sepia desaturated instead of warming, so a grey wall shifted by 12
    while plain Vivid — with no warm grade at all — measured warmer overall."""
    graded = grade(NEUTRAL)[name]
    for label, rgb in graded.items():
        assert tint_of(rgb) >= 12, f"{name} barely warms a neutral {label}: {rgb}"


@pytest.mark.parametrize("name", COOL)
def test_cool_grades_actually_tint_cool(name):
    """The worse bug: hue-rotate cannot touch a neutral, so these tinted a grey wall by exactly
    zero. They were not subtle, they were absent."""
    graded = grade(NEUTRAL)[name]
    for label, rgb in graded.items():
        assert tint_of(rgb) <= -10, f"{name} does not cool a neutral {label}: {rgb}"


@pytest.mark.parametrize("name", NEUTRAL_GRADES)
def test_the_untinted_grades_stay_untinted(name):
    """Vivid, Dramatic and Noir were judged good as they were. A colour cast creeping into them
    would be a regression against that judgement, not an improvement."""
    graded = grade(NEUTRAL)[name]
    for label, rgb in graded.items():
        assert abs(tint_of(rgb)) <= 3, f"{name} has picked up a cast on {label}: {rgb}"


def test_the_tint_is_noticeable_but_not_a_costume():
    """Subtle and noticeable are both requirements. Silvertone — which was judged fine — tints a
    neutral by +15, so that is the reference for tasteful; past about 35 a grade stops looking like
    white balance and starts looking like a novelty."""
    graded = grade(NEUTRAL)
    for name in WARM + COOL:
        shift = abs(tint_of(graded[name]["grey"]))
        assert 12 <= shift <= 35, f"{name} shifts grey by {shift}, outside the tasteful band"


@pytest.mark.parametrize("name", WARM + COOL)
def test_skin_survives_the_grade(name):
    """A white-balance shift strong enough to be seen is also strong enough to make people look
    ill. Skin has to stay in a believable range: still clearly skin, never grey or green."""
    graded = grade(SKIN)[name]
    for label, (r, g, b) in graded.items():
        assert r > g > b, f"{name} broke the red>green>blue order skin needs on {label}: {r},{g},{b}"
        assert r - b >= 20, f"{name} flattened {label} skin toward grey: {r},{g},{b}"


def test_warm_is_warmer_than_its_own_base_grade():
    """'Vivid Warm' used to measure *less* warm than plain 'Vivid', because sepia desaturated the
    reds it was supposed to be lifting. A warm variant that is cooler than its base is just wrong.
    """
    graded = grade(NEUTRAL)
    assert tint_of(graded["Vivid Warm"]["grey"]) > tint_of(graded["Vivid"]["grey"])
    assert tint_of(graded["Dramatic Warm"]["grey"]) > tint_of(graded["Dramatic"]["grey"])
    assert tint_of(graded["Vivid Cool"]["grey"]) < tint_of(graded["Vivid"]["grey"])
    assert tint_of(graded["Dramatic Cool"]["grey"]) < tint_of(graded["Dramatic"]["grey"])


def test_every_tint_reference_resolves():
    """A url() in a grade needs three things to line up: the SVG filter the preview uses, the TINT
    entry the baking path uses, and the same id in both. Miss the SVG and the preview silently
    renders ungraded; miss the TINT entry and the saved file does. Neither errors."""
    page = _page()
    used = set()
    for css in filter_defs().values():
        used |= set(re.findall(r"url\((#[\w-]+)\)", css))
    assert used, "no tint references found — did the grades change shape?"

    tint_block = re.search(r"const TINT = \{([^}]*)\};", page).group(1)
    declared = set(re.findall(r"'(#[\w-]+)'", tint_block))
    svg_ids = {"#" + i for i in re.findall(r'<filter id="([\w-]+)"', page)}

    assert used <= declared, f"no TINT entry for {used - declared} — saved photos would be ungraded"
    assert used <= svg_ids, f"no SVG filter for {used - svg_ids} — the preview would be ungraded"
