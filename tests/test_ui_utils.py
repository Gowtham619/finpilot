"""Node-based unit tests for the pure UI helpers (static/util.js, static/tour.js)."""
import json
import shutil
import subprocess

import pytest
from finpilot import api

STATIC = api.HERE / "static"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_js(code: str):
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, cwd=str(STATIC), timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def U(expr: str):
    return run_js(f"const U=require('./util.js');console.log(JSON.stringify({expr}))")


def test_fmt_date_has_no_timezone_shift():
    code = "const U=require('./util.js');console.log(JSON.stringify(['2026-03-01','2026-12-31','2027-01-09'].map(U.fmtDate)))"
    for tz in ("UTC", "Pacific/Kiritimati", "America/Los_Angeles"):
        r = subprocess.run(["node", "-e", code], capture_output=True, text=True, cwd=str(STATIC),
                           env={"TZ": tz, "PATH": "/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin"})
        assert json.loads(r.stdout) == ["Mar 1", "Dec 31", "Jan 9"], tz


def test_weekday_alignment_several_months():
    # Mon=0..Sun=6; 2026-11-01 is a Sunday, 2026-06-01 a Monday, 2026-02-01 a Sunday, 2026-10-20 a Tuesday
    got = U("['2026-11-01','2026-06-01','2026-02-01','2026-10-20','2024-02-29','2026-01-01'].map(U.weekdayMon)")
    assert got == [6, 0, 6, 1, 3, 3]


def test_worst_label_ordering():
    assert U("U.worstLabel([{label:'clear'},{label:'rain'},{label:'cloudy'}])") == "rain"
    assert U("U.worstLabel([{label:'storm'},{label:'rain'}])") == "storm"
    assert U("U.worstLabel([])") == "clear"
    assert U("U.worstLabel([{label:'cloudy'},{label:'clear'}])") == "cloudy"


def test_pct_below_one_percent():
    assert U("[0,0.003,0.004,0.005,0.0049,0.5,1].map(U.pct)") == ["0%", "<1%", "<1%", "1%", "<1%", "50%", "100%"]


def test_delta_pts_omits_tiny_changes():
    assert U("U.deltaPts(0.304,0.30)") is None
    assert U("U.deltaPts(0.5,0.5)") is None
    assert U("U.deltaPts(0.5,0.3)") == "−20 pts"
    assert U("U.deltaPts(0.3,0.4)") == "+10 pts"


def test_indian_rupee_grouping():
    got = U("[0,999,1000,12345,100000,1234567,12345678,-45000].map(U.money)")
    assert got == ["₹0", "₹999", "₹1,000", "₹12,345", "₹1,00,000", "₹12,34,567", "₹1,23,45,678", "−₹45,000"]


def test_add_days_and_round():
    assert U("U.addDays('2026-12-20',56)") == "2027-02-14"
    assert U("[U.roundTo(61234,1000),U.roundTo(100,1000)]") == [61000, 1000]


def test_tour_runner_semantics_with_fake_timers():
    r = subprocess.run(["node", str(api.HERE.parent / "tests" / "js" / "tour_runner_check.js")],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    o = res["order"]
    assert o["s0"] == 0 and o["s1"] == 0 and o["s2"] == 1 and o["mid"] == 0 and o["ends"] == [["done", None]]
    assert res["pause"] == {"stillFirst": 0, "before": 0, "after": 1}
    assert res["skip"]["index"] == 1
    assert res["skipPaused"] == {"index": 1, "paused": False}
    assert res["stop"] == {"ran2": False, "ends": [["stopped", None]], "state": "stopped"}
    assert res["error"] == {"ran2": False, "ends": [["error", "boom"]], "state": "error"}
    assert res["abort"]["ends"] == [["stopped", None]]
    assert res["reduced"]["m"] == 0 and res["reduced"]["ends"] == [["done", None]]


def test_tour_ui_elements_and_no_autoplay():
    html = (api.HERE / "templates" / "index.html").read_text()
    js = (STATIC / "tour.js").read_text()
    for i in ("tourBtn", "tourBar", "tourCap", "tourPause", "tourSkip", "tourStop", "tourReplay"):
        assert f'id="{i}"' in html, i
    assert 'id="tourCap" role="status" aria-live="polite"' in html
    assert html.index("util.js") < html.index("app.js") < html.index("tour.js")
    assert "Escape" in js
    # the tour starts only from a click handler: nothing calls startTour() at load time
    assert js.count("startTour()") == 1 and "addEventListener(\"click\", startTour)" in js


def test_all_static_js_passes_node_check_and_has_no_sinks():
    from tests.test_api import _assert_ui_has_no_html_sinks
    _assert_ui_has_no_html_sinks()
    for f in STATIC.glob("*.js"):
        src = f.read_text()
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
            assert bad not in src, (f.name, bad)
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
