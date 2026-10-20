import re
import shutil
import subprocess

import pytest
from finpilot import api
from tests.test_api import client  # noqa: F401  (fixture)

JS = api.HERE / "static" / "app.js"
HTML = api.HERE / "templates" / "index.html"

TOP_KEYS = {"calendar", "storms", "autopsy", "umbrellas", "headline", "buffer", "headline_text", "autopsy_text",
            "umbrella_text", "scenario", "scenario_flows", "scenario_amount", "run_id"}
CAL_KEYS = {"date", "p_below_buffer", "label", "p10", "p50", "p90"}
STORM_KEYS = {"start", "end", "days", "peak_p", "peak_date", "label"}
AUTOPSY_KEYS = {"p_any_breach", "n_failing", "n_paths", "causes", "bills", "insufficient_failing", "first_breach"}
CAUSE_KEYS = {"category", "excess_per_path", "failing_mean", "other_mean", "share_of_gap"}
BILL_KEYS = {"merchant", "date", "amount"}
FIRST_KEYS = {"date", "share", "already_below"}
UMB_KEYS = {"category", "cut_pct", "cumulative_cut_pct", "p_any_breach_before", "p_any_breach_after",
            "risk_reduction", "monthly_sacrifice"}


def test_html_has_weather_tab_and_sections():
    html = HTML.read_text()
    for i in ("weather", "wScen", "wAmount", "wLife", "wReset", "wRefresh", "wChip", "wCal", "wStorms",
              "wAutopsy", "wUmbrellas", "wApplied"):
        assert f'id="{i}"' in html, i
    assert "Money Weather" in JS.read_text()
    assert JS.read_text().index('"weather", "Money Weather"') < JS.read_text().index('"afford"')


def test_js_references_weather_api_and_encodes_profile():
    js = JS.read_text()
    assert "/api/weather/${encodeURIComponent(profile)}" in js


def test_weather_ui_has_no_html_sinks():
    from tests.test_api import _assert_ui_has_no_html_sinks
    _assert_ui_has_no_html_sinks()
    js, html = JS.read_text(), HTML.read_text()
    assert not re.search(r"<[^>]+\son\w+\s*=", html, re.I)
    assert not re.search(r"setAttribute\(\s*[\"']on", js, re.I)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_syntax_check():
    r = subprocess.run(["node", "--check", str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_api_fields_used_by_ui_exist(client):  # noqa: F811
    assert client.post("/api/load/priya").status_code == 200
    d = client.get("/api/weather/priya?scenario=purchase").json()
    assert TOP_KEYS <= set(d)
    assert d["calendar"] and CAL_KEYS <= set(d["calendar"][0])
    assert d["storms"] and STORM_KEYS <= set(d["storms"][0])
    a = d["autopsy"]
    assert AUTOPSY_KEYS <= set(a)
    assert a["causes"] and CAUSE_KEYS <= set(a["causes"][0])
    assert a["bills"] and BILL_KEYS <= set(a["bills"][0])
    assert FIRST_KEYS <= set(a["first_breach"])
    assert d["umbrellas"] and UMB_KEYS <= set(d["umbrellas"][0])
    assert d["scenario_flows"] and {"date", "amount", "label"} <= set(d["scenario_flows"][0])
    assert {"p_any_breach", "worst_day"} <= set(d["headline"])
