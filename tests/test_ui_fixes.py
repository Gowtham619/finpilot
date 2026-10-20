"""Static and node checks for the final-review UI fixes (duel result card, deep links, calendar markers,
tour presets, labels). Behaviour in a real browser is checked by hand (see the fix report)."""
import json
import re
import shutil
import subprocess

import pytest
from finpilot import api

STATIC = api.HERE / "static"
APP = (STATIC / "app.js").read_text()
TOUR = (STATIC / "tour.js").read_text()
HTML = (api.HERE / "templates" / "index.html").read_text()
node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def U(expr):
    r = subprocess.run(["node", "-e", f"const U=require('./util.js');console.log(JSON.stringify({expr}))"],
                       capture_output=True, text=True, cwd=str(STATIC), timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@node
def test_timestamps_without_microseconds_and_whitelist_pick():
    assert U("U.fmtStamp('2026-10-20 12:34:56.123456')") == "2026-10-20 12:34:56 UTC"
    assert U("U.fmtStamp('2026-10-20T01:02:03')") == "2026-10-20 01:02:03 UTC"
    assert U("[U.pick('weather',['dashboard','weather']), U.pick('x<y',['weather']), U.pick(null,['weather'])]") \
        == ["weather", None, None]
    assert "fmtStamp(d.created_at)" in APP


def test_duel_always_shows_the_computed_result():
    assert '"Result: " + a.reason' in APP
    assert "a.narrative !== a.reason" in APP


def test_persona_select_shows_current_profile_and_load_uses_it():
    assert '$("#persona").value = profile' in APP
    assert "/api/load/${encodeURIComponent($(\"#persona\").value)}" in APP


def test_storm_highlight_cleared_when_storm_disappears():
    assert "W.hl ? (d.storms.find((s) => s.start === W.hl.start) || null) : null" in APP


def test_planned_prefix_and_afford_notice_shown_once():
    assert "`planned: ${" not in APP
    assert "a.notice && !a.narrative.includes(a.notice)" in APP


def test_deep_link_values_are_whitelisted_and_no_autoplay():
    assert 'pick(QS.get("tab"), TABS.map((t) => t[0]))' in APP
    assert 'pick(QS.get("scenario"), SCENARIOS.map((x) => x[0]))' in APP
    assert "startTour" not in APP and not re.search(r"tour\w*\s*=\s*QS", APP)


def test_calendar_month_markers_flows_paydays_and_legend():
    assert "mstart" in APP and "d.paydays" in APP and 'marker("pay", "₹+", "payday")' in APP
    assert 'marker("out", "₹−", t)' in APP and "missing salary" in APP
    assert 'setAttribute("aria-label", label)' in APP
    for word in ("scenario outflow", "missing salary", "payday"):
        assert word in HTML


def test_tour_uses_current_persona_months_from_state_and_duel_presets():
    assert "six months" not in TOUR and 'value = "priya"' not in TOUR
    assert "st.monthly.months.length" in TOUR
    assert "[0.95, 0.9, 0.85]" in TOUR and "/api/duel/" in TOUR and "paydays" in TOUR
    assert "re-computes" not in TOUR and "engine's own evidence" in TOUR


def test_labels_and_non_bill_wording():
    assert 'placeholder="profile name"' in HTML and 'placeholder="opening balance ₹"' in HTML
    assert "Profile name <input" in HTML and "Opening balance ₹ <input" in HTML
    assert "Non-bill spending 100%" in HTML and "Lifestyle spending" not in APP


def test_makefile_prefers_venv_python():
    mk = (api.HERE.parent / "Makefile").read_text()
    assert "PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)" in mk


def test_readme_screenshot_exists_and_is_referenced():
    root = api.HERE.parent
    img = root / "docs" / "img" / "money-weather-storm.png"
    assert img.exists() and 10_000 < img.stat().st_size < 400_000
    assert img.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    readme = (root / "README.md").read_text()
    assert "](docs/img/money-weather-storm.png)" in readme
    assert "re-computes the numbers in every recommendation" not in readme
