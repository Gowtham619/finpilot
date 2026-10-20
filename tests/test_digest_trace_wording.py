"""Honest verification wording (digest, trace), the Money Weather digest line and the chat length cap."""
import pytest
from fastapi.testclient import TestClient
from finpilot import api, db, digest, service
from finpilot.llm import LLMUnavailable, set_llm
from finpilot.trace import trace
from tests.conftest import FakeLLM


def _down(s, u):
    raise LLMUnavailable("test")


@pytest.fixture
def loaded(monkeypatch):
    monkeypatch.setenv("FINPILOT_SYNC", "1")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    db.configure("sqlite://")
    set_llm(FakeLLM(_down))
    _, run_id = service.load_persona("priya")
    return run_id


def test_digest_why_line_depends_on_verification_kind(loaded):
    with db.session() as s:
        recs = db.load_recs(s, "priya")
    best = digest.pick_action([r for r in recs if r.get("verified")])
    text = digest.build_digest("priya")
    assert "verified against your transactions" not in text
    want = ("recomputed from your transactions" if best["verification"] == "recomputed"
            else "checked against engine evidence")
    assert f"({want})." in text
    assert digest.verification_phrase({"verification": "evidence_only"}) == "checked against engine evidence"
    assert digest.verification_phrase({"verification": "recomputed"}) == "recomputed from your transactions"


def test_digest_has_one_money_weather_line(loaded):
    text = digest.build_digest("priya")
    lines = [ln for ln in text.splitlines() if ln.startswith("Money Weather: ")]
    assert len(lines) == 1 and "Clear skies" in lines[0] and "Worst storm window" not in lines[0]
    with db.session() as s:
        stored = db.load_forecast(s, "priya")
    wx = stored["weather"]
    wx["storms"] = [{"start": "2026-10-24", "end": "2026-10-31", "days": 8, "peak_p": 0.749,
                     "peak_date": "2026-10-31", "label": "storm"}]
    line = digest.weather_line(wx)
    assert "Worst storm window: 2026-10-24 to 2026-10-31 (peak 75% chance" in line
    assert digest.weather_line(None) is None and digest.weather_line({"x": 1}) is None


def test_verifier_trace_reports_recomputed_vs_evidence_counts(loaded):
    evs = [e for e in trace.since(loaded) if e["agent"] == "verifier"]
    start = next(e for e in evs if e["kind"] == "start")["summary"]
    result = next(e for e in evs if e["kind"] == "result")["summary"]
    assert "recomputing every cited number" not in start
    assert "recomputed from raw transactions" in start and "engine's own evidence" in start
    with db.session() as s:
        recs = db.load_recs(s, "priya")
    n_re = sum(r["verification"] == "recomputed" for r in recs if r["verified"])
    assert f"({n_re} recomputed from transactions, {len(recs) - n_re} checked against engine evidence only)" in result


def test_chat_question_length_is_capped(monkeypatch):
    monkeypatch.setenv("FINPILOT_SYNC", "1")
    db.configure("sqlite://")
    set_llm(FakeLLM(_down))
    c = TestClient(api.create_app())
    c.post("/api/load/priya")
    assert c.post("/api/chat/priya", json={"question": "x" * 501}).status_code == 400
    assert c.post("/api/chat/priya", json={"question": "  "}).status_code == 400
    assert c.post("/api/chat/priya", json={"question": "how much on food?" + " " * 400}).status_code == 200
