import pandas as pd
import pytest
from eval.generate import generate
from finpilot import db, digest
from finpilot.engines import pace
from finpilot.engines.categorize import normalize_merchant


def prep():
    p = generate(seed=0)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    return df


def test_pace_alert_fires_on_injected_overspend():
    df = prep()
    today = pd.Timestamp("2026-09-15")
    df = df[df["date"] <= today]
    extra = pd.DataFrame({"date": [pd.Timestamp("2026-09-10")] * 4, "description": ["ZOMATO"] * 4,
                          "amount": [-1500.0] * 4, "merchant": ["ZOMATO"] * 4, "category": ["food_delivery"] * 4})
    alerts = pace.pace_alerts(pd.concat([df, extra], ignore_index=True), today)
    assert "pace:food_delivery:2026-09" in {a["id"] for a in alerts}


def test_pace_needs_history():
    df = prep()
    df = df[df["date"] >= "2026-08-01"]
    assert pace.pace_alerts(df, pd.Timestamp("2026-09-15")) == []


def test_list_profiles_and_digest(monkeypatch):
    from finpilot import service
    from finpilot.llm import LLMUnavailable, set_llm
    from tests.conftest import FakeLLM
    monkeypatch.setenv("FINPILOT_SYNC", "1")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    db.configure("sqlite://")
    set_llm(FakeLLM(lambda s, u: (_ for _ in ()).throw(LLMUnavailable("x"))))
    service.load_persona("priya")
    with db.session() as s:
        assert db.list_profiles(s) == ["priya"]
    text = digest.build_digest("priya")
    assert "Key insight" in text and "One action" in text
    out = digest.deliver("priya")
    assert out["channel"] == "ui" and out["sent"] is False
    with db.session() as s:
        assert db.list_digests(s, "priya")[0]["content"] == text


def test_email_sent_when_configured(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port=587): sent["host"] = host
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): sent["tls"] = True
        def login(self, u, p): sent["login"] = u
        def send_message(self, msg): sent["to"] = msg["To"]

    monkeypatch.setattr("finpilot.digest.smtplib.SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "p")
    monkeypatch.setenv("DIGEST_TO", "me@example.com")
    assert digest.send_email("s", "b") is True
    assert sent == {"host": "smtp.test", "tls": True, "login": "u", "to": "me@example.com"}


def test_scheduler_has_real_jobs():
    from finpilot.scheduler import build_scheduler
    s = build_scheduler()
    assert {j.id for j in s.get_jobs()} == {"weekly_digest", "pace_check"}


def test_pace_alert_has_expected_fields():
    df = prep()
    today = pd.Timestamp("2026-09-15")
    df = df[df["date"] <= today]
    extra = pd.DataFrame({"date": [pd.Timestamp("2026-09-10")] * 4, "description": ["ZOMATO"] * 4,
                          "amount": [-1500.0] * 4, "merchant": ["ZOMATO"] * 4, "category": ["food_delivery"] * 4})
    a = next(a for a in pace.pace_alerts(pd.concat([df, extra], ignore_index=True), today)
             if a["category"] == "food_delivery")
    assert a["type"] == "pace_alert" and a["day"] == 15 and a["mtd"] >= 6000 and a["over_pct"] > 30


@pytest.fixture
def loaded(monkeypatch):
    from finpilot import service
    from finpilot.llm import LLMUnavailable, set_llm
    from tests.conftest import FakeLLM
    monkeypatch.setenv("FINPILOT_SYNC", "1")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    db.configure("sqlite://")
    set_llm(FakeLLM(lambda s, u: (_ for _ in ()).throw(LLMUnavailable("x"))))
    service.load_persona("priya")


def test_digest_skips_goal_error_entries(loaded):
    with db.session() as s:
        fc = db.load_forecast(s, "priya")
        fc["forecast"]["goals"] = [{"name": "bad", "error": "invalid goal: x"},
                                   {"name": "ok", "target": 1000, "deadline": "2027-01-01", "p_met": 0.5}]
        db.save_forecast(s, "priya", fc)
    text = digest.build_digest("priya")
    assert "Goal 'ok'" in text and "bad" not in text


def test_digest_unknown_profile():
    db.configure("sqlite://")
    assert digest.build_digest("ghost").startswith("No data yet")


def test_weekly_job_delivers_each_profile(monkeypatch):
    from finpilot import scheduler
    db.configure("sqlite://")
    with db.session() as s:
        for n in ("a", "b"):
            db.save_transactions(s, n, pd.DataFrame(
                {"date": [pd.Timestamp("2026-01-01")], "description": ["x"], "amount": [1.0], "merchant": ["X"],
                 "category": ["other"], "category_source": ["rule"], "confidence": [1.0]}), 0)
    called = []

    def fake(p):
        called.append(p)
        if p == "a":
            raise RuntimeError("boom")  # one failure must not stop the rest
    monkeypatch.setattr("finpilot.digest.deliver", fake)
    scheduler.weekly_digest_job()
    assert called == ["a", "b"]


def test_pace_check_job_records_alert(loaded, monkeypatch):
    from finpilot import scheduler
    with db.session() as s:
        df, opening = db.load_df(s, "priya")
    end = df["date"].max()
    monkeypatch.setattr(scheduler, "_today", lambda: end)
    monkeypatch.setattr("finpilot.engines.pace.pace_alerts", lambda d, t: [
        {"id": "pace:x:2026-01", "category": "dining", "over_pct": 50.0, "day": 5, "mtd": 5000.0, "expected": 3000.0}])
    scheduler.pace_check_job()
    with db.session() as s:
        assert "Pace alert" in db.list_digests(s, "priya")[0]["content"]


def test_scheduler_lifespan_starts_and_stops(monkeypatch):
    from fastapi.testclient import TestClient
    from finpilot import api, scheduler
    monkeypatch.setenv("FINPILOT_SYNC", "1")
    monkeypatch.setenv("FINPILOT_SCHEDULER", "1")
    db.configure("sqlite://")
    made, events = [], []
    real = scheduler.build_scheduler

    def spy():
        s = real()
        made.append(s)
        s_start, s_shut = s.start, s.shutdown
        s.start = lambda *a, **k: (events.append("start"), s_start(*a, **k))[1]
        s.shutdown = lambda *a, **k: (events.append("shutdown"), s_shut(*a, **k))[1]
        return s
    monkeypatch.setattr(scheduler, "build_scheduler", spy)
    with TestClient(api.create_app()) as c:
        assert events == ["start"] and made[0].running
        assert {j.id for j in made[0].get_jobs()} == {"weekly_digest", "pace_check"}
        assert c.get("/api/personas").status_code == 200
    assert events == ["start", "shutdown"]
