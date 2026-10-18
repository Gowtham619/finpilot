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
        def __init__(self, host, port=587, timeout=None): sent["host"] = host
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


def _setup_smtp(monkeypatch, smtp_cls, to="me@example.com"):
    monkeypatch.setattr("finpilot.digest.smtplib.SMTP", smtp_cls)
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "s3cretpw")
    monkeypatch.setenv("DIGEST_TO", to)


def test_smtp_failure_logged_without_password_and_reported(loaded, monkeypatch, caplog):
    import smtplib
    seen = {}

    class Bad:
        def __init__(self, host, port=587, timeout=None):
            seen["port"], seen["timeout"] = port, timeout
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, u, p): raise smtplib.SMTPAuthenticationError(535, b"bad s3cretpw creds")
    _setup_smtp(monkeypatch, Bad)
    monkeypatch.setenv("SMTP_PORT", "2525")
    with caplog.at_level("WARNING", logger="finpilot.digest"):
        assert digest.send_email("s", "b") is False
        out = digest.deliver("priya")
    assert seen == {"port": 2525, "timeout": 10}
    assert out["sent"] is False and out["channel"] == "ui" and out["error"] == "SMTPAuthenticationError"
    assert "SMTPAuthenticationError" in caplog.text and "s3cretpw" not in caplog.text


def test_not_configured_is_quiet(monkeypatch, caplog):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    with caplog.at_level("WARNING", logger="finpilot.digest"):
        assert digest.send_email("s", "b") is False
    assert caplog.text == ""


def test_newline_in_recipient_does_not_raise(loaded, monkeypatch):
    class Never:
        def __init__(self, *a, **k): pass
    _setup_smtp(monkeypatch, Never, to="a@b.com\nBcc: evil@x.com")
    out = digest.deliver("priya")
    assert out["sent"] is False and "error" in out


def test_digest_without_narrative(loaded):
    with db.session() as s:
        fc = db.load_forecast(s, "priya")
        del fc["forecast"]["narrative"]
        db.save_forecast(s, "priya", fc)
    assert "Key insight" in digest.build_digest("priya")


def test_pace_zero_expected_flagged_and_sorted():
    df = prep()
    today = pd.Timestamp("2026-09-15")
    df = df[df["date"] <= today]
    extra = pd.DataFrame({"date": [pd.Timestamp("2026-09-10")] * 2, "description": ["X"] * 2,
                          "amount": [-5000.0] * 2, "merchant": ["BRANDNEWSHOP"] * 2, "category": ["travel"] * 2})
    alerts = pace.pace_alerts(pd.concat([df, extra], ignore_index=True), today)
    a = next(a for a in alerts if a["category"] == "travel")
    if a["expected"] == 0:
        assert a["new_spend"] is True and a["over_pct"] is None
    overs = [x["mtd"] - x["expected"] for x in alerts]
    assert overs == sorted(overs, reverse=True)


def _job_digests():
    with db.session() as s:
        return [d["content"] for d in db.list_digests(s, "priya")]


def test_pace_job_messages_largest_dedupes_and_no_none(loaded, monkeypatch):
    from finpilot import scheduler
    alerts = [{"category": "dining", "over_pct": 50.0, "day": 5, "mtd": 4000.0, "expected": 3000.0},
              {"category": "travel", "over_pct": None, "new_spend": True, "day": 5, "mtd": 9000.0, "expected": 0.0},
              {"category": "shopping", "over_pct": 40.0, "day": 5, "mtd": 3000.0, "expected": 2000.0}]
    monkeypatch.setattr("finpilot.engines.pace.pace_alerts", lambda d, t: alerts)
    before = len(_job_digests())
    scheduler.pace_check_job()
    scheduler.pace_check_job()  # identical alert within 24h -> skipped
    got = _job_digests()
    assert len(got) == before + 1
    assert got[0] == "Pace alert: new spending in travel this month: ₹9,000 by day 5." and "None" not in got[0]
    alerts[:] = alerts[:1]  # different content is still recorded
    scheduler.pace_check_job()
    assert len(_job_digests()) == before + 2


def test_scheduler_jobs_coalesce_and_grace():
    from finpilot.scheduler import build_scheduler
    for j in build_scheduler().get_jobs():
        assert j.coalesce is True and j.misfire_grace_time == 3600


def test_pace_job_dedupes_on_stable_id_when_text_changes(loaded, monkeypatch):
    """The alert text changes daily (day, amounts) but the id pace:{category}:{month} does not."""
    from finpilot import scheduler
    a = {"id": "pace:dining:2026-09", "category": "dining", "over_pct": 50.0, "day": 5,
         "mtd": 4000.0, "expected": 3000.0}
    monkeypatch.setattr("finpilot.engines.pace.pace_alerts", lambda d, t: [dict(a)])
    before = len(_job_digests())
    scheduler.pace_check_job()
    a.update(day=6, mtd=4100.0, over_pct=36.7)  # new text, same id, within 24h
    scheduler.pace_check_job()
    assert len(_job_digests()) == before + 1
    with db.session() as s:  # age the ledger past 24h
        for row in s.scalars(db.select(db.PaceAlert)).all():
            row.created_at = row.created_at - pd.Timedelta(hours=25)
    scheduler.pace_check_job()  # overage 1100 vs 1000: +10%, not >25%
    assert len(_job_digests()) == before + 1
    a.update(day=9, mtd=4400.0)  # overage 1400: +40%
    scheduler.pace_check_job()
    assert len(_job_digests()) == before + 2
