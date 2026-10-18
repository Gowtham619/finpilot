import pandas as pd
import pytest
from finpilot import db
from finpilot.trace import Trace


@pytest.fixture
def s():
    db.configure("sqlite://")
    with db.session() as sess:
        yield sess


def test_transactions_roundtrip(s):
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                       "description": ["a", "b"], "amount": [-1.0, 2.0],
                       "merchant": ["A", "B"], "category": ["other", "income"],
                       "category_source": ["rule", "rule"], "confidence": [0.9, 0.9]})
    db.save_transactions(s, "p1", df, 100.0)
    out, opening = db.load_df(s, "p1")
    assert opening == 100.0 and list(out.merchant) == ["A", "B"] and list(out.index) == [0, 1]
    assert db.load_df(s, "nobody") == (None, None)


def test_save_replaces_not_appends(s):
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "description": ["a"], "amount": [-1.0],
                       "merchant": ["A"], "category": ["other"], "category_source": ["rule"], "confidence": [1.0]})
    db.save_transactions(s, "p1", df, 0.0)
    db.save_transactions(s, "p1", df, 0.0)
    assert len(db.load_df(s, "p1")[0]) == 1


def test_recs_goals_corrections_digests(s):
    db.save_recs(s, "p1", [{"id": "r1", "text": "x", "verified": True}])
    assert db.load_recs(s, "p1")[0]["id"] == "r1"
    db.add_goal(s, "p1", "Trip", 50000, "2027-03-01")
    assert db.list_goals(s, "p1")[0]["name"] == "Trip"
    db.add_correction(s, "p1", "PVR CINEMAS", "entertainment")
    assert db.corrections(s, "p1") == {"PVR CINEMAS": "entertainment"}
    db.save_digest(s, "p1", "ui", "hello", False)
    assert db.list_digests(s, "p1")[0]["content"] == "hello"


def test_trace_bus():
    t = Trace()
    t.emit("r", "categorizer", "start", "go")
    t.emit("r", "verifier", "caught", "bad", {"x": 1})
    assert [e["seq"] for e in t.since("r")] == [0, 1]
    assert t.since("r", 1)[0]["agent"] == "verifier"
    assert not t.is_done("r")
    t.finish("r")
    assert t.is_done("r")


def test_blob_with_numpy_and_timestamp(s):
    import numpy as np
    db.save_recs(s, "p1", [{"id": "r", "n": np.int64(5), "f": np.float64(1.5), "d": pd.Timestamp("2026-01-01")}])
    s.flush()
    assert db.load_recs(s, "p1")[0]["id"] == "r"


def test_trace_unknown_run_and_idempotent_finish():
    t = Trace()
    assert t.since("ghost") == [] and "ghost" not in t._ev
    t.emit("r", "a", "start", "x")
    t.finish("r")
    t.finish("r")
    assert [e["kind"] for e in t.since("r")].count("done") == 1
