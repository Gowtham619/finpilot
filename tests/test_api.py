import io
import json
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from finpilot import api, db, service
from finpilot.llm import LLMUnavailable, set_llm
from finpilot.trace import trace
from tests.conftest import FakeLLM


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FINPILOT_SYNC", "1")
    db.configure("sqlite://")

    def down(s, u):
        raise LLMUnavailable("test")
    set_llm(FakeLLM(down))
    return TestClient(api.create_app())


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200 and "FinPilot" in r.text


def test_no_data_state(client):
    assert client.get("/api/state/nobody").json()["has_data"] is False


def test_load_persona_then_state(client):
    r = client.post("/api/load/priya")
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    st = client.get("/api/state/priya").json()
    assert st["has_data"] and st["recs"] and all(x["verified"] for x in st["recs"])
    assert len(st["forecast"]["bands"]["p50"]) == 60
    assert st["monthly"]["months"] and st["monthly"]["series"]
    events = client.get(f"/trace/{run_id}").json()
    assert {"categorizer", "analyzer", "forecaster", "advisor", "verifier"} <= {e["agent"] for e in events}


def test_sse_stream_ends_with_done(client):
    run_id = client.post("/api/load/priya").json()["run_id"]
    with client.stream("GET", f"/trace/{run_id}/stream") as r:
        body = "".join(r.iter_text())
    assert "event: trace" in body and '"kind": "done"' in body


def test_sse_stream_for_unknown_run_ends(client):
    t0 = time.time()
    with client.stream("GET", "/trace/neverexisted/stream") as r:
        body = "".join(r.iter_text())
    assert time.time() - t0 < 6
    assert "done" not in body


def test_afford_and_duel_and_chat_need_data(client):
    assert client.post("/api/afford/nobody", json={"amount": 100}).status_code == 404


def test_afford_after_load(client):
    client.post("/api/load/priya")
    r = client.post("/api/afford/priya", json={"amount": 80000})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]["verdict"] in {"safe", "risky", "unsafe"} and body["run_id"]


def test_afford_bad_inputs_are_400(client):
    client.post("/api/load/priya")
    r = client.post("/api/afford/priya", json={"amount": 5000, "date": "2020-01-01"})
    assert r.status_code == 400 and "outside" in r.json()["detail"]
    assert client.post("/api/afford/priya", json={"amount": 5000, "date": "not-a-date"}).status_code == 400
    assert client.post("/api/afford/priya", json={}).status_code == 400
    assert client.post("/api/afford/priya", json={"amount": "abc"}).status_code == 400
    assert client.post("/api/afford/priya", json={"amount": -5}).status_code == 400


def test_duel_bad_inputs_are_400(client):
    client.post("/api/load/priya")
    assert client.post("/api/duel/priya", json={"a": {"amount": 1}}).status_code == 400
    assert client.post("/api/duel/priya", json={"a": {"amount": "x"}, "b": {"amount": 0}}).status_code == 400
    assert client.post("/api/duel/priya", json={"a": {"amount": 1, "date": "zzz"}, "b": {"amount": 0}}).status_code == 400
    r = client.post("/api/duel/priya", json={"a": {"name": "Early", "amount": 1, "date": "2020-01-01"}, "b": {"amount": 0}})
    assert r.status_code == 400 and "Early" in r.json()["detail"] and "outside" in r.json()["detail"]
    assert client.post("/api/duel/priya", json={"a": {"amount": -1}, "b": {"amount": 0}}).status_code == 400
    assert client.post("/api/duel/priya", json={"a": {"amount": 1, "financed_months": 99}, "b": {"amount": 0}}).status_code == 400
    # zero-amount option needs no valid date window
    r = client.post("/api/duel/priya", json={"a": {"amount": 0, "date": "2020-01-01"}, "b": {"amount": 0}})
    assert r.status_code == 200


def test_duel_installments_past_horizon_are_noticed(client):
    client.post("/api/load/priya")
    last = pd.Timestamp(client.get("/api/state/priya").json()["forecast"]["bands"]["dates"][0])
    r = client.post("/api/duel/priya", json={"a": {"name": "EMI", "amount": 60000, "financed_months": 12,
                                                    "date": str(last.date())}, "b": {"amount": 0}})
    assert r.status_code == 200
    a = r.json()["answer"]
    assert a["ignored_flows"] >= 1 and "outside the 90-day window" in a["notice"]


def test_afford_negative_and_financed_months_limits(client):
    client.post("/api/load/priya")
    assert client.post("/api/afford/priya", json={"amount": 100, "financed_months": 61}).status_code == 400
    assert client.post("/api/afford/priya", json={"amount": 100, "financed_months": -1}).status_code == 400


def test_unknown_request_type_rejected():
    db.configure("sqlite://")
    with pytest.raises(ValueError):
        service.request("nobody", {"type": "bogus"})


def test_graph_failure_emits_error_then_done(client, monkeypatch):
    client.post("/api/load/priya")

    def boom(*a, **k):
        raise RuntimeError("kaput")
    monkeypatch.setattr(service, "build_graph", boom)
    with pytest.raises(RuntimeError):
        service.request("priya", {"type": "afford", "amount": 10})
    runs = [r for r, evs in trace._ev.items() if any(e["summary"].startswith("RuntimeError") for e in evs)]
    assert runs
    kinds = [e["kind"] for e in trace.since(runs[-1])]
    assert kinds[-2:] == ["error", "done"] and trace.is_done(runs[-1])


def test_duel_after_load(client):
    client.post("/api/load/priya")
    r = client.post("/api/duel/priya", json={"a": {"name": "Buy now", "amount": 400000}, "b": {"name": "Skip", "amount": 0}})
    assert r.status_code == 200 and r.json()["answer"]["winner"] == "b"


def test_chat_endpoint(client):
    client.post("/api/load/priya")
    r = client.post("/api/chat/priya", json={"question": "how much did I spend on food?"})
    assert r.status_code == 200 and r.json()["answer"]["text"]


def _csv(desc="UPI-SWIGGY"):
    return "date,description,amount\n" + "\n".join(
        f"2026-0{m}-0{d},{desc}-{m}{d},-{300 + m}" for m in range(1, 7) for d in (2, 5, 9)) + \
        "\n2026-01-01,ACME CORP SALARY,50000\n"


def test_upload_csv_and_bad_csv(client):
    ok = client.post("/api/upload", data={"profile": "u1", "opening_balance": "1000"},
                     files={"file": ("t.csv", io.BytesIO(_csv().encode()), "text/csv")})
    assert ok.status_code == 200
    assert client.get("/api/state/u1").json()["has_data"]
    bad = client.post("/api/upload", data={"profile": "u2", "opening_balance": "0"},
                      files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert bad.status_code == 400 and "date" in bad.json()["detail"]


def _assert_ui_has_no_html_sinks():
    import re
    js = (api.HERE / "static" / "app.js").read_text()
    html = (api.HERE / "templates" / "index.html").read_text()
    for src in (js, html):
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
                    "new Function", "srcdoc"):
            assert bad not in src, bad
    assert not re.search(r"\son\w+\s*=", html, re.I), "inline event handler in index.html"


def test_script_in_description_roundtrips_as_text(client):
    evil = "<script>alert(1)</script>"
    rows = ["date,description,amount", "2026-01-01,ACME CORP SALARY,90000"]
    for m in range(1, 7):
        rows.append(f"2026-0{m}-03,{evil},-799")
        for d in (5, 9, 14, 20):
            rows.append(f"2026-0{m}-{d:02d},UPI-SWIGGY-{m}{d},-{250 + m * d}")
    r = client.post("/api/upload", data={"profile": "xss", "opening_balance": "1000"},
                    files={"file": ("t.csv", io.BytesIO("\n".join(rows).encode()), "text/csv")})
    assert r.status_code == 200
    st = client.get("/api/state/xss")
    assert st.status_code == 200 and st.headers["content-type"].startswith("application/json")
    with db.session() as sess:  # stored verbatim; merchants are normalised to letters so never echo markup
        assert evil in set(db.load_df(sess, "xss")[0]["description"])
    # user-supplied text that the API does echo (goal names) roundtrips as inert JSON text
    client.post("/api/goals/xss", json={"name": evil, "target": 90000, "deadline": "2026-12-01"})
    client.post("/api/rerun/xss")
    st = client.get("/api/state/xss").json()
    assert st["goals"][0]["name"] == evil and st["forecast"]["goals"][0]["name"] == evil
    _assert_ui_has_no_html_sinks()


def test_goals_roundtrip(client):
    client.post("/api/load/priya")
    r = client.post("/api/goals/priya", json={"name": "Trip", "target": 300000, "deadline": "2027-03-01"})
    assert r.status_code == 200
    gs = client.get("/api/goals/priya").json()
    assert gs[0]["name"] == "Trip"
    run = client.post("/api/rerun/priya").json()["run_id"]
    st = client.get("/api/state/priya").json()
    assert st["forecast"]["goals"][0]["name"] == "Trip" and run


def test_correction_changes_category(client):
    client.post("/api/load/priya")
    r = client.post("/api/correct/priya", json={"merchant": "PVR CINEMAS", "category": "dining"})
    assert r.status_code == 200 and r.json()["run_id"]
    with db.session() as sess:
        assert db.corrections(sess, "priya")["PVR CINEMAS"] == "dining"


def _post_csv(client, profile, body, balance="0"):
    return client.post("/api/upload", data={"profile": profile, "opening_balance": balance},
                       files={"file": ("t.csv", io.BytesIO(body), "text/csv")})


def test_upload_size_limit_413(client):
    big = b"date,description,amount\n" + b"x" * (5 * 1024 * 1024 + 10)
    assert _post_csv(client, "big", big).status_code == 413


def test_upload_row_cap_400(client, monkeypatch):
    monkeypatch.setattr(service, "MAX_ROWS", 10)
    rows = "date,description,amount\n" + "\n".join(f"2026-01-{d:02d},A,-1" for d in range(1, 21))
    r = _post_csv(client, "rows", rows.encode())
    assert r.status_code == 400 and "too many rows" in r.json()["detail"]


def test_upload_rejects_bad_profile_and_balance(client):
    ok_csv = _csv().encode()
    assert _post_csv(client, "bad name!", ok_csv).status_code == 400
    assert _post_csv(client, "a" * 41, ok_csv).status_code == 400
    assert _post_csv(client, "p", ok_csv, "nan").status_code == 400
    assert _post_csv(client, "p", ok_csv, "inf").status_code == 400
    assert _post_csv(client, "p", ok_csv, "100").status_code == 200


def test_profile_regex_on_every_endpoint(client):
    bad = "bad%20name"
    assert client.get(f"/api/state/{bad}").status_code == 400
    assert client.get(f"/api/goals/{bad}").status_code == 400
    assert client.post(f"/api/goals/{bad}", json={}).status_code == 400
    assert client.post(f"/api/afford/{bad}", json={"amount": 1}).status_code == 400
    assert client.post(f"/api/duel/{bad}", json={}).status_code == 400
    assert client.post(f"/api/chat/{bad}", json={}).status_code == 400
    assert client.post(f"/api/rerun/{bad}").status_code == 400
    assert client.post(f"/api/correct/{bad}", json={"category": "dining"}).status_code == 400
    with pytest.raises(ValueError):
        service.start_pipeline("bad name", pd.DataFrame(), 0)


def test_afford_financed_notice_and_unfinanced_none(client):
    client.post("/api/load/priya")
    start = pd.Timestamp(client.get("/api/state/priya").json()["forecast"]["bands"]["dates"][0])
    fin = client.post("/api/afford/priya", json={"amount": 120000, "financed_months": 12,
                                                  "date": str(start.date())}).json()["answer"]
    n = fin["sim"]["ignored_flows"]
    assert n > 0 and fin["notice"] == f"{n} installment(s) fall outside the 60-day window and are not simulated"
    assert fin["notice"] in fin["narrative"]
    plain = client.post("/api/afford/priya", json={"amount": 5000}).json()["answer"]
    assert not plain.get("notice")


def test_duel_horizon_bounds(client):
    client.post("/api/load/priya")
    for h in (0, 366, -5, "abc", 1.5, True):
        r = client.post("/api/duel/priya", json={"horizon": h, "a": {"amount": 0}, "b": {"amount": 0}})
        assert r.status_code == 400, h
    r = client.post("/api/duel/priya", json={"horizon": 30, "a": {"amount": 0}, "b": {"amount": 0}})
    assert r.status_code == 200


def test_digest_now(client):
    client.post("/api/load/priya")
    out = client.post("/api/digest/priya").json()
    assert "Key insight" in out["content"] and out["channel"] == "ui"
    with db.session() as s:
        assert db.list_digests(s, "priya")[0]["content"] == out["content"]


def test_digest_endpoint_validates_profile_and_data(client):
    assert client.post("/api/digest/nobody").status_code == 404
    assert client.post("/api/digest/bad name").status_code == 400
    with db.session() as s:
        assert db.list_digests(s, "nobody") == []
