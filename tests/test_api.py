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
    fin = client.post("/api/afford/priya", json={"amount": 150000, "financed_months": 12,
                                                  "date": str(start.date())}).json()["answer"]
    assert fin["sim"]["ignored_flows"] == 0 and fin["sim"]["horizon"] == 365 and not fin.get("notice")
    assert "365 days" in fin["narrative"] and "12 monthly installments" in fin["narrative"]
    for d in fin["best_dates"]:  # best dates simulate the installments, not the lump sum
        assert d["p_overdraft"] <= 0.5, d
    long = client.post("/api/afford/priya", json={"amount": 150000, "financed_months": 24,
                                                   "date": str(start.date())}).json()["answer"]
    n = long["sim"]["ignored_flows"]
    assert n == 12 and long["notice"] == f"{n} installments fall outside the 365-day window and are not simulated"
    assert long["notice"] in long["narrative"]
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


# ---- final review fixes: C5 duel cuts, C3 demo duel ----
@pytest.mark.parametrize("cuts", [{"dining": "x"}, ["dining"], {"dining": 5}, {"dining": float("nan")},
                                  {"dining": -0.1}, {"dining": True}, {"nope": 0.2}, "dining", 5])
def test_duel_cuts_validated(client, cuts):
    client.post("/api/load/priya")
    body = {"a": {"amount": 0, "cuts": cuts}, "b": {"amount": 0}}
    payload = json.dumps(body, allow_nan=True)
    r = client.post("/api/duel/priya", content=payload, headers={"Content-Type": "application/json"})
    assert r.status_code == 400, (cuts, r.text)


def test_duel_valid_cuts_ok(client):
    client.post("/api/load/priya")
    r = client.post("/api/duel/priya", json={"a": {"amount": 0, "cuts": {"dining": 0.5, "shopping": 1}},
                                             "b": {"amount": 0}})
    assert r.status_code == 200 and r.json()["answer"]["winner"] in ("a", "b", "tie")


def test_demo_duel_has_a_numeric_winner(client):
    client.post("/api/load/priya")
    r = client.post("/api/duel/priya", json={"a": {"name": "Buy now", "amount": 80000},
                                             "b": {"name": "Wait 2 months", "amount": 80000, "date": "2026-12-01"}})
    ans = r.json()["answer"]
    assert ans["winner"] == "b" and "lowest balance" in ans["reason"] and "₹" in ans["reason"]


def test_body_cannot_override_route_type(client):
    client.post("/api/load/priya")
    r = client.post("/api/afford/priya", json={"type": "chat", "question": "hi", "amount": 1000})
    assert r.status_code == 200 and "verdict" in r.json()["answer"]
    r = client.post("/api/duel/priya", json={"type": "afford", "amount": 5, "a": {"amount": 0}, "b": {"amount": 0}})
    assert r.status_code == 200 and "winner" in r.json()["answer"]


# ---- final review fixes: C6 goals, I5 digests, I6 corrections, minor snapshots ----
@pytest.mark.parametrize("body,msg", [
    ({"name": "Trip", "target": 1000, "deadline": "notadate"}, "deadline"),
    ({"name": "Trip", "target": float("nan"), "deadline": "2027-03-01"}, "finite"),
    ({"name": "Trip", "target": 1e308, "deadline": "2027-03-01"}, "target"),
    ({"name": "Trip", "target": 0, "deadline": "2027-03-01"}, "target"),
    ({"name": "Trip", "target": -5, "deadline": "2027-03-01"}, "target"),
    ({"name": "Trip", "target": "abc", "deadline": "2027-03-01"}, "target"),
    ({"name": "Trip", "target": True, "deadline": "2027-03-01"}, "target"),
    ({"name": "Trip", "target": 1000, "deadline": "9999-12-31"}, "deadline"),
    ({"name": "Trip", "target": 1000, "deadline": "2020-01-01"}, "deadline"),
    ({"name": "   ", "target": 1000, "deadline": "2027-03-01"}, "name"),
    ({"name": "x" * 61, "target": 1000, "deadline": "2027-03-01"}, "name"),
    ({"name": 5, "target": 1000, "deadline": "2027-03-01"}, "name"),
    ({"target": 1000, "deadline": "2027-03-01"}, "name"),
    ({"name": "Trip", "target": 1000}, "deadline"),
])
def test_goal_validation(client, body, msg):
    client.post("/api/load/priya")
    r = client.post("/api/goals/priya", content=json.dumps(body, allow_nan=True),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and msg in r.json()["detail"], r.text
    assert client.get("/api/goals/priya").json() == []


def test_goal_name_is_stripped_and_deadline_normalized(client):
    client.post("/api/load/priya")
    r = client.post("/api/goals/priya", json={"name": "  Bike 150cc ", "target": "90000", "deadline": "2027-03-01T10:00"})
    assert r.status_code == 200
    g = client.get("/api/goals/priya").json()[0]
    assert g["name"] == "Bike 150cc" and g["target"] == 90000 and g["deadline"] == "2027-03-01"


def test_goal_snapshots_upsert_by_date(client):
    client.post("/api/load/priya")
    client.post("/api/goals/priya", json={"name": "Trip", "target": 300000, "deadline": "2027-03-01"})
    client.post("/api/rerun/priya")
    client.post("/api/rerun/priya")
    hist = client.get("/api/goals/priya").json()[0]["history"]
    assert len(hist) == 1 and hist[0]["on"] == "2026-09-30"


def test_digests_endpoint(client):
    assert client.get("/api/digests/bad name").status_code == 400
    assert client.get("/api/digests/nobody").json() == []
    client.post("/api/load/priya")
    sent = client.post("/api/digest/priya").json()
    got = client.get("/api/digests/priya").json()
    assert got[0]["content"] == sent["content"] and got[0]["channel"] == "ui" and got[0]["sent"] is False


def test_correction_normalizes_merchant_and_rejects_empty(client):
    client.post("/api/load/priya")
    r = client.post("/api/correct/priya", json={"merchant": "upi-blue tokai coffee BLR", "category": "dining"})
    assert r.status_code == 200
    with db.session() as sess:
        assert db.corrections(sess, "priya")["BLUE TOKAI COFFEE"] == "dining"
    for m in ("", "  ", "UPI", None, 5):
        assert client.post("/api/correct/priya", json={"merchant": m, "category": "dining"}).status_code == 400
    with db.session() as sess:
        assert "" not in db.corrections(sess, "priya")


def test_rerun_response_names_the_profile_so_the_ui_keeps_it(client):
    client.post("/api/load/priya")
    r = client.post("/api/rerun/priya")
    assert r.status_code == 200 and r.json()["profile"] == "priya" and r.json()["run_id"]
