import json
import pandas as pd
import pytest
from eval.generate import generate
from finpilot import graph
from finpilot.llm import LLMUnavailable
from finpilot.trace import Trace
from tests.conftest import FakeLLM


def raw_df():
    return generate(seed=0).df.copy(), 40000.0


def run(request, df, tr, llm=None, **extra):
    g = graph.build_graph(llm, tr)
    return g.invoke({"run_id": "r1", "profile": "t", "request": request, "df": df, "opening": 40000.0,
                     "corrections": {}, "goals": [], **extra})


def test_route():
    assert graph.route({"request": {"type": "afford"}}) == "afford"


def test_pipeline_end_to_end_without_llm():
    tr = Trace()
    df, _ = raw_df()
    out = run({"type": "pipeline"}, df, tr)
    assert "category" in out["df"].columns
    assert out["findings"] and out["recs"]
    assert all(r["verified"] for r in out["recs"])
    assert all("verification" in r for r in out["recs"])
    assert all("model" not in out["forecast"] for _ in [0])
    agents = {e["agent"] for e in tr.since("r1")}
    assert {"categorizer", "analyzer", "forecaster", "advisor", "verifier"} <= agents
    json.dumps(out["recs"], default=str)  # JSON-safe


def test_pipeline_persist_callback_called():
    seen = []
    g = graph.build_graph(None, Trace(), persist=lambda st: seen.append(st["run_id"]))
    df, _ = raw_df()
    g.invoke({"run_id": "r2", "profile": "t", "request": {"type": "pipeline"}, "df": df,
              "opening": 40000.0, "corrections": {}, "goals": []})
    assert seen == ["r2"]


def test_pipeline_verifier_catches_llm_hallucination():
    tr = Trace()
    df, _ = raw_df()

    def handler(system, user):
        if "financial advisor" in system:
            cands = json.loads(user)
            recs = []
            for c in cands:
                claims = [dict(cl, value=cl["value"] * 3) for cl in c["claims"]]
                recs.append({"cid": c["cid"], "text": f"Do this: {c['action']} and save ₹99999.", "claims": claims})
            return {"recommendations": recs}
        raise LLMUnavailable("not scripted")

    out = run({"type": "pipeline"}, df, tr, FakeLLM(handler))
    kinds = [e["kind"] for e in tr.since("r1") if e["agent"] == "verifier"]
    assert "caught" in kinds
    assert all(r["verified"] for r in out["recs"])
    assert all("99999" not in r["text"] for r in out["recs"])


def test_afford_route():
    tr = Trace()
    df, _ = raw_df()
    cat = run({"type": "pipeline"}, df, Trace())["df"]
    last = cat["date"].max().date()
    req = {"type": "afford", "amount": 80000, "date": str(last + pd.Timedelta(days=5))}
    out = run(req, cat, tr)
    assert out["answer"]["verdict"] in {"safe", "risky", "unsafe"}
    assert any(e["agent"] == "simulator" for e in tr.since("r1"))


def test_duel_route():
    cat = run({"type": "pipeline"}, raw_df()[0], Trace())["df"]
    d0 = str(cat["date"].max().date() + pd.Timedelta(days=5))
    req = {"type": "duel", "a": {"name": "Buy now", "amount": 150000, "date": d0},
           "b": {"name": "Skip", "amount": 0, "date": d0}}
    out = run(req, cat, Trace())
    assert out["answer"]["winner"] == "b"


def test_chat_route_answers_and_traces():
    tr = Trace()
    cat = run({"type": "pipeline"}, raw_df()[0], Trace())["df"]
    out = run({"type": "chat", "question": "what subscriptions do I pay for?"}, cat, tr)
    ans = out["answer"]
    assert ans["text"] and ans["tool_calls"][0]["name"] == "list_subscriptions"
    assert {e["kind"] for e in tr.since("r1") if e["agent"] == "chat"} == {"tool", "result"}
