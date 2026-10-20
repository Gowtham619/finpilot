import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finpilot import api, db, service
from finpilot.agents import meteorologist as M
from finpilot.engines import weather as W
from finpilot.engines.forecast import current_balance, fit
from finpilot.llm import LLMUnavailable, set_llm
from finpilot.trace import trace
from tests.conftest import FakeLLM

SHOCK = {"scenario": "purchase", "amount": 100000}
LABELS = {"clear", "cloudy", "rain", "storm"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FINPILOT_SYNC", "1")
    db.configure("sqlite://")

    def down(s, u):
        raise LLMUnavailable("test")
    set_llm(FakeLLM(down))
    c = TestClient(api.create_app())
    c.post("/api/load/priya")
    return c


@pytest.fixture(scope="module")
def priya():
    from eval.generate import load_persona
    from finpilot.agents.categorizer import categorize_df
    p = load_persona("priya")
    df = categorize_df(p.df)
    return fit(df), current_balance(df, p.opening_balance)


def get(client, **q):
    return client.get("/api/weather/priya", params=q)


def test_happy_path(client):
    r = get(client)
    assert r.status_code == 200
    j = r.json()
    assert len(j["calendar"]) == 60 and {c["label"] for c in j["calendar"]} <= LABELS
    assert all(isinstance(j[k], str) and j[k].strip() for k in ("headline_text", "autopsy_text", "umbrella_text"))
    assert j["run_id"] and j["scenario"] == "none" and j["scenario_flows"] == []
    json.dumps(j)


def test_lifestyle_monotone_on_purchase(client):
    ps = [get(client, lifestyle=l, scenario="purchase").json()["headline"]["p_any_breach"] for l in (0.8, 1.0, 1.2)]
    assert ps[0] <= ps[1] <= ps[2]


def test_scenarios_on_priya(client):
    j = get(client, scenario="salary_skip").json()
    assert {c["label"] for c in j["calendar"]} & {"rain", "storm"}
    assert len(j["scenario_flows"]) == 2 and all(f["label"] == "salary skipped" and f["amount"] == -62000
                                                  for f in j["scenario_flows"])
    assert "2 pay date" in j["scenario_note"] and j["scenario_amount"] is None
    for sc, day, lab, frac in (("purchase", 10, "planned purchase", 0.95), ("emergency", 15, "emergency expense", 0.7)):
        j = get(client, scenario=sc).json()
        f = j["scenario_flows"][0]
        assert f["label"] == lab and f["date"] == j["calendar"][day - 1]["date"]
        assert f["amount"] == -j["scenario_amount"] and j["scenario_amount"] % 1000 == 0
        assert j["scenario_amount"] == round(frac * 98464.97 / 1000) * 1000
    assert get(client, scenario="purchase", amount=1234).json()["scenario_amount"] == 1234
    assert get(client).json()["scenario_amount"] is None


def test_default_amount_and_multi_salary(priya):
    import dataclasses
    assert M.default_amount("purchase", 0) == 10000 and M.default_amount("emergency", -5) == 10000
    assert M.default_amount("purchase", 98464.97) == 94000 and M.default_amount("emergency", 98464.97) == 69000
    model, bal = priya
    m = dataclasses.replace(model, recurring=[r for r in model.recurring if r["amount"] > 0])
    f = M.scenario_flows(m, "salary_skip", None, 90)
    assert len(f) == 3 and len({x[0] for x in f}) == 3 and all(x[1] == -62000 for x in f)
    assert "3 pay date" in M.run(m, bal, {"scenario": "salary_skip", "horizon": 90})["scenario_note"]


@pytest.mark.parametrize("q", [
    {"lifestyle": "-1"}, {"lifestyle": "5"}, {"lifestyle": "nan"}, {"lifestyle": "abc"},
    {"horizon": "3"}, {"horizon": "500"}, {"horizon": "x"}, {"horizon": "60.5"},
    {"buffer": "nan"}, {"buffer": "-5"}, {"buffer": "0"}, {"buffer": "1e12"}, {"buffer": "zzz"},
    {"cuts": "garbage"}, {"cuts": '{"nope": 0.2}'}, {"cuts": '{"dining": 7}'}, {"cuts": "[1]"},
    {"scenario": "bogus"},
    {"scenario": "purchase", "amount": "0"}, {"scenario": "purchase", "amount": "-5"},
    {"scenario": "purchase", "amount": "nan"}, {"scenario": "purchase", "amount": "1e12"},
    {"scenario": "purchase", "amount": "abc"}])
def test_bad_params_400(client, q):
    assert get(client, **q).status_code == 400


def test_profiles(client):
    assert client.get("/api/weather/ghost").status_code == 404
    assert client.get("/api/weather/bad name!").status_code == 400


def test_opening_cannot_be_overridden(client):
    base = get(client).json()["buffer"]
    r = client.get("/api/weather/priya", params={"opening": "0", "balance": "0"}).json()
    assert r["buffer"] == base
    with pytest.raises(ValueError):
        service.weather_params({"opening": 5})


def test_state_has_weather(client):
    w = client.get("/api/state/priya").json()["weather"]
    assert len(w["calendar"]) == 60 and "headline" in w


def test_graph_route_emits_meteorologist_events(client):
    run_id = get(client).json()["run_id"]
    ev = [e for e in client.get(f"/trace/{run_id}").json() if e["agent"] == "meteorologist"]
    assert {"start", "tool", "result"} <= {e["kind"] for e in ev}
    assert ev[-1]["summary"] == get(client).json()["headline_text"]


# ---- meteorologist unit tests -------------------------------------------------------------

def test_scenario_flows_salary_skip(priya):
    model, _ = priya
    f = M.scenario_flows(model, "salary_skip", None, 60)
    sal = max((r for r in model.recurring if r["amount"] > 0), key=lambda r: r["amount"])
    assert f and all(x[1] == -sal["amount"] and x[2] == "salary skipped" for x in f)
    for x in f:
        d = pd.Timestamp(x[0])
        assert d.day == min(sal["day"], d.days_in_month) and model.last_date < d <= model.last_date + pd.Timedelta(days=60)


def test_scenario_flows_no_income_and_note(priya):
    model, bal = priya
    import dataclasses
    m2 = dataclasses.replace(model, recurring=[r for r in model.recurring if r["amount"] < 0])
    assert M.scenario_flows(m2, "salary_skip", None, 60) == []
    out = M.run(m2, bal, {"scenario": "salary_skip"})
    assert out["scenario_flows"] == [] and out["scenario_note"]
    assert M.scenario_flows(model, "none", None, 60) == []
    with pytest.raises(ValueError):
        M.scenario_flows(model, "bogus", None, 60)
    for bad in (0, -1, float("nan"), float("inf"), 2e9, True):
        with pytest.raises(ValueError):
            M.scenario_flows(model, "purchase", bad, 60)


def test_umbrellas_skip_non_actionable(priya):
    model, bal = priya
    for sc in ("salary_skip", "emergency", "purchase"):
        rep = M.run(model, bal, {"scenario": sc})
        assert all(u["category"] not in W.NON_ACTIONABLE for u in rep["umbrellas"])
    assert W.NON_ACTIONABLE == {"uncategorized", "other", "health", "rent", "income", "utilities"}


def test_storm_warning_rec_names_actionable_category():
    from eval.generate import load_persona
    from finpilot.agents import advisor
    from finpilot.agents.categorizer import categorize_df
    p = load_persona("priya")
    df = categorize_df(p.df)
    model, bal = fit(df), current_balance(df, p.opening_balance)
    rep = W.report(model, bal - 90000)
    assert rep["storms"] and rep["umbrellas"]
    cands = advisor.build_candidates([], df, None, None, weather=rep, ctx={"opening": p.opening_balance - 90000})
    sw = [c for c in cands if c["trigger_type"] == "storm_warning"]
    assert sw
    for c in sw:
        assert advisor.template_text(c) and \
            all(cat not in advisor.template_text(c).replace(" ", "_") for cat in W.NON_ACTIONABLE)


def test_llm_narration_grounded(priya):
    model, bal = priya
    base = M.run(model, bal, SHOCK)
    good = base["headline_text"]
    llm = FakeLLM(lambda s, u: {"headline_text": good + " Stay alert.", "autopsy_text": base["autopsy_text"],
                                "umbrella_text": base["umbrella_text"]})
    assert M.run(model, bal, SHOCK, llm)["headline_text"] == good + " Stay alert."
    worst = W.label(base["headline"]["worst_p"])
    assert worst in ("rain", "storm")
    bad = {"headline_text": "There is a 97% chance of a ₹987,654 shortfall.",   # invented numbers
           "autopsy_text": "Skies are clear and there is no risk.",              # contradicts worst label
           "umbrella_text": "Cut dining by 33%."}                               # invented number
    out = M.run(model, bal, SHOCK, FakeLLM(lambda s, u: bad))
    for k in bad:
        assert out[k] == base[k]


def test_label_contradiction_both_ways(priya):
    model, bal = priya
    calm = M.run(model, bal, {})
    assert W.label(calm["headline"]["worst_p"]) == "clear"
    out = M.run(model, bal, {}, FakeLLM(lambda s, u: {"headline_text": "A violent storm is coming.",
                                                      "autopsy_text": "Heavy rain ahead.", "umbrella_text": "x"}))
    assert out["headline_text"] == calm["headline_text"] and out["autopsy_text"] == calm["autopsy_text"]
    assert not M.label_consistent("clear skies", "storm") and M.label_consistent("storm", "storm")
    assert not M.label_consistent("no risk at all", "rain") and not M.label_consistent(None, "rain")


@pytest.mark.parametrize("bad", [[], None, "text", 5, {"headline_text": 5, "autopsy_text": None, "umbrella_text": []},
                                 {"headline_text": ""}, {}])
def test_malformed_llm_falls_back(priya, bad):
    model, bal = priya
    base = M.run(model, bal, {"scenario": "purchase"})
    out = M.run(model, bal, {"scenario": "purchase"}, FakeLLM(lambda s, u: bad))
    for k in ("headline_text", "autopsy_text", "umbrella_text"):
        assert out[k] == base[k] and out[k].strip()


def test_llm_exception_falls_back(priya):
    model, bal = priya

    def boom(s, u):
        raise LLMUnavailable("down")
    out = M.run(model, bal, {}, FakeLLM(boom))
    assert out["headline_text"]


def test_llm_prompt_is_numeric_report_only(priya):
    model, bal = priya
    llm = FakeLLM(lambda s, u: {})
    M.run(model, bal, {"scenario": "purchase"}, llm)
    payload = json.loads(llm.calls[0][1])
    assert set(payload) == {"buffer", "lifestyle", "horizon", "headline", "storms", "autopsy", "umbrellas"}
