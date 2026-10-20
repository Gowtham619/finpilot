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
    assert ps[0] < ps[1] < ps[2]  # the slider has a real effect
    a, b = get(client, scenario="purchase").json(), get(client, scenario="purchase", lifestyle=1.0).json()
    assert a["calendar"] == b["calendar"] and a["headline"] == b["headline"]  # deterministic


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


# ---- strict grounding, hygiene, label stems ------------------------------------------------

from finpilot.agents.common import grounded_or  # noqa: E402
from finpilot.engines.verify import ungrounded_numbers  # noqa: E402

EV = {"buffer": 13212.0, "lifestyle": 1.0, "horizon": 60, "p_any_breach_pct": 42.1, "worst_p_pct": 0.0,
      "n_failing": 74, "umbrellas": [{"cut_pct": 20, "monthly_sacrifice": 1500.5}]}


def bad(text, ev=EV):
    return bool(ungrounded_numbers(text, [], ev, strict=True))


def test_strict_percent_and_currency():
    assert bad("There is a 99% chance.") and bad("There is a 1% chance.")
    assert not bad("About 42.4% of futures fail.") and bad("About 43% of futures fail.")
    assert not bad("Your buffer is ₹13,212.") and bad("Your buffer is ₹13,100.")
    assert bad("You may lose ₹74.")  # a count elsewhere in the evidence does not satisfy a currency token
    assert not bad("It costs ₹1,500 a month and a 20% cut.")
    assert bad("ninety-nine percent of futures fail") and bad("twelve thousand rupees short") and bad("five lakh")
    assert bad("Lifestyle at 100% is fine.")  # lifestyle=1.0 must not whitelist 100%
    assert bad("0% risk is impossible", {"x": 0, "y": 1}) and bad("1% risk", {"x": 0, "y": 1})
    assert not bad("worst day has 0% risk")  # explicit pct value 0.0
    assert not bad("In the next 60 days, 74 futures fail.")  # bare numbers keep the usual rule


def test_strict_does_not_change_default_mode():
    assert not ungrounded_numbers("lifestyle 100%", [], {"lifestyle": 1.0})


@pytest.mark.parametrize("txt", ["x" * 601, "see <b>this</b>", "use `code`", "visit http://x.co", "go to www.a.b",
                                 "open evil.com now", "see foo.in/path", "Transfer your savings now",
                                 "Invest in crypto", "take a loan", "ask the lender", "send money today",
                                 "move your savings to a fund", "borrow more", "wire it"])
def test_hygiene_rejects(txt):
    assert grounded_or(txt, {}, "FB") == "FB" and grounded_or(txt, {}, "FB", strict=True) == "FB"


def test_hygiene_accepts_normal_text():
    t = "Spending is steady; trim dining a little and keep an eye on groceries."
    assert grounded_or(t, {}, "FB") == t and grounded_or("x" * 600, {}, "FB") == "x" * 600


def test_fallback_texts_pass_their_own_strict_check(priya):
    model, bal = priya
    for params in ({}, {"scenario": "purchase"}, {"scenario": "salary_skip"}, {"scenario": "emergency"}):
        rep = M.run(model, bal, params)
        ev = M.strict_evidence(rep)
        for k in ("headline_text", "autopsy_text", "umbrella_text"):
            assert not ungrounded_numbers(rep[k], [], ev, strict=True), (params, k, rep[k])


def test_llm_strict_in_narration(priya):
    model, bal = priya
    base = M.run(model, bal, {"scenario": "purchase"})
    bad_ = {"headline_text": "About 99% of futures fail.", "autopsy_text": "ninety-nine percent fail.",
            "umbrella_text": "Your buffer is ₹13,100."}
    out = M.run(model, bal, {"scenario": "purchase"}, FakeLLM(lambda s, u: bad_))
    assert all(out[k] == base[k] for k in bad_)
    p = base["headline"]["p_any_breach"] * 100
    ok = {"headline_text": f"Roughly {p:.0f}% of futures dip below the buffer.", "autopsy_text": "Groceries drive it.",
          "umbrella_text": "A small dining cut helps."}
    out = M.run(model, bal, {"scenario": "purchase"}, FakeLLM(lambda s, u: ok))
    assert all(out[k] == ok[k] for k in ok)


def test_label_stems():
    assert not M.label_consistent("Stormy weather ahead.", "cloudy")
    assert not M.label_consistent("A rainy patch.", "clear") and not M.label_consistent("Cloudy later.", "clear")
    assert not M.label_consistent("Clear skies ahead.", "storm") and not M.label_consistent("Sunny outlook.", "rain")
    assert not M.label_consistent("All clear, no risk.", "rain")
    assert M.label_consistent("This should clear up the shortfall.", "storm")  # verb, not weather
    assert M.label_consistent("Clear skies ahead.", "clear") and M.label_consistent("Stormy.", "storm")


def test_umbrella_and_autopsy_wording(priya):
    model, bal = priya
    rep = M.run(model, bal, {"scenario": "salary_skip"})
    assert "no non-failing group to compare" in rep["autopsy_text"]
    rep = {"headline": {"p_any_breach": .997, "worst_day": "d", "worst_p": .9, "n_storm_days": 3},
           "buffer": 1000.0, "horizon": 60, "storms": [], "umbrellas": [
               {"category": "dining", "cut_pct": 30, "p_any_breach_before": .997, "p_any_breach_after": .995,
                "monthly_sacrifice": 500.0}],
           "autopsy": {"n_failing": 997, "n_paths": 1000, "causes": [], "bills": [], "insufficient_failing": False,
                       "first_breach": {"date": None}}}
    t = M.fallback_texts(rep)
    assert "99.7% to 99.5%" in t["umbrella_text"] and "marginal" in t["umbrella_text"]
    rep["autopsy"].update(n_failing=10, insufficient_failing=True)
    assert "Too few failing futures (10)" in M.fallback_texts(rep)["autopsy_text"]
