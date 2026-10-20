"""Final-review trust fixes: strict, risk-consistent narration on the forecaster / afford / duel / weather paths,
the duel winner check, and the scenario-aware autopsy text. All LLM output here is scripted (FakeLLM)."""
import copy

import pandas as pd
import pytest
from eval.generate import load_persona
from finpilot.agents import forecaster, meteorologist as M, simulator
from finpilot.agents.categorizer import categorize_df
from finpilot.engines import simulate as S
from finpilot.engines.forecast import current_balance, fit
from finpilot.engines.verify import ungrounded_numbers
from tests.conftest import FakeLLM


def say(text, key="narrative"):
    return FakeLLM(lambda s, u: {key: text})


@pytest.fixture(scope="module")
def priya():
    p = load_persona("priya")
    df = categorize_df(p.df)
    return p, df, fit(df), current_balance(df, p.opening_balance)


@pytest.fixture(scope="module")
def reports(priya):
    _, _, model, bal = priya
    return {sc: M.run(model, bal, {"scenario": sc}) for sc in M.SCENARIOS}


# ---------------------------------------------------------------- forecaster
def _risky_opening(p, df):
    for cut in (55000, 65000, 75000, 85000, 95000):
        f = forecaster.run(df, p.opening_balance - cut, (), None)
        if 0.02 <= f["summary"]["p_overdraft"] < 0.5:
            return p.opening_balance - cut, f
    raise AssertionError("no opening with a 2-50% overdraft risk")


def test_forecaster_rejects_number_free_no_risk_claim(priya):
    p, df, _, _ = priya
    opening, fb = _risky_opening(p, df)
    out = forecaster.run(df, opening, (), say("You will definitely not go overdrawn; zero chance."))
    assert out["narrative"] == fb["narrative"]
    ok = forecaster.run(df, opening, (), say("Spending looks steady, but an overdraft is possible."))
    assert ok["narrative"] == "Spending looks steady, but an overdraft is possible."


def test_forecaster_is_strict_and_fallback_passes_its_own_strict_check(priya):
    p, df, _, _ = priya
    for opening in (p.opening_balance, _risky_opening(p, df)[0]):
        f = forecaster.run(df, opening, (), None)
        ev = forecaster.strict_evidence(f["summary"], f["horizon"], f["balance"])
        assert ungrounded_numbers(f["narrative"], [], ev, strict=True) == [], f["narrative"]
    # a percent that only matches a raw fraction (not an intentional percent) is rejected in strict mode
    f = forecaster.run(df, p.opening_balance, (), None)
    bad = forecaster.run(df, p.opening_balance, (), say("Ninety percent of futures stay positive."))
    assert bad["narrative"] == f["narrative"]


# ---------------------------------------------------------------- afford
def test_afford_rejects_go_ahead_when_unsafe(priya):
    _, _, model, bal = priya
    d0 = str((model.last_date + pd.Timedelta(days=5)).date())
    req = {"amount": 400000, "date": d0, "financed_months": 0}
    base = simulator.afford(model, bal, dict(req), None)
    assert base["verdict"] == "unsafe"
    out = simulator.afford(model, bal, dict(req), say("Go ahead, this purchase will not hurt you at all."))
    assert out["narrative"] == base["narrative"]


def test_afford_fallbacks_pass_strict_check_and_read_cleanly(priya):
    _, _, model, bal = priya
    d0 = str((model.last_date + pd.Timedelta(days=5)).date())
    for amount in (5000, 80000, 400000):
        for fm in (0, 6):
            out = simulator.afford(model, bal, {"amount": amount, "date": d0, "financed_months": fm}, None)
            ev = simulator.afford_strict_evidence(out["sim"], out["best_dates"], out["fix"], amount, fm)
            text = out["narrative"].split(" (", 1)[0] if out.get("notice") else out["narrative"]
            assert ungrounded_numbers(text, [], ev, strict=True) == [], out["narrative"]
            assert "(overdrawn))" not in out["narrative"] and "(median balance" not in out["narrative"]


# ---------------------------------------------------------------- duel
def _duel_req(model, a_amt=150000, b_amt=0):
    d0 = str((model.last_date + pd.Timedelta(days=5)).date())
    return {"a": {"name": "Buy now", "amount": a_amt, "date": d0}, "b": {"name": "Skip", "amount": b_amt, "date": d0}}


def test_duel_rejects_narrative_naming_the_wrong_winner(priya):
    _, _, model, bal = priya
    base = simulator.run_duel(model, bal, _duel_req(model), None)
    assert base["winner"] == "b"
    for lie in ("Buy now wins clearly.", "Skip is fine but Buy now is better.", "The winner is Buy now.",
                "Both are close; no clear winner."):
        out = simulator.run_duel(model, bal, _duel_req(model), say(lie))
        assert out["narrative"] == base["reason"], lie
    good = "Skip wins: skipping keeps the overdraft risk lower than buying now."
    assert simulator.run_duel(model, bal, _duel_req(model), say(good))["narrative"] == good


def test_duel_tie_rejects_a_declared_winner(priya):
    _, _, model, bal = priya
    base = simulator.run_duel(model, bal, _duel_req(model, 0, 0), None)
    assert base["winner"] == "tie"
    out = simulator.run_duel(model, bal, _duel_req(model, 0, 0), say("Buy now is better."))
    assert out["narrative"] == base["reason"]
    ok = "Neither option changes anything: they are identical."
    assert simulator.run_duel(model, bal, _duel_req(model, 0, 0), say(ok))["narrative"] == ok


def test_duel_strict_grounding_and_fallback_passes_it(priya):
    _, _, model, bal = priya
    base = simulator.run_duel(model, bal, _duel_req(model), None)
    ev = simulator.duel_evidence(base)
    checked = base["reason"].replace("Buy now", " ").replace("Skip", " ")
    assert ungrounded_numbers(checked, [], ev, strict=True) == [], base["reason"]
    out = simulator.run_duel(model, bal, _duel_req(model), say("Skip wins; ninety percent of futures are fine."))
    assert out["narrative"] == base["reason"]


def test_small_probabilities_read_less_than_one_percent():
    assert [S._pct(x) for x in (0, 0.003, 0.005, 0.126)] == ["0%", "<1%", "1%", "13%"]


# ---------------------------------------------------------------- weather narration
def test_weather_rejects_fine_wording_in_a_storm(priya, reports):
    _, _, model, bal = priya
    rep = reports["purchase"]
    assert rep["storms"]
    out = M.run(model, bal, {"scenario": "purchase"},
                FakeLLM(lambda s, u: {"headline_text": "Skies look fine; no breach expected.",
                                      "autopsy_text": "Nothing to worry about.",
                                      "umbrella_text": "You will definitely not go overdrawn; zero chance."}))
    for k in ("headline_text", "autopsy_text", "umbrella_text"):
        assert out[k] == rep[k], k


def test_weather_quantity_words_checked_against_failing_share(reports):
    rep = copy.deepcopy(reports["purchase"])
    rep["headline"]["p_any_breach"] = 0.2
    llm = FakeLLM(lambda s, u: {"headline_text": "Half of futures fail."})
    assert M.narrate_report(rep, llm)["headline_text"] != "Half of futures fail."
    rep["headline"]["p_any_breach"] = 0.5
    assert M.narrate_report(rep, llm)["headline_text"] == "Half of futures fail."


# ---------------------------------------------------------------- autopsy text (I3)
def test_autopsy_text_purchase_scenario(reports):
    t = reports["purchase"]["autopsy_text"]
    assert t.startswith("Every simulated future includes the planned ₹94,000 purchase on Oct 10; among them, the "
                        "failing ones also spent more on groceries (46% of the gap).")
    assert "Largest bill before then: SUNRISE APARTMENTS RENT ₹18,000" in t
    assert "planned: planned" not in t and "Earliest outflow" not in t and "NETFLIX" not in t


def test_autopsy_text_emergency_and_none(reports):
    assert reports["emergency"]["autopsy_text"].startswith(
        "Every simulated future includes the ₹69,000 emergency expense on Oct 15;")
    assert reports["none"]["autopsy_text"] == "No failing futures, so there is nothing to explain."


def test_autopsy_text_salary_skip_is_missing_income_not_an_outflow(reports):
    rep = reports["salary_skip"]
    t = rep["autopsy_text"]
    assert t.startswith("Salary of ₹62,000 does not arrive on Oct 1 and Nov 1;")
    assert "outflow" not in t.lower() and "salary skipped" not in t
    assert not any("salary" in b["merchant"].lower() for b in rep["autopsy"]["bills"])


def test_autopsy_texts_pass_their_own_strict_check(reports):
    for sc, rep in reports.items():
        assert not ungrounded_numbers(rep["autopsy_text"], [], M.strict_evidence(rep), strict=True), sc
        assert not ungrounded_numbers(rep["headline_text"], [], M.strict_evidence(rep), strict=True), sc


def test_autopsy_names_the_largest_bill_not_the_earliest():
    rep = {"headline": {"p_any_breach": .3, "worst_day": "2026-10-20", "worst_p": .3, "n_storm_days": 0},
           "buffer": 1000.0, "horizon": 60, "storms": [], "umbrellas": [], "scenario": "none", "scenario_flows": [],
           "autopsy": {"n_failing": 300, "n_paths": 1000, "causes": [], "insufficient_failing": False,
                       "first_breach": {"date": "2026-10-20"},
                       "bills": [{"merchant": "NETFLIX", "date": "2026-10-02", "amount": -649.0},
                                 {"merchant": "RENT", "date": "2026-10-05", "amount": -18000.0}]}}
    t = M.fallback_texts(rep)["autopsy_text"]
    assert "Largest bill before then: RENT ₹18,000 on Oct 5." in t and "NETFLIX" not in t


def test_paydays_in_weather_report(reports):
    assert reports["none"]["paydays"] == ["2026-10-01", "2026-11-01"]
    assert reports["purchase"]["paydays"] == ["2026-10-01", "2026-11-01"]
    assert reports["salary_skip"]["paydays"] == []  # the skipped pay dates do not happen


def test_planned_purchase_bill_is_not_double_prefixed(reports):
    names = [b["merchant"] for b in reports["purchase"]["autopsy"]["bills"]]
    assert "planned purchase" in names and not any(n.startswith("planned: planned") for n in names)
