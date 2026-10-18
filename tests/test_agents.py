import pytest
import pandas as pd
from eval.generate import generate
from finpilot.agents import analyzer, forecaster, simulator
from finpilot.agents.common import grounded_or
from finpilot.engines.categorize import normalize_merchant
from finpilot.llm import LLMUnavailable
from tests.conftest import FakeLLM


@pytest.fixture(scope="module")
def world():
    p = generate(seed=0)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    return p, df


def test_analyzer_adds_summaries_and_overview(world):
    _, df = world
    out = analyzer.run(df, None)
    assert out["findings"] and all(f["summary"] for f in out["findings"])
    assert out["overview"]


def test_analyzer_llm_priority_is_validated(world):
    _, df = world
    ids = [f["id"] for f in analyzer.run(df, None)["findings"]]
    llm = FakeLLM(lambda s, u: {"overview": "Look at HOTSTAR.", "priority": [ids[-1], "bogus"]})
    out = analyzer.run(df, llm)
    assert out["findings"][0]["id"] == ids[-1]
    assert out["overview"] == "Look at HOTSTAR."


def test_forecaster_output_shape(world):
    p, df = world
    goals = [{"name": "Trip", "target": 200000, "deadline": "2027-03-01"}]
    out = forecaster.run(df, p.opening_balance, goals, None)
    assert len(out["bands"]["p50"]) == 60
    assert 0 <= out["summary"]["p_overdraft"] <= 1
    assert out["goals"][0]["name"] == "Trip" and "p_met" in out["goals"][0]
    assert out["narrative"]


def test_grounded_or_rejects_invented_numbers():
    assert grounded_or("Median ₹1000.", {"x": 1000}, "fb") == "Median ₹1000."
    assert grounded_or("Median ₹7777.", {"x": 1000}, "fb") == "fb"


def test_afford_answer(world):
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), None)
    req = {"amount": 80000, "date": str(f["model"].last_date.date() + pd.Timedelta(days=5)), "financed_months": 0}
    out = simulator.afford(f["model"], f["balance"], req, None)
    assert out["verdict"] in {"safe", "risky", "unsafe"}
    assert out["sim"]["with_purchase"]["p_overdraft"] >= out["sim"]["baseline"]["p_overdraft"]
    assert out["best_dates"] and out["narrative"]


def test_duel_answer(world):
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), None)
    d0 = str(f["model"].last_date.date() + pd.Timedelta(days=5))
    req = {"a": {"name": "Buy now", "amount": 150000, "date": d0},
           "b": {"name": "Skip", "amount": 0, "date": d0}}
    out = simulator.run_duel(f["model"], f["balance"], req, None)
    assert out["winner"] == "b" and out["narrative"]


def test_afford_out_of_window_date_raises(world):
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), None)
    req = {"amount": 1000, "date": str(f["model"].last_date.date() - pd.Timedelta(days=5)), "financed_months": 0}
    with pytest.raises(ValueError):
        simulator.afford(f["model"], f["balance"], req, None)


def test_afford_without_candidate_dates_does_not_crash(world, monkeypatch):
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), None)
    monkeypatch.setattr(simulator.S, "best_date", lambda *a, **k: [])
    req = {"amount": 5000, "date": str(f["model"].last_date.date() + pd.Timedelta(days=5)), "financed_months": 0}
    out = simulator.afford(f["model"], f["balance"], req, None)
    assert out["best_dates"] == [] and out["narrative"]


def test_fallback_narratives_pass_own_grounding_check(world):
    from finpilot.engines.verify import ungrounded_numbers
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), None)
    ev = {"s": f["summary"], "h": f["horizon"], "b": f["balance"]}
    assert ungrounded_numbers(f["narrative"], [], ev) == []
    d0 = str(f["model"].last_date.date() + pd.Timedelta(days=5))
    for amount in (80000, 5000, 400000):
        for fm in (0, 6):
            req = {"amount": amount, "date": d0, "financed_months": fm}
            out = simulator.afford(f["model"], f["balance"], req, None)
            ev = {"sim": out["sim"], "best": out["best_dates"], "fix": out["fix"], "amount": amount}
            assert ungrounded_numbers(out["narrative"], [], ev) == [], out["narrative"]
    req = {"a": {"name": "Buy now", "amount": 150000, "date": d0}, "b": {"name": "Skip", "amount": 0, "date": d0}}
    out = simulator.run_duel(f["model"], f["balance"], req, None)
    ev = {"a": out["a"]["summary"], "b": out["b"]["summary"], "h": out["horizon"]}
    assert ungrounded_numbers(out["narrative"], [], ev) == []
