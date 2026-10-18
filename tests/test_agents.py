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
    assert ungrounded_numbers(out["narrative"], [], simulator.duel_evidence(out)) == []


def _ids(df):
    return [f["id"] for f in analyzer.run(df, None)["findings"]]


def test_analyzer_overview_with_invented_number_falls_back(world):
    _, df = world
    base = analyzer.run(df, None)["overview"]
    llm = FakeLLM(lambda s, u: {"overview": "Cancel it and save ₹91357 a month.", "priority": []})
    assert analyzer.run(df, llm)["overview"] == base


@pytest.mark.parametrize("pri", [None, [["a"]], "x", 5, [{"k": 1}, None, 3]])
def test_analyzer_malformed_priority_never_crashes(world, pri):
    _, df = world
    ids = _ids(df)
    out = analyzer.run(df, FakeLLM(lambda s, u: {"overview": "ok", "priority": pri}))
    assert [f["id"] for f in out["findings"]] == ids


def test_analyzer_priority_duplicates_and_non_dict(world):
    _, df = world
    ids = _ids(df)
    out = analyzer.run(df, FakeLLM(lambda s, u: {"priority": [ids[-1], ids[-1], ids[0]]}))
    got = [f["id"] for f in out["findings"]]
    assert got[0] == ids[-1] and got[1] == ids[0] and sorted(got) == sorted(ids)
    assert analyzer.run(df, FakeLLM(lambda s, u: ["nope"]))["findings"]


def test_forecaster_bad_goals_yield_error_entries(world):
    p, df = world
    goals = [{"name": "Bad", "target": 1000, "deadline": "not a date"},
             {"name": "NoTarget", "deadline": "2027-01-01"},
             "junk",
             {"name": "Str", "target": "abc", "deadline": "2027-01-01"},
             {"name": "Past", "target": 1000, "deadline": "2020-01-01"}]
    out = forecaster.run(df, p.opening_balance, goals, None)["goals"]
    assert [("error" in g) for g in out] == [True, True, True, True, False]
    assert out[2]["name"] == "?" and out[1]["name"] == "NoTarget"
    assert out[4]["horizon"] == 1 and "p_met" in out[4]


def test_advisor_skips_goal_error_entries(world):
    from finpilot.agents import advisor
    out = advisor.build_candidates([], world[1], goals=[{"name": "X", "error": "bad"}, "junk"])
    assert not any(c["type"] == "goal_behind" for c in out)


def test_llm_invented_numbers_fall_back_end_to_end(world):
    p, df = world
    bad = FakeLLM(lambda s, u: {"narrative": "You will have ₹7777777 left."})
    fb = forecaster.run(df, p.opening_balance, (), None)
    f = forecaster.run(df, p.opening_balance, (), bad)
    assert f["narrative"] == fb["narrative"] and "7777777" not in f["narrative"]
    d0 = str(fb["model"].last_date.date() + pd.Timedelta(days=5))
    req = {"amount": 80000, "date": d0, "financed_months": 0}
    assert (simulator.afford(fb["model"], fb["balance"], req, bad)["narrative"]
            == simulator.afford(fb["model"], fb["balance"], req, None)["narrative"])
    dreq = {"a": {"name": "Buy now", "amount": 150000, "date": d0}, "b": {"name": "Skip", "amount": 0, "date": d0}}
    assert (simulator.run_duel(fb["model"], fb["balance"], dreq, bad)["narrative"]
            == simulator.run_duel(fb["model"], fb["balance"], dreq, None)["narrative"])


# ---- final review fixes: I1 horizon, I2 alternative date, I3 verdict words / adversarial narration ----
def _chosen(p, floor, date="2026-10-01"):
    return {"date": date, "p_overdraft": p, "floor_p50": floor}


@pytest.mark.parametrize("best,chosen,expect", [
    ([_chosen(1.0, -5000, "2026-10-08")], _chosen(1.0, -9000), False),       # 100% vs 100%: useless
    ([_chosen(0.0, 20000, "2026-10-08")], _chosen(0.0, 20500), False),       # 0% vs 0%, no better floor
    ([_chosen(0.0, 21000, "2026-10-08")], _chosen(0.0, 20500), False),       # floor gain under the margin
    ([_chosen(0.0, 22000, "2026-10-08")], _chosen(0.0, 20500), True),        # visible floor gain
    ([_chosen(0.2, 1000, "2026-10-08")], _chosen(0.4, 500), True),           # strictly lower risk
    ([_chosen(0.4, 9000, "2026-10-08")], _chosen(0.2, 500), False),          # higher risk
    ([_chosen(0.2, 500)], _chosen(0.2, 500), False),                         # same date
    ([], _chosen(0.2, 500), False),
])
def test_better_date_rule(best, chosen, expect):
    assert (simulator.better_date(best, chosen) is not None) is expect


def test_afford_text_only_suggests_really_better_dates(world):
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), None)
    d0 = str(f["model"].last_date.date() + pd.Timedelta(days=1))
    for amount in (5000, f["balance"] * 3):
        out = simulator.afford(f["model"], f["balance"], {"amount": amount, "date": d0, "financed_months": 0})
        chosen = {"date": out["sim"]["date"], **out["sim"]["with_purchase"]}
        alt = simulator.better_date(out["best_dates"], chosen)
        assert ("date tested" in out["narrative"]) is (alt is not None), out["narrative"]


def test_afford_horizon_covers_installments():
    assert simulator.afford_horizon(0) == 60 and simulator.afford_horizon(1) == 62
    assert simulator.afford_horizon(6) == 217 and simulator.afford_horizon(11) == 365
    assert simulator.afford_horizon(60) == 365


@pytest.mark.parametrize("text,verdict,ok", [
    ("Yes, this is completely safe.", "unsafe", False),
    ("This purchase is unsafe.", "unsafe", True),
    ("Risky but possible.", "risky", True),
    ("It is not safe.", "unsafe", False),
    ("No verdict words here.", "safe", True),
])
def test_verdict_consistency(text, verdict, ok):
    assert simulator.verdict_consistent(text, verdict) is ok


def test_afford_rejects_narrative_with_wrong_verdict(world):
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), None)
    d0 = str(f["model"].last_date.date() + pd.Timedelta(days=1))
    req = {"amount": f["balance"] * 3, "date": d0, "financed_months": 0}
    out = simulator.afford(f["model"], f["balance"], dict(req),
                           FakeLLM(lambda s, u: {"narrative": "Yes, this is completely safe."}))
    assert out["verdict"] == "unsafe" and "completely safe" not in out["narrative"]


def test_adversarial_narration_with_false_percent_is_rejected(world):
    p, df = world
    f = forecaster.run(df, p.opening_balance, (), FakeLLM(
        lambda s, u: {"narrative": "There is a 60% chance you go overdrawn within 30 days."}))
    if f["summary"]["p_overdraft"] != 0.6:
        assert "60%" not in f["narrative"]
