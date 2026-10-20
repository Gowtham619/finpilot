"""Verifiable weather/forecast claims: recompute vs the engine, ctx handling, advisor and graph wiring."""
import json

import pytest

from eval.generate import load_persona
from finpilot import graph
from finpilot.agents import advisor as AD
from finpilot.agents import verifier as VF
from finpilot.agents.categorizer import categorize_df
from finpilot.engines import forecast as F
from finpilot.engines import verify as V
from finpilot.engines import weather as W
from finpilot.engines.analyze import analyze_all
from finpilot.trace import Trace

BAL = 10_000.0  # target current balance that gives priya a storm (14,000 only gives rain)


@pytest.fixture(scope="module")
def world():
    df = categorize_df(load_persona("priya").df.copy())
    opening = round(BAL - float(df["amount"].sum()), 2)  # balance = opening + sum(amount)
    ctx = {"opening": opening}
    model = F.fit(df)
    rep = W.report(model, F.current_balance(df, opening))
    assert rep["storms"], "fixture must contain a storm"
    return df, ctx, model, rep


def _cands(world, **kw):
    df, ctx, model, rep = world
    fc = {"summary": {"p_overdraft": 0.37, "floor_p50": -1234.5, "end_p10": 2000.0, "end_p50": 9000.0,
                      "end_p90": 15000.0, "risk_day": "2026-09-14"}, "horizon": 60, "goals": []}
    return AD.build_candidates(analyze_all(df), df, fc, [], weather=rep, ctx=ctx, **kw)


def test_breach_probability_matches_engine(world):
    df, ctx, _, rep = world
    v = V.recompute({"kind": "breach_probability", "horizon": 60}, df, ctx)
    assert v == pytest.approx(rep["headline"]["p_any_breach"] * 100, abs=0.06)


def test_storm_peak_matches_engine(world):
    df, ctx, _, rep = world
    s = rep["storms"][0]
    for d in (s["start"], s["peak_date"], s["end"]):
        assert V.recompute({"kind": "storm_peak", "date": d}, df, ctx) == pytest.approx(s["peak_p"] * 100)
    outside = rep["calendar"][0]["date"]
    assert V.recompute({"kind": "storm_peak", "date": outside}, df, ctx) is None


def test_storm_peak_after_cut_matches_engine(world):
    df, ctx, model, rep = world
    s = rep["storms"][0]
    cal = W.risk_calendar(model, F.current_balance(df, ctx["opening"]), cuts={"shopping": 0.3})
    want = max(c["p_below_buffer"] for c in cal if s["start"] <= c["date"] <= s["end"]) * 100
    got = V.recompute({"kind": "storm_peak_after_cut", "date": s["peak_date"], "category": "shopping",
                       "cut_pct": 30}, df, ctx)
    assert got == pytest.approx(want)
    assert got < s["peak_p"] * 100


BAD = [{"kind": "breach_probability"}, {"kind": "breach_probability", "horizon": 0},
       {"kind": "breach_probability", "horizon": "x"}, {"kind": "breach_probability", "horizon": float("nan")},
       {"kind": "storm_peak"}, {"kind": "storm_peak", "date": "not-a-date"},
       {"kind": "storm_peak_after_cut", "date": "2026-10-30", "category": "shopping"},
       {"kind": "storm_peak_after_cut", "date": "2026-10-30", "category": "shopping", "cut_pct": 250},
       {"kind": "storm_peak_after_cut", "date": "2026-10-30", "category": "shopping", "cut_pct": float("nan")},
       {"kind": "storm_peak_after_cut", "date": "2026-10-30", "category": "nope", "cut_pct": 10}]


@pytest.mark.parametrize("bad", BAD, ids=range(len(BAD)))
def test_malformed_weather_claims_are_unverifiable(world, bad):
    df, ctx, _, _ = world
    assert V.recompute(bad, df, ctx) is None
    assert V.check_claim(dict(bad, value=50.0), df, ctx)["ok"] is False


@pytest.mark.parametrize("ctx", [None, {}, {"opening": None}, {"opening": float("nan")}, {"opening": "x"}, "junk"])
def test_missing_ctx_is_unverifiable_for_new_kinds_only(world, ctx):
    df, _, _, rep = world
    for c in ({"kind": "breach_probability", "horizon": 60},
              {"kind": "storm_peak", "date": rep["storms"][0]["peak_date"]}):
        assert V.recompute(c, df, ctx) is None
    merchant = df[df["amount"] < 0]["merchant"].iloc[0]
    assert V.recompute({"kind": "monthly_cost", "merchant": merchant}, df, ctx) is not None


def test_tolerance_for_percent_kinds(world):
    df, ctx, _, rep = world
    pk = rep["storms"][0]["peak_p"] * 100
    c = {"kind": "storm_peak", "date": rep["storms"][0]["peak_date"]}
    assert V.check_claim(dict(c, value=pk + 0.9), df, ctx)["ok"]
    assert not V.check_claim(dict(c, value=pk + 1.5), df, ctx)["ok"]


def test_percent_claims_accepted_in_text_but_wrong_percent_is_not(world):
    claims = [{"kind": "storm_peak", "date": "2026-10-30", "value": 42.1}]
    assert V.ungrounded_numbers("peak 42% chance", claims, {}) == []
    assert V.ungrounded_numbers("peak 42.1% chance", claims, {}) == []
    assert V.ungrounded_numbers("peak 60% chance", claims, {}) == [60.0]


def test_storm_warning_and_forecast_risk_candidates(world):
    cands = _cands(world)
    sw = [c for c in cands if c["trigger_type"] == "storm_warning"]
    fr = [c for c in cands if c["trigger_type"] == "forecast_risk"]
    assert len(sw) == 1 and len(fr) == 1
    assert {c["kind"] for c in sw[0]["claims"]} >= {"storm_peak"}
    assert [c["kind"] for c in fr[0]["claims"]] == ["breach_probability"]
    json.dumps(cands)


def test_weather_templates_pass_audit_and_are_recomputed(world):
    df, ctx, _, _ = world
    for c in _cands(world):
        if c["trigger_type"] not in ("storm_warning", "forecast_risk"):
            continue
        rec = AD.advise([c], None)[0]
        assert rec["verification"] == "recomputed", c["trigger_type"]
        assert VF.audit(rec, df, ctx) == [], (rec["text"], VF.audit(rec, df, ctx))
        kept, dropped = VF.verify_all([rec], df, ctx=ctx)
        assert kept and not dropped and kept[0]["verification"] == "recomputed"


def test_storm_warning_without_ctx_is_dropped_not_crashed(world):
    df, _, _, _ = world
    c = next(c for c in _cands(world) if c["trigger_type"] == "storm_warning")
    kept, dropped = VF.verify_all(AD.advise([c], None), df)
    assert dropped and not kept


def test_lie_about_storm_peak_is_caught_and_corrected(world):
    df, ctx, _, rep = world
    c = next(c for c in _cands(world) if c["trigger_type"] == "storm_warning")
    rec = AD.advise([c], None)[0]
    true_peak = next(x for x in rec["claims"] if x["kind"] == "storm_peak")["value"]
    rec["claims"] = [dict(x, value=95.0) if x["kind"] == "storm_peak" else x for x in rec["claims"]]
    rec["text"] = rec["text"].replace(f"{true_peak:.0f}%", "95%", 1)
    assert any(i["type"] == "mismatch" for i in VF.audit(rec, df, ctx))
    kept, dropped = VF.verify_all([rec], df, ctx=ctx)
    assert kept and not dropped and "mismatch caught" in kept[0]["verifier_notes"]
    fixed = next(x for x in kept[0]["claims"] if x["kind"] == "storm_peak")
    assert fixed["value"] == pytest.approx(true_peak, abs=1.0)
    assert "95%" not in kept[0]["text"]


def test_without_weather_forecast_risk_is_unchanged(world):
    df, _, _, _ = world
    fc = {"summary": {"p_overdraft": 0.37, "floor_p50": -1.0, "end_p10": 1.0, "end_p50": 2.0, "end_p90": 3.0,
                      "risk_day": "2026-09-14"}, "horizon": 60, "goals": []}
    c = next(c for c in AD.build_candidates([], df, fc) if c["trigger_type"] == "forecast_risk")
    assert c["claims"] == []


def test_pipeline_has_verified_storm_warning():
    df = load_persona("priya").df.copy()
    opening = round(BAL - float(df["amount"].sum()), 2)
    out = graph.build_graph(None, Trace()).invoke(
        {"run_id": "w1", "profile": "t", "request": {"type": "pipeline"}, "df": df, "opening": opening,
         "corrections": {}, "goals": []})
    json.dumps(out["weather"])
    assert "model" not in out["forecast"]
    sw = [r for r in out["recs"] if r["trigger_type"] == "storm_warning"]
    assert sw and sw[0]["verified"] and sw[0]["verification"] == "recomputed"
    assert not any(r["trigger_type"] in ("storm_warning", "forecast_risk") for r in out["dropped"])


def test_pipeline_has_verified_forecast_risk_when_overdraft_likely():
    df = load_persona("priya").df.copy()
    opening = round(1_000.0 - float(df["amount"].sum()), 2)
    out = graph.build_graph(None, Trace()).invoke(
        {"run_id": "w3", "profile": "t", "request": {"type": "pipeline"}, "df": df, "opening": opening,
         "corrections": {}, "goals": []})
    fr = [r for r in out["recs"] if r["trigger_type"] == "forecast_risk"]
    assert fr and fr[0]["verified"] and fr[0]["verification"] == "recomputed"
    assert fr[0]["claims"][0]["kind"] == "breach_probability"


def test_clear_skies_pipeline_has_no_storm_warning():
    df = load_persona("priya").df.copy()
    out = graph.build_graph(None, Trace()).invoke(
        {"run_id": "w2", "profile": "t", "request": {"type": "pipeline"}, "df": df, "opening": 40000.0,
         "corrections": {}, "goals": []})
    assert out["weather"]["storms"] == []
    assert not any(r["trigger_type"] == "storm_warning" for r in out["recs"])
