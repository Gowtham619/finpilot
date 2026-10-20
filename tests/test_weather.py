import json
import math

import numpy as np
import pandas as pd
import pytest

from eval.generate import generate, load_persona
from finpilot.engines import forecast as F
from finpilot.engines import weather as W
from finpilot.engines.categorize import normalize_merchant


def persona_model(name=None, **kw):
    p = load_persona(name) if name else generate(**kw)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    return F.fit(df), F.current_balance(df, p.opening_balance)


def toy(daily, recurring=(), cats=None):
    daily = np.asarray(daily, dtype=float)
    if daily.ndim == 1:
        daily = daily[:, None]
    cats = cats or [f"c{i}" for i in range(daily.shape[1])]
    idx = pd.date_range("2026-01-01", periods=len(daily))
    return F.Model(list(recurring), cats, daily, np.asarray(idx.weekday), idx[-1])


def driver_model():
    """'big' is bimodal (0 or 300/day) and drives failure; 'flat' is a constant 10/day."""
    rng = np.random.default_rng(1)
    big = np.where(rng.random(90) < 0.5, 300.0, 0.0)
    return toy(np.column_stack([big, np.full(90, 10.0)]), cats=["big", "flat"])


def cal_with(ps, start="2026-03-01"):
    d = pd.date_range(start, periods=len(ps))
    return [{"date": str(x.date()), "p_below_buffer": p, "label": W.label(p), "p10": 0.0, "p50": 0.0, "p90": 0.0}
            for x, p in zip(d, ps)]


# ---- forecast.monte_carlo extensions
def test_spend_scale_validation():
    m, bal = persona_model(seed=0)
    for bad in (float("nan"), float("inf"), -1, 3.1, "x", None):
        with pytest.raises(ValueError):
            F.monte_carlo(m, bal, 10, 50, spend_scale=bad)
    F.monte_carlo(m, bal, 10, 50, spend_scale=0)
    F.monte_carlo(m, bal, 10, 50, spend_scale=3)


def test_spend_scale_default_identical_and_crn():
    m, bal = persona_model(seed=0)
    a = F.monte_carlo(m, bal, 30, 200, seed=3)
    b = F.monte_carlo(m, bal, 30, 200, seed=3, spend_scale=1.0, detail=True)
    assert np.array_equal(a.paths, b.paths) and a.disc_by_cat is None
    lo = F.monte_carlo(m, bal, 30, 200, seed=3, spend_scale=0.8)
    assert (lo.paths >= a.paths - 1e-9).all()


def test_disc_by_cat_matches_paths():
    m, bal = persona_model(seed=0)
    r = F.monte_carlo(m, bal, 30, 200, seed=3, spend_scale=0.7, cuts={m.cats[0]: 0.5}, detail=True)
    assert r.disc_by_cat.shape == (200, len(m.cats))
    fixed = F.recurring_flows(m, r.dates)
    assert np.allclose(r.paths[:, -1], bal + fixed.sum() - r.disc_by_cat.sum(axis=1))


# ---- labels / storms
def test_labels_at_thresholds():
    assert [W.label(p) for p in (0, 0.049, 0.05, 0.149, 0.15, 0.349, 0.35, 1)] == \
        ["clear", "clear", "cloudy", "cloudy", "rain", "rain", "storm", "storm"]


def test_storms_merge_and_ignore_isolated():
    ps = [0.0, 0.3, 0.4, 0.5, 0.1, 0.24, 0.0, 0.26, 0.0]
    s = W.storms(cal_with(ps))
    assert [(x["start"], x["end"], x["days"]) for x in s] == \
        [("2026-03-02", "2026-03-04", 3), ("2026-03-08", "2026-03-08", 1)]
    assert s[0]["peak_p"] == 0.5 and s[0]["peak_date"] == "2026-03-04" and s[0]["label"] == "storm"
    assert s[1]["label"] == "rain"
    assert W.storms(cal_with([0.1, 0.24, 0.2])) == []
    assert W.storms([]) == []


def test_storms_sorted_by_peak():
    s = W.storms(cal_with([0.3, 0, 0.9, 0, 0.5]))
    assert [x["peak_p"] for x in s] == [0.9, 0.5, 0.3]


# ---- buffer / calendar
def test_default_buffer_formula():
    rec = [{"merchant": "RENT", "amount": -3040.0, "day": 1}, {"merchant": "PAY", "amount": 9000.0, "day": 1}]
    m = toy(np.full(90, 100.0), recurring=rec)
    assert W.default_buffer(m) == pytest.approx(7 * (100 + 3040 / 30.4))
    assert W.default_buffer(toy(np.zeros((10, 0)))) == 0.0


def test_calendar_shape_and_crn_lifestyle():
    m, bal = persona_model(seed=0)
    c1 = W.risk_calendar(m, bal, 60, 500, buffer=bal * 0.8, lifestyle=1.0)
    c8 = W.risk_calendar(m, bal, 60, 500, buffer=bal * 0.8, lifestyle=0.8)
    assert len(c1) == 60 and c1[0]["date"] == str((m.last_date + pd.Timedelta(days=1)).date())
    for k in c1[0]:
        assert k in {"date", "p_below_buffer", "label", "p10", "p50", "p90"}
    assert all(a["p_below_buffer"] >= b["p_below_buffer"] for a, b in zip(c1, c8))
    assert all(0 <= c["p_below_buffer"] <= 1 and c["p10"] <= c["p50"] <= c["p90"] for c in c1)
    assert all(c["label"] == W.label(c["p_below_buffer"]) for c in c1)


def test_lifestyle_zero_is_fixed_flows_only():
    m, bal = persona_model(seed=0)
    c = W.risk_calendar(m, bal, 40, 100, buffer=0, lifestyle=0)
    fixed = bal + np.cumsum(F.recurring_flows(m, pd.date_range(m.last_date + pd.Timedelta(days=1), periods=40)))
    assert np.allclose([x["p10"] for x in c], fixed.round(2)) and np.allclose([x["p90"] for x in c], fixed.round(2))


def test_lifestyle_validation():
    m, bal = persona_model(seed=0)
    for bad in (-0.1, 2.1, float("nan")):
        with pytest.raises(ValueError):
            W.risk_calendar(m, bal, 10, 50, lifestyle=bad)


def test_degenerate_models():
    empty = toy(np.zeros((30, 0)))
    r = W.report(empty, 1000.0, 20, 50)
    assert r["buffer"] == 0.0 and r["autopsy"]["p_any_breach"] == 0.0 and r["umbrellas"] == []
    # buffer larger than balance: every path fails on day 1
    m = driver_model()
    a = W.autopsy(m, 100.0, 10, 100, buffer=10_000)
    assert a["p_any_breach"] == 1.0 and a["n_failing"] == 100 and a["first_breach"]["share"] == 1.0
    assert a["causes"] == []  # nothing to compare failing paths against
    assert W.risk_calendar(m, 100.0, 10, 100, buffer=10_000)[0]["p_below_buffer"] == 1.0
    # buffer 0 with ample balance never breaches
    assert W.autopsy(m, 1e9, 10, 50, buffer=0)["p_any_breach"] == 0.0
    # horizon shorter than the model window and horizon 1
    assert len(W.risk_calendar(m, 5000.0, 1, 50)) == 1


# ---- autopsy
def test_autopsy_ranks_driver_first():
    m = driver_model()
    a = W.autopsy(m, 3000.0, 30, 1000, buffer=300.0)
    assert 0 < a["n_failing"] < a["n_paths"] == 1000
    assert a["p_any_breach"] == pytest.approx(a["n_failing"] / 1000, abs=1e-3)
    top = a["causes"][0]
    assert top["category"] == "big" and top["share_of_gap"] > 0.9
    assert top["failing_mean"] > top["other_mean"]
    assert sum(c["share_of_gap"] for c in a["causes"]) <= 1.0 + 1e-9
    assert all(c["excess_per_path"] > 0 for c in a["causes"])
    assert a["first_breach"]["date"] is not None and 0 < a["first_breach"]["share"] <= 1


def test_autopsy_no_failure():
    m = driver_model()
    a = W.autopsy(m, 1e7, 30, 200, buffer=100.0)
    assert a["causes"] == [] and a["bills"] == [] and a["n_failing"] == 0 and a["p_any_breach"] == 0.0
    assert a["first_breach"] == {"date": None, "share": 0.0}


def test_autopsy_bills_before_median_breach():
    rec = [{"merchant": "RENT", "amount": -2000.0, "day": 10}]
    m = toy(np.full(90, 20.0), recurring=rec)
    start = m.last_date + pd.Timedelta(days=1)
    a = W.autopsy(m, 2100.0, 60, 50, buffer=500.0)
    assert a["p_any_breach"] == 1.0
    assert a["bills"] and all(b["merchant"] == "RENT" and b["amount"] == -2000.0 for b in a["bills"])
    fb = a["first_breach"]["date"]
    assert all(b["date"] <= fb for b in a["bills"])
    assert all(pd.Timestamp(b["date"]) >= start for b in a["bills"])


# ---- umbrellas
def test_umbrellas_ordered_and_monotone():
    m = driver_model()
    u = W.umbrellas(m, 3000.0, 30, 1000, buffer=300.0)
    assert 0 < len(u) <= 3
    assert [x["reduction_per_1000"] for x in u] == sorted((x["reduction_per_1000"] for x in u), reverse=True)
    for x in u:
        assert x["p_any_breach_after"] <= x["p_any_breach_before"] and x["risk_reduction"] > 0
        assert x["cut_pct"] in (10, 20, 30)
        assert x["monthly_sacrifice"] > 0
    assert u[0]["category"] == "big"
    big_mean = m.daily[:, 0].mean()
    x = next(y for y in u if y["category"] == "big")
    assert x["monthly_sacrifice"] == round(big_mean * 30 * x["cut_pct"] / 100, 2)


def test_umbrellas_empty_when_no_breach():
    assert W.umbrellas(driver_model(), 1e7, 30, 200, buffer=100.0) == []


# ---- report
def test_report_json_and_determinism_and_headline():
    m, bal = persona_model("priya")
    r = W.report(m, bal, 60, 400, seed=2)
    assert r == W.report(m, bal, 60, 400, seed=2)
    assert json.loads(json.dumps(r, allow_nan=False)) == r
    assert set(r) == {"buffer", "lifestyle", "horizon", "calendar", "storms", "autopsy", "umbrellas", "headline"}
    h = r["headline"]
    assert h["worst_p"] == max(c["p_below_buffer"] for c in r["calendar"])
    assert h["worst_day"] in {c["date"] for c in r["calendar"]}
    assert h["n_storm_days"] == sum(s["days"] for s in r["storms"])
    assert h["p_any_breach"] == r["autopsy"]["p_any_breach"]
    for v in (r["buffer"], r["lifestyle"], h["worst_p"]):
        assert type(v) in (float, int)


def test_report_with_cuts_lowers_risk():
    m, bal = persona_model("priya")
    base = W.report(m, bal, 60, 400, seed=2)
    cut = W.report(m, bal, 60, 400, seed=2, cuts={c: 0.3 for c in m.cats})
    assert cut["headline"]["p_any_breach"] <= base["headline"]["p_any_breach"]
    assert all(a["p_below_buffer"] <= b["p_below_buffer"] for a, b in zip(cut["calendar"], base["calendar"]))
