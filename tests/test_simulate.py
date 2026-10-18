import pandas as pd
import pytest
from eval.generate import generate
from finpilot.engines.categorize import normalize_merchant
from finpilot.engines import forecast as F
from finpilot.engines import simulate as S


@pytest.fixture(scope="module")
def world():
    p = generate(seed=0)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    m = F.fit(df)
    return m, F.current_balance(df, p.opening_balance)


def test_zero_purchase_has_zero_delta(world):
    m, bal = world
    r = S.simulate_purchase(m, bal, 0, m.last_date + pd.Timedelta(days=3))
    assert r["delta_p_overdraft"] == 0 and r["delta_end_p50"] == 0


def test_small_purchase_delta_is_exact_outflow(world):
    m, bal = world
    r = S.simulate_purchase(m, bal, 1000, m.last_date + pd.Timedelta(days=3))
    assert r["delta_end_p50"] == -1000
    assert r["ignored_flows"] == 0


def test_out_of_window_purchase_raises(world):
    m, bal = world
    with pytest.raises(ValueError, match="outside the forecast window"):
        S.simulate_purchase(m, bal, 1000, m.last_date + pd.Timedelta(days=200))
    with pytest.raises(ValueError, match="outside the forecast window"):
        S.simulate_purchase(m, bal, 1000, m.last_date)


def test_installments_beyond_horizon_counted(world):
    m, bal = world
    r = S.simulate_purchase(m, bal, 3000, m.last_date + pd.Timedelta(days=50),
                            horizon=60, financed_months=3)
    assert r["ignored_flows"] == 2


def test_best_date_excludes_out_of_window(world):
    m, bal = world
    cands = [m.last_date + pd.Timedelta(days=d) for d in (0, 5, 500)]
    out = S.best_date(m, bal, 1000, cands)
    assert [o["date"] for o in out] == [str((m.last_date + pd.Timedelta(days=5)).date())]
    assert S.best_date(m, bal, 1000, [m.last_date + pd.Timedelta(days=500)]) == []


def test_duel_reports_ignored_flows(world):
    m, bal = world
    far = S.purchase_flows(1000, m.last_date + pd.Timedelta(days=500))
    r = S.duel(m, bal, {"name": "A", "flows": far, "cuts": {}}, {"name": "B", "flows": [], "cuts": {}})
    assert r["ignored_flows"] == 1


def test_big_purchase_hurts(world):
    m, bal = world
    r = S.simulate_purchase(m, bal, bal * 1.5, m.last_date + pd.Timedelta(days=3))
    assert r["with_purchase"]["p_overdraft"] > r["baseline"]["p_overdraft"]
    assert r["delta_end_p50"] < -bal


def test_financing_splits_the_outflow():
    flows = S.purchase_flows(12000, "2026-10-10", financed_months=3)
    assert len(flows) == 3 and sum(a for _, a in flows) == -12000
    assert flows[1][0] == pd.Timestamp("2026-11-10")


def test_financing_across_month_end():
    flows = S.purchase_flows(3000, "2026-01-31", financed_months=2)
    assert flows[1][0] == pd.Timestamp("2026-02-28")
    assert sum(a for _, a in flows) == -3000


def test_best_date_sorted(world):
    m, bal = world
    cands = [m.last_date + pd.Timedelta(days=d) for d in (2, 9, 16, 23)]
    out = S.best_date(m, bal, bal * 0.9, cands)
    assert len(out) == 4
    ps = [o["p_overdraft"] for o in out]
    assert ps == sorted(ps)


def test_cheapest_fix_no_cut_needed(world):
    m, bal = world
    fix = S.cheapest_fix(m, bal, [])
    assert fix["category"] is None and fix["cut_pct"] == 0
    assert fix["p_overdraft"] <= 0.10 and fix["est_monthly_saving"] == 0.0


def test_cheapest_fix_result_meets_target(world):
    m, bal = world
    flows = S.purchase_flows(bal * 1.1, m.last_date + pd.Timedelta(days=3))
    base = F.monte_carlo(m, bal, 60, 1000, 0, extra_flows=flows)
    assert base.p_overdraft > 0.10
    fix = S.cheapest_fix(m, bal, flows)
    assert fix is not None
    assert fix["cut_pct"] > 0 and fix["p_overdraft"] <= 0.10
    assert fix["est_monthly_saving"] > 0


def test_cheapest_fix_empty_cats(world):
    m, bal = world
    from dataclasses import replace
    m2 = replace(m, cats=[])
    assert m2.cats == []
    flows = S.purchase_flows(bal * 5, m.last_date + pd.Timedelta(days=3))
    assert S.cheapest_fix(m2, bal, flows) is None


def test_duel_picks_safer_option(world):
    m, bal = world
    a = {"name": "Buy now", "flows": S.purchase_flows(bal * 3, m.last_date + pd.Timedelta(days=3)), "cuts": {}}
    b = {"name": "Skip it", "flows": [], "cuts": {}}
    r = S.duel(m, bal, a, b)
    assert r["winner"] == "b"
    assert "Skip it" in r["reason"]
    assert len(r["a"]["bands"]["p50"]) == 90


def test_duel_tie(world):
    m, bal = world
    a = {"name": "A", "flows": [], "cuts": {}}
    b = {"name": "B", "flows": [], "cuts": {}}
    r = S.duel(m, bal, a, b)
    assert r["winner"] == "tie"


# ---- final review fixes: C3 tie-breaks, C5 cut clamp, I1 financed best_date ----
import numpy as np  # noqa: E402
from finpilot.engines.forecast import Model  # noqa: E402


def _flat_model(days=28):
    """Deterministic model: one category spending 100 every day, no recurring flows."""
    last = pd.Timestamp("2026-09-30")
    idx = pd.date_range(last - pd.Timedelta(days=days - 1), last)
    return Model([], ["food"], np.full((days, 1), 100.0), np.asarray(idx.weekday), last)


def test_duel_same_end_decided_by_lowest_balance():
    m = _flat_model()
    d1 = m.last_date + pd.Timedelta(days=1)
    a = {"name": "Buy now", "flows": S.purchase_flows(5000, d1), "cuts": {}}
    b = {"name": "Wait", "flows": S.purchase_flows(5000, m.last_date + pd.Timedelta(days=30)), "cuts": {}}
    r = S.duel(m, 20000, a, b, horizon=30)
    assert r["a"]["summary"]["end_p50"] == r["b"]["summary"]["end_p50"]
    assert r["winner"] == "b" and "lowest balance" in r["reason"]
    assert "₹12,000" in r["reason"] and "₹12,000" in r["reason"]


def test_duel_same_end_and_floor_decided_by_average():
    m = _flat_model()
    # both purchases happen before the balance bottoms out at the horizon end, so floor and end
    # are identical; the earlier purchase leaves a lower balance on average
    a = {"name": "Early", "flows": S.purchase_flows(1000, m.last_date + pd.Timedelta(days=2)), "cuts": {}}
    b = {"name": "Late", "flows": S.purchase_flows(1000, m.last_date + pd.Timedelta(days=20)), "cuts": {}}
    r = S.duel(m, 20000, a, b, horizon=30)
    sa, sb = r["a"]["summary"], r["b"]["summary"]
    assert sa["end_p50"] == sb["end_p50"] and sa["floor_p50"] == sb["floor_p50"]
    assert r["winner"] == "b" and "average balance" in r["reason"]
    assert f"₹{sb['avg_p50']:,.0f}" in r["reason"] and f"₹{sa['avg_p50']:,.0f}" in r["reason"]


def test_duel_true_tie_prints_numbers():
    m = _flat_model()
    r = S.duel(m, 20000, {"name": "A", "flows": [], "cuts": {}}, {"name": "B", "flows": [], "cuts": {}}, horizon=30)
    assert r["winner"] == "tie"
    assert "0%" in r["reason"] and "₹17,000" in r["reason"]


def test_duel_negative_balances_are_worded():
    m = _flat_model()
    r = S.duel(m, 1000, {"name": "A", "flows": [], "cuts": {}},
               {"name": "B", "flows": S.purchase_flows(500, m.last_date + pd.Timedelta(days=1)), "cuts": {}},
               horizon=30)
    assert "overdrawn by ₹" in r["reason"] and "₹-" not in r["reason"]


@pytest.mark.parametrize("cut", [5, -3, 1.0])
def test_monte_carlo_clamps_cuts(cut):
    m = _flat_model()
    r = F.monte_carlo(m, 1000, 10, 50, 0, cuts={"food": cut})
    expected = 1000 - 100 * 10 * min(max(1 - cut, 0), 1)
    assert r.p50[-1] == pytest.approx(expected)


def test_best_date_simulates_installments():
    m = _flat_model()
    d1 = m.last_date + pd.Timedelta(days=1)
    lump = S.best_date(m, 10000, 9000, [d1], horizon=35)
    fin = S.best_date(m, 10000, 9000, [d1], horizon=35, financed_months=3)
    assert lump[0]["p_overdraft"] == 1.0  # 9000 at once: overdrawn by day 11
    assert fin[0]["p_overdraft"] == 0.0 and fin[0]["floor_p50"] == 500.0  # two 3000 installments in window
