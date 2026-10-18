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
