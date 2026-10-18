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
    assert fix is None or fix["cut_pct"] == 0 or fix["p_overdraft"] <= 0.10


def test_cheapest_fix_result_meets_target(world):
    m, bal = world
    flows = S.purchase_flows(bal * 0.9, m.last_date + pd.Timedelta(days=3))
    fix = S.cheapest_fix(m, bal, flows)
    assert fix is None or fix["p_overdraft"] <= 0.10


def test_cheapest_fix_empty_cats(world):
    m, bal = world
    from dataclasses import replace, is_dataclass
    flows = S.purchase_flows(bal * 5, m.last_date + pd.Timedelta(days=3))
    m2 = replace(m, cats=[]) if is_dataclass(m) else m
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
