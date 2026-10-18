import numpy as np
import pandas as pd
from eval.generate import generate
from finpilot.engines.categorize import normalize_merchant
from finpilot.engines import forecast as F


def prep(**kw):
    p = generate(**kw)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    return p, df


def test_fit_separates_recurring_from_discretionary():
    _, df = prep(seed=0)
    m = F.fit(df)
    assert any("RENT" in r["merchant"] for r in m.recurring)
    assert "rent" not in m.cats and "food_delivery" in m.cats
    assert m.daily.shape[0] == 90 and m.daily.shape[1] == len(m.cats)


def test_deterministic_with_seed():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    a = F.monte_carlo(m, bal, 30, 200, seed=7)
    b = F.monte_carlo(m, bal, 30, 200, seed=7)
    assert np.array_equal(a.paths, b.paths)


def test_bands_ordered_and_start_near_balance():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    r = F.monte_carlo(m, bal, 45, 500)
    assert (r.p10 <= r.p50 + 1e-9).all() and (r.p50 <= r.p90 + 1e-9).all()
    # day-1 recurring flows (e.g. salary) land on the first forecast day
    assert abs(r.p50[0] - (bal + F.recurring_flows(m, r.dates)[0])) < 5000
    assert r.paths.shape == (500, 45)


def test_extra_outflow_lowers_balance_and_raises_risk():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    base = F.monte_carlo(m, bal, 45, 500, seed=1)
    d = m.last_date + pd.Timedelta(days=5)
    big = F.monte_carlo(m, bal, 45, 500, seed=1, extra_flows=[(d, -bal * 2)])
    assert big.end_p50 < base.end_p50 - bal
    assert big.p_overdraft > base.p_overdraft


def test_cuts_raise_end_balance():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    base = F.monte_carlo(m, bal, 45, 500, seed=1)
    cut = F.monte_carlo(m, bal, 45, 500, seed=1, cuts={c: 0.5 for c in m.cats})
    assert cut.end_p50 > base.end_p50


def test_recurring_income_and_rent_hit_expected_days():
    p, df = prep(seed=0)
    m = F.fit(df)
    flows = F.recurring_flows(m, pd.date_range(m.last_date + pd.Timedelta(days=1), periods=60))
    assert (flows > 50000).any()   # salary lands inside 60 days
    assert (flows < -15000).any()  # rent lands inside 60 days


def test_goal_status():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    deadline = m.last_date + pd.Timedelta(days=60)
    easy = F.goal_status(m, bal, bal - 1000, deadline)
    hard = F.goal_status(m, bal, bal + 10_000_000, deadline)
    assert easy["p_met"] > 0.9 and hard["p_met"] == 0.0 and hard["eta"] is None


def test_no_discretionary_categories():
    m = F.Model([], [], np.zeros((90, 0)), np.arange(90) % 7, pd.Timestamp("2026-01-31"))
    r = F.monte_carlo(m, 100.0, 10, 20)
    assert (r.paths == 100.0).all()


def test_extra_flow_outside_horizon_ignored():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    base = F.monte_carlo(m, bal, 20, 100, seed=2)
    far = m.last_date + pd.Timedelta(days=400)
    past = m.last_date - pd.Timedelta(days=3)
    r = F.monte_carlo(m, bal, 20, 100, seed=2, extra_flows=[(far, -1e6), (past, -1e6)])
    assert np.array_equal(r.paths, base.paths)


def test_goal_status_past_deadline():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    g = F.goal_status(m, bal, bal - 1000, m.last_date - pd.Timedelta(days=10))
    assert g["horizon"] == 1 and 0.0 <= g["p_met"] <= 1.0


def test_goal_status_capped_for_far_deadline():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    g = F.goal_status(m, bal, bal, m.last_date + pd.Timedelta(days=730))
    assert g["capped"] is True and g["horizon"] == 365
    g2 = F.goal_status(m, bal, bal, m.last_date + pd.Timedelta(days=60))
    assert g2["capped"] is False and g2["horizon"] == 60


def test_common_random_numbers():
    p, df = prep(seed=0)
    m = F.fit(df)
    bal = F.current_balance(df, p.opening_balance)
    base = F.monte_carlo(m, bal, 30, 300, seed=3)
    assert np.array_equal(base.paths, F.monte_carlo(m, bal, 30, 300, seed=3).paths)
    d = m.last_date + pd.Timedelta(days=4)
    flow = F.monte_carlo(m, bal, 30, 300, seed=3, extra_flows=[(d, -5000.0)])
    diff = flow.paths - base.paths
    expected = np.where(np.arange(30) >= 3, -5000.0, 0.0)
    assert np.allclose(diff, expected[None, :])
    # cuts change only discretionary scale: same draws, so cut paths dominate pathwise
    cut = F.monte_carlo(m, bal, 30, 300, seed=3, cuts={c: 0.5 for c in m.cats})
    assert (cut.paths >= base.paths - 1e-9).all()


def test_short_history_fit():
    p, df = prep(seed=0)
    short = df[df.date >= df.date.max().normalize() - pd.Timedelta(days=19)]
    m = F.fit(short)
    nd = (short.date.max().normalize() - short.date.min().normalize()).days + 1
    assert m.daily.shape[0] == nd and nd >= 15
    true_mean = -short[(short.amount < 0) & (~short.merchant.isin({r["merchant"] for r in m.recurring}))].amount.sum() / nd
    assert abs(m.daily.sum(axis=1).mean() - true_mean) < 1e-6
    tiny = df[df.date >= df.date.max().normalize() - pd.Timedelta(days=2)]
    mt = F.fit(tiny)
    r = F.monte_carlo(mt, 1000.0, 15, 50)
    assert r.paths.shape == (50, 15) and np.isfinite(r.paths).all()


def test_empty_df_raises():
    import pytest
    _, df = prep(seed=0)
    with pytest.raises(ValueError):
        F.fit(df.iloc[0:0])
