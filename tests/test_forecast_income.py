import numpy as np
import pandas as pd

from finpilot.agents import forecaster
from finpilot.engines import forecast as F


def _freelancer(days=150, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.date_range("2026-01-01", periods=days)):
        rows.append((d, "BIGBASKET GROCERIES", -float(rng.integers(300, 900)), "BIGBASKET GROCERIES", "groceries"))
        if rng.random() < 0.12:  # invoices land on random days from different clients
            rows.append((d, f"CLIENT {i} INVOICE", float(rng.integers(15000, 40000)), f"CLIENT{i} INVOICE", "income"))
    return pd.DataFrame(rows, columns=["date", "description", "amount", "merchant", "category"])


def test_irregular_income_is_part_of_the_forecast():
    df = _freelancer()
    m = F.fit(df)
    assert m.inflow is not None and m.inflow.sum() > 0
    bal = 50000.0
    r = F.monte_carlo(m, bal, 60)
    # monthly income ~ 0.12*30*27500 = 99k vs spend ~ 18k: the balance must rise, not drain like a no-income model would
    assert r.end_p50 > bal
    no_income = F.Model(m.recurring, m.cats, m.daily, m.weekdays, m.last_date)
    assert F.monte_carlo(no_income, bal, 60).end_p50 < bal


def test_no_inflow_means_unchanged_paths():
    df = _freelancer()
    df = df[df["amount"] < 0]
    assert F.fit(df).inflow is None


def test_typical_range_ignores_one_off_but_risk_keeps_it():
    df = _freelancer()
    big = pd.DataFrame([(pd.Timestamp("2026-05-20"), "CROMA ELECTRONICS", -90000.0, "CROMA ELECTRONICS", "shopping")], columns=df.columns)
    d = pd.concat([df, big], ignore_index=True).sort_values("date").reset_index(drop=True)
    out = forecaster.run(d, 20000.0, (), None)
    full = F.monte_carlo(F.fit(d), out["balance"], 60)
    assert out["summary"]["p_overdraft"] == round(full.p_overdraft, 3)  # risk still sees the real charge
    assert out["summary"]["end_p50"] > full.end_p50                      # typical median is not dragged down by it
