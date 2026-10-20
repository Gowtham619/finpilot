import hashlib
import json

import pandas as pd

from eval import run_eval
from eval.generate_messy import OOD_PERSONAS, generate_messy, load_messy
from eval.generate import Persona, Truth
from finpilot.agents.categorizer import categorize_df
from finpilot.engines.categorize import normalize_merchant


def _all():
    for n in OOD_PERSONAS:
        yield n, load_messy(n)


def test_shape_and_determinism():
    a, b = generate_messy(seed=5), generate_messy(seed=5)
    assert isinstance(a, Persona) and isinstance(a.truth, Truth)
    assert list(a.df.columns) == ["date", "description", "amount"]
    assert a.df.equals(b.df) and a.truth.categories.equals(b.truth.categories)
    assert not generate_messy(seed=5).df.equals(generate_messy(seed=6).df)
    assert len(OOD_PERSONAS) == 3
    assert a.df.date.dt.to_period("M").nunique() == 12


def test_truth_alignment():
    for n, p in _all():
        assert len(p.truth.categories) == len(p.df)
        assert (p.truth.categories.index == p.df.index).all()
        cats, d = p.truth.categories, p.df
        assert (cats[d.description.str.contains("NETFLIX", case=False)] == "subscriptions").all()
        assert (cats[d.description.str.contains("SALARY")] == "income").all()
        assert (cats[d.description.str.contains("ATM WDL")] == "other").all()
        assert d.date.is_monotonic_increasing


def test_planted_items():
    for n, p in _all():
        t, d = p.truth, p.df
        assert len(t.anomaly_ids) == 2 and len(t.duplicate_ids) == 2, n
        for i in t.anomaly_ids:
            assert d.loc[i, "amount"] <= -30000
        for i in t.duplicate_ids:
            row = d.loc[i]
            m = normalize_merchant(row.description)
            twins = d[(d.index < i) & (d.amount == row.amount) & (d.description.map(normalize_merchant) == m)]
            assert len(twins) == 1 and row.amount < 0
            assert pd.Timedelta(0) <= row.date - twins.iloc[0].date <= pd.Timedelta(days=1)


def test_spike_is_in_a_different_category_and_elevated():
    for n, p in _all():
        month, cat = p.truth.spike
        assert cat == "travel"
        s = -p.df.amount[(p.truth.categories == cat) & (p.df.amount < 0)]
        by_month = s.groupby(p.df.date.dt.to_period("M").astype(str)).sum().reindex(
            sorted(p.df.date.dt.to_period("M").astype(str).unique()), fill_value=0.0)
        assert by_month[month] >= 1.5 * by_month.drop(month).median(), n


def test_refunds_are_credits_but_not_income():
    for n, p in _all():
        refunds = p.df.description.str.startswith("REFUND CR")
        assert refunds.sum() >= 2
        assert (p.df.amount[refunds] > 0).all()
        assert (p.truth.categories[refunds] == "shopping").all()
        credits = p.df.amount > 0
        assert (p.truth.categories[credits] == "income").sum() < credits.sum()


def test_irregular_salary_and_bonus():
    for n, p in _all():
        sal = p.df[p.df.description.str.contains("SALARY")]
        assert sal.date.dt.day.nunique() >= 4
        inc = p.df[p.truth.categories == "income"]
        bonus = inc[inc.description.str.contains("BONUS")]
        assert len(bonus) == 1


def test_unused_subs_and_subscription_truth():
    ps = dict(_all())
    for n, p in ps.items():
        assert p.subscription_merchants >= p.truth.unused_subs
        assert len(p.subscription_merchants) == 5  # annual subscription excluded
        assert any(m.startswith("PRIME VIDEO") for m in p.truth.unused_subs)
        assert p.truth.payday_trigger_category is None
        assert p.df.description.str.contains("AMAZON PRIME ANNUAL").sum() == 1
    assert len(ps["kavya"].truth.unused_subs) == 2 and len(ps["rohan"].truth.unused_subs) == 1
    assert {normalize_merchant(d) for d in ps["kavya"].df.description[ps["kavya"].df.description.str.upper().str.contains("ANYTIME")]} <= ps["kavya"].subscription_merchants


def test_rules_only_accuracy_is_stressed_but_not_nonsense():
    for n, p in _all():
        pred = categorize_df(p.df, None)["category"]
        acc = float((pred == p.truth.categories).mean())
        assert 0.3 < acc < 1.0, (n, acc)
        # look-alike merchants fool the rules
        d = p.df.description
        for key, wrong in [("RENT THE RUNWAY", "rent"), ("METRO CASH", "transport")]:
            m = d.str.upper().str.contains(key) & (p.df.amount < 0)
            if m.any():
                assert (pred[m] == wrong).all()


def test_run_eval_has_ood(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    res = run_eval.main(["--no-llm"])
    o = res["ood"]
    assert set(o["personas"]) == set(OOD_PERSONAS)
    assert o["categorization"]["rules_only"]["accuracy"]["mean"] < 1.0
    assert set(o["detection"]) >= {"large_charge", "duplicate", "category_spike", "subscriptions",
                                   "unused_subscriptions", "payday_trigger"}
    assert {"model", "flat_balance", "seasonal_naive", "trailing_mean"} <= set(o["forecast"])
    assert o["planted"]["duplicates"] == 2 and o["planted"]["large_charges"] == 2
    assert any("same author" in m for m in res["methodology"])
    md = (tmp_path / "results.md").read_text()
    assert "Out-of-distribution check (second generator, engines unchanged)" in md
    json.loads((tmp_path / "results.json").read_text())


def test_first_generator_results_unchanged(tmp_path, monkeypatch):
    """Sections for the original generator must not move when the OOD block is added."""
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    res = run_eval.main(["--no-llm"])
    keep = {k: res[k] for k in ("categorization", "detection", "forecast", "forecast_meta", "calibration", "backtest")}
    h = hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()
    assert h == ORIGINAL_SECTIONS_SHA256


ORIGINAL_SECTIONS_SHA256 = "0f5e550668557b1076992add540567471b56131b9fd0f1124a765724fd2c2161"
