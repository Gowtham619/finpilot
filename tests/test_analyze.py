import pytest
from eval.generate import generate, PERSONAS
from finpilot.engines.categorize import normalize_merchant
from finpilot.engines import analyze as A


def prep(**kw):
    p = generate(**kw)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories  # isolate analysis from categorization
    return p, df


def test_recurring_finds_bills_not_habits():
    _, df = prep(seed=0)
    ms = {r["merchant"] for r in A.detect_recurring(df)}
    assert {"SUNRISE APARTMENTS RENT", "BESCOM ELECTRICITY", "ACME CORP SALARY", "NETFLIX"} <= ms
    assert "SWIGGY" not in ms and "BIGBASKET" not in ms


def test_subscriptions():
    _, df = prep(seed=0)
    names = {s["merchant"] for s in A.detect_subscriptions(df)}
    assert names == {"NETFLIX", "HOTSTAR", "SPOTIFY", "CULT FIT GYM"}


def test_unused_overlap_and_no_usage():
    _, df = prep(seed=0)
    unused = A.unused_subscriptions(df, A.detect_subscriptions(df))
    assert {u["merchant"] for u in unused} == {"HOTSTAR", "CULT FIT GYM"}
    assert {u["reason"] for u in unused} == {"overlap", "no_usage_signal"}


def test_gym_with_usage_not_flagged():
    _, df = prep(seed=0, gym_used=True)
    unused = A.unused_subscriptions(df, A.detect_subscriptions(df))
    assert {u["merchant"] for u in unused} == {"HOTSTAR"}


def test_category_spike_found_in_planted_month():
    p, df = prep(seed=0)
    month, cat = p.truth.spike
    ids = {f["id"] for f in A.category_spikes(df)}
    assert f"spike:{cat}:{month}" in ids


def test_outlier_and_duplicate_found():
    p, df = prep(seed=0)
    rec = {r["merchant"] for r in A.detect_recurring(df)}
    out = {f["txn_id"] for f in A.txn_outliers(df, rec)}
    assert p.truth.anomaly_ids <= out
    dups = {f["txn_id"] for f in A.duplicate_charges(df)}
    assert dups == p.truth.duplicate_ids


def test_rent_never_flagged_as_outlier():
    _, df = prep(seed=0)
    rec = {r["merchant"] for r in A.detect_recurring(df)}
    assert not [f for f in A.txn_outliers(df, rec) if "RENT" in f["merchant"]]


def test_payday_trigger():
    p, df = prep(seed=0)
    rec = {r["merchant"] for r in A.detect_recurring(df)}
    ids = {f["id"] for f in A.behavior_triggers(df, rec)}
    assert f"trigger:{p.truth.payday_trigger_category}:payday" in ids


def test_analyze_all_has_unique_ids():
    _, df = prep(seed=0)
    ids = [f["id"] for f in A.analyze_all(df)]
    assert len(ids) == len(set(ids)) > 5


CASES = [dict(seed=0)] + list(PERSONAS.values())


@pytest.mark.parametrize("kw", CASES)
def test_large_charge_precision(kw):
    p, df = prep(**kw)
    rec = {r["merchant"] for r in A.detect_recurring(df)}
    out = {f["txn_id"] for f in A.txn_outliers(df, rec)}
    assert p.truth.anomaly_ids <= out
    assert len(out) <= 3


@pytest.mark.parametrize("kw", CASES)
def test_category_spike_precision(kw):
    p, df = prep(**kw)
    month, cat = p.truth.spike
    ids = [f["id"] for f in A.category_spikes(df)]
    assert f"spike:{cat}:{month}" in ids
    assert len(ids) <= 3
