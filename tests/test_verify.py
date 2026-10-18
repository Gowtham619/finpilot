import pytest
from eval.generate import generate
from finpilot.engines.categorize import normalize_merchant
from finpilot.engines import verify as V


@pytest.fixture(scope="module")
def world():
    p = generate(seed=0)
    df = p.df.copy()
    df["merchant"] = df.description.map(normalize_merchant)
    df["category"] = p.truth.categories
    return p, df


def test_recompute_monthly_and_annual(world):
    _, df = world
    assert V.recompute({"kind": "monthly_cost", "merchant": "HOTSTAR"}, df) == pytest.approx(299, abs=1)
    assert V.recompute({"kind": "annual_cost", "merchant": "HOTSTAR"}, df) == pytest.approx(3588, abs=12)
    assert V.recompute({"kind": "monthly_cost", "merchant": "GHOST"}, df) is None


def test_recompute_month_total_and_txn(world):
    p, df = world
    month, cat = p.truth.spike
    assert V.recompute({"kind": "month_total", "category": cat, "month": month}, df) > 5000
    i = next(iter(p.truth.anomaly_ids))
    assert V.recompute({"kind": "txn_amount", "txn_id": i}, df) == 24000


def test_check_claim_tolerance(world):
    _, df = world
    good = {"kind": "monthly_cost", "merchant": "HOTSTAR", "value": 300}
    bad = {"kind": "monthly_cost", "merchant": "HOTSTAR", "value": 999}
    assert V.check_claim(good, df)["ok"] and not V.check_claim(bad, df)["ok"]
    assert V.check_claim(bad, df)["actual"] == pytest.approx(299, abs=1)


def test_ungrounded_numbers():
    claims = [{"kind": "monthly_cost", "merchant": "X", "value": 299}]
    assert V.ungrounded_numbers("Cancel X (₹299/month) in 2026.", claims, {}) == []
    assert V.ungrounded_numbers("Cancel X and save ₹1,500.", claims, {}) == [1500.0]
    assert V.ungrounded_numbers("Saves 41% of 3588", claims, {"annual_cost": 3588, "pct": 41}) == []
    assert V.ungrounded_numbers("Charged on 2026-06-22 in 2026-06.", claims, {}) == []
    assert V.ungrounded_numbers("There is a 37% chance.", claims, {"p_overdraft": 0.37}) == []


def test_unknown_claim_kind_is_unverifiable(world):
    _, df = world
    c = {"kind": "mystery", "value": 5}
    assert V.recompute(c, df) is None
    assert V.check_claim(c, df) == {"ok": False, "actual": None}


@pytest.mark.parametrize("claim", [
    {}, "monthly_cost", None, 5, {"kind": "monthly_cost", "value": 100},
    {"kind": "monthly_cost", "merchant": "HOTSTAR", "value": "₹5"},
    {"kind": "monthly_cost", "merchant": "HOTSTAR"},
    {"kind": "monthly_cost", "merchant": "HOTSTAR", "value": float("nan")},
    {"kind": "monthly_cost", "merchant": "HOTSTAR", "value": True},
    {"kind": "txn_amount", "txn_id": None, "value": 1},
    {"kind": "txn_amount", "txn_id": "abc", "value": 1},
    {"kind": "txn_amount", "txn_id": [1, 2], "value": 1},
    {"kind": "month_total", "category": "x", "value": 1},
])
def test_malformed_claims_are_unverifiable_not_crashes(world, claim):
    _, df = world
    assert V.check_claim(claim, df)["ok"] is False
    if not isinstance(claim, dict) or claim.get("kind") != "monthly_cost" or "merchant" not in claim:
        assert V.check_claim(claim, df)["actual"] is None
    V.recompute(claim, df)


def test_ungrounded_suffixes_and_currency_escapes():
    claims = [{"kind": "monthly_cost", "merchant": "X", "value": 299}]
    flag = lambda t, ev=None: V.ungrounded_numbers(t, claims, ev or {})
    assert flag("Save ₹1.5k a month") == [1500.0]
    assert flag("Save 5 lakh") == [500000.0]
    assert flag("Save 2 crore") == [20000000.0]
    assert flag("Save ₹2,000") == [2000.0]
    assert flag("Save ₹2000") == [2000.0]
    assert flag("Save ₹60") == [60.0]
    assert flag("Save ₹1000-2500") == [1000.0, 2500.0]
    assert flag("Save 1.5k", {"x": 1500}) == []
    assert flag("in 2026") == []
    assert flag("over 60 days") == []
    assert flag("Charged 2026-06-22 and 2026-06.") == []
