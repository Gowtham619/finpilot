import pandas as pd
from eval.generate import generate, load_persona, PERSONAS


def test_shape_and_determinism():
    a, b = generate(seed=0), generate(seed=0)
    assert list(a.df.columns) == ["date", "description", "amount"]
    assert a.df.equals(b.df)
    assert len(a.truth.categories) == len(a.df)


def test_different_seeds_differ():
    assert not generate(seed=0).df.equals(generate(seed=1).df)


def test_plants():
    p = generate(seed=0)
    assert len(p.truth.anomaly_ids) == 1
    assert len(p.truth.duplicate_ids) == 1
    i = next(iter(p.truth.anomaly_ids))
    assert p.df.loc[i, "amount"] == -24000
    assert p.truth.spike == ("2026-07", "food_delivery")
    assert p.truth.unused_subs == {"HOTSTAR", "CULT FIT GYM"}


def test_gym_used_removes_gym_from_unused():
    assert generate(seed=0, gym_used=True).truth.unused_subs == {"HOTSTAR"}


def test_span_is_twelve_months():
    p = generate(seed=0)
    assert p.df.date.dt.to_period("M").nunique() == 12


def test_personas_load():
    for name in PERSONAS:
        assert len(load_persona(name).df) > 300


def _all_personas():
    yield "seed0", generate(seed=0)
    for name in PERSONAS:
        yield name, load_persona(name)


def test_duplicate_row_matches_original():
    p = generate(seed=0)
    i = next(iter(p.truth.duplicate_ids))
    row = p.df.loc[i]
    assert "AMAZON" in row.description
    assert row.amount == -2199.0
    orig = p.df.loc[i - 1]
    assert orig.amount == -2199.0
    assert row.date - orig.date == pd.Timedelta(days=1)


def test_category_alignment():
    p = generate(seed=0)
    cats = p.truth.categories
    for key, cat in [("ZOMATO", "food_delivery"), ("SWIGGY", "food_delivery"),
                     ("NETFLIX", "subscriptions")]:
        m = p.df.description.str.contains(key)
        assert m.any()
        assert (cats[m] == cat).all()


def test_spike_month_is_elevated():
    for name, p in _all_personas():
        month, cat = p.truth.spike
        fd = -p.df.amount[p.truth.categories == cat]
        by_month = fd.groupby(p.df.date.dt.to_period("M").astype(str)).sum()
        others = by_month.drop(month)
        assert by_month[month] >= 1.5 * others.median(), (name, by_month[month], others.median())


def test_plants_for_each_persona():
    for name, p in _all_personas():
        assert len(p.truth.anomaly_ids) == 1, name
        assert len(p.truth.duplicate_ids) == 1, name
