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
