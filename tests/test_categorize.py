import json
import pandas as pd
from finpilot.engines.categorize import normalize_merchant, rule_category
from finpilot.agents.categorizer import categorize_df
from finpilot.llm import LLMUnavailable
from tests.conftest import FakeLLM


def test_normalize():
    assert normalize_merchant("UPI-SWIGGY-482913") == "SWIGGY"
    assert normalize_merchant("POS CULT.FIT GYM BANGALORE") == "CULT FIT GYM"
    assert normalize_merchant("AMAZON PAYMENT 123456") == "AMAZON"


def test_rules():
    assert rule_category("SWIGGY", -300) == "food_delivery"
    assert rule_category("SUNRISE APARTMENTS RENT", -18000) == "rent"
    assert rule_category("PVR CINEMAS", -800) is None
    assert rule_category("ACME CORP SALARY", 62000) == "income"


def test_llm_only_merchants_not_in_rules():
    for desc in ["BLUE TOKAI COFFEE", "NYKAA", "PVR CINEMAS", "DR SHAH CLINIC",
                 "REDBUS", "GLOBALTECH TRADING LTD"]:
        assert rule_category(normalize_merchant(desc), -500) is None, desc


def _df(descs, amts):
    return pd.DataFrame({"date": pd.to_datetime(["2026-01-01"] * len(descs)),
                         "description": descs, "amount": amts})


def test_llm_fallback_batches_unique_merchants():
    llm = FakeLLM(lambda s, u: {m: "entertainment" for m in json.loads(u)})
    out = categorize_df(_df(["UPI-PVR CINEMAS-1", "PVR CINEMAS", "SWIGGY"], [-800, -900, -300]), llm)
    assert list(out.category) == ["entertainment", "entertainment", "food_delivery"]
    assert list(out.category_source) == ["llm", "llm", "rule"]
    assert len(llm.calls) == 1  # one batch, one unique unknown merchant


def test_llm_failure_degrades():
    def boom(s, u):
        raise LLMUnavailable("down")
    out = categorize_df(_df(["PVR CINEMAS", "SWIGGY"], [-800, -300]), FakeLLM(boom))
    assert list(out.category) == ["uncategorized", "food_delivery"]


def test_invalid_llm_category_becomes_other():
    llm = FakeLLM(lambda s, u: {m: "spaceships" for m in json.loads(u)})
    assert categorize_df(_df(["PVR CINEMAS"], [-800]), llm).category.iloc[0] == "other"


def test_user_override_wins():
    out = categorize_df(_df(["SWIGGY"], [-300]), None, overrides={"SWIGGY": "dining"})
    assert out.category.iloc[0] == "dining" and out.category_source.iloc[0] == "user"


def test_non_dict_llm_response_degrades():
    for bad in ([1, 2], None, "x"):
        out = categorize_df(_df(["PVR CINEMAS", "SWIGGY"], [-800, -300]), FakeLLM(lambda s, u, b=bad: b))
        assert list(out.category) == ["uncategorized", "food_delivery"]


def test_partial_llm_response():
    llm = FakeLLM(lambda s, u: {"PVR CINEMAS": "entertainment"})
    out = categorize_df(_df(["PVR CINEMAS", "NYKAA"], [-800, -500]), llm)
    assert list(out.category) == ["entertainment", "uncategorized"]
    assert list(out.category_source) == ["llm", "none"]


def test_non_string_category_not_accepted():
    for bad in (["food"], 5, {"a": 1}):
        llm = FakeLLM(lambda s, u, b=bad: {m: b for m in json.loads(u)})
        out = categorize_df(_df(["PVR CINEMAS"], [-800]), llm)
        assert out.category.iloc[0] == "uncategorized"


def test_index_preserved():
    df = _df(["PVR CINEMAS", "SWIGGY"], [-800, -300])
    df.index = [10, 11]
    llm = FakeLLM(lambda s, u: {m: "entertainment" for m in json.loads(u)})
    out = categorize_df(df, llm)
    assert list(out.index) == [10, 11]
    assert out.loc[10, "category"] == "entertainment" and out.loc[11, "category"] == "food_delivery"
