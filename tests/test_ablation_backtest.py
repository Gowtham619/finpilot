import json
import pytest
from eval import ablation, backtest
from eval.generate import generate
from tests.conftest import FakeLLM


def test_backtest_finds_real_savings():
    p = generate(seed=0)
    r = backtest.run(p, days=90)
    assert r["cancel_savings"] > 0 and r["excess_savings"] >= 0
    assert 0 <= r["cancel_pct_of_spend"] < 1 and 0 <= r["excess_pct_of_spend"] < 1
    assert r["cancel_precision"] == 1.0  # every cancelled sub is a planted unused one
    assert r["n_correct"] == r["n_cancelled"] >= 1
    assert 0 < r["cancel_recall"] <= 1 and r["n_unused_truth"] >= r["n_correct"]
    assert r["followed"]


def test_backtest_never_peeks_at_future():
    import dataclasses
    from eval.run_eval import _with_truth
    p = generate(seed=0)
    r = backtest.run(p, days=90)
    assert r["cutoff"] < str(p.df["date"].max().date())
    df = _with_truth(p)
    cutoff, train, _ = backtest.split_at_cutoff(df, 90)
    assert train["date"].max() <= cutoff
    base = backtest.candidates_at(df, cutoff)
    mutated = p.df.copy()
    future = mutated["date"] > cutoff
    mutated.loc[future, "amount"] = mutated.loc[future, "amount"] * 7 - 123
    p2 = dataclasses.replace(p, df=mutated)
    again = backtest.candidates_at(_with_truth(p2), cutoff)
    assert base == again


def test_backtest_no_double_count_and_proration():
    p = generate(seed=0)
    r = backtest.run(p, days=90)
    kinds = {f["kind"] for f in r["followed"]}
    assert kinds <= {"cancel", "category_excess"}
    assert abs(sum(f["saved"] for f in r["followed"] if f["kind"] == "cancel") - r["cancel_savings"]) < 0.01
    assert abs(sum(f["saved"] for f in r["followed"] if f["kind"] == "category_excess") - r["excess_savings"]) < 0.01


def test_llm_only_advice_counts_hallucinations():
    p = generate(seed=0)

    def handler(system, user):
        return {"recommendations": [
            {"text": "Cancel HOTSTAR to save ₹999 a month.", "claims": [{"kind": "monthly_cost", "merchant": "HOTSTAR", "value": 999}]},
            {"text": "Cancel NETFLIX to save ₹649 a month.", "claims": [{"kind": "monthly_cost", "merchant": "NETFLIX", "value": 649}]}]}

    r = ablation.llm_only_advice(p, FakeLLM(handler))
    assert r["recs"] == 2 and r["with_issues"] == 1 and r["hallucination_rate"] == 0.5


def test_llm_only_categorization_uses_every_merchant():
    p = generate(seed=0)
    llm = FakeLLM(lambda s, u: {m: "other" for m in json.loads(u)})
    r = ablation.llm_only_categorization(p, llm)
    assert 0 <= r["accuracy"] < 0.5  # everything labelled "other" is mostly wrong


def test_llm_only_forecast_parses_number():
    p = generate(seed=0)
    r = ablation.llm_only_forecast(p, FakeLLM(lambda s, u: {"next_month_net_flow": 1000}))
    assert r["net_flow_ape"] > 0 and r["n_used"] == 3


def test_baselines_survive_llm_unavailable():
    from finpilot.llm import LLMUnavailable
    p = generate(seed=0)

    def boom(s, u):
        raise LLMUnavailable("no key")

    llm = FakeLLM(boom)
    assert ablation.llm_only_categorization(p, llm) == {"accuracy": None, "macro_f1": None}
    assert ablation.llm_only_advice(p, llm)["hallucination_rate"] is None
    assert ablation.llm_only_forecast(p, llm) == {"net_flow_ape": None, "n_used": 0}
    r = ablation.run({"a": p}, llm)
    assert r["categorization"]["llm_only"]["accuracy"] is None
    assert r["categorization"]["llm_only"] == {"accuracy": None, "macro_f1": None, "n_used": 0}
    assert r["categorization"]["hybrid"]["n_used"] == 1
    assert r["forecast"]["llm_only"]["net_flow_ape"] is None


def test_real_data_path(tmp_path):
    from eval import real_data
    rows = ["when,desc,amt,lab"]
    for m in range(1, 7):
        rows.append(f"2025-{m:02d}-01,SALARY ACME,-50000,income")
        rows.append(f"2025-{m:02d}-03,NETFLIX.COM,649,fun")
        rows.append(f"2025-{m:02d}-09,BIGBASKET ORDER,2500,food")
    csv = tmp_path / "t.csv"
    csv.write_text("\n".join(rows))
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"date": "when", "description": "desc", "amount": "amt", "label": "lab",
                              "label_map": {"income": "income", "fun": "entertainment", "food": "groceries"},
                              "negate_amount": True, "source": "unit-test"}))
    out = real_data.run(str(csv), str(mp), FakeLLM(lambda s, u: {}))
    assert out["rows"] == 18 and out["labelled_rows"] == 18
    assert 0 <= out["accuracy_vs_dataset_labels"] <= 1
    assert "findings_by_type" in out


def test_real_data_label_alignment_unsorted(tmp_path):
    from eval import real_data
    # rows deliberately in DESCENDING date order so the internal sort reverses them
    rows = ["when,desc,amt,lab",
            "2025-04-09,ZOMATO ORDER,300,x",
            "2025-03-09,BIGBASKET ORDER,2500,food",
            "2025-02-03,NETFLIX.COM,649,fun",
            "2025-01-01,SALARY ACME,-50000,income"]
    csv = tmp_path / "t.csv"
    csv.write_text("\n".join(rows))
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"date": "when", "description": "desc", "amount": "amt", "label": "lab", "negate_amount": True,
                              "label_map": {"income": "income", "fun": "subscriptions", "food": "groceries", "x": "dining"}}))
    out = real_data.run(str(csv), str(mp), FakeLLM(lambda s, u: {}))
    # rules give income, subscriptions, groceries, food_delivery; the zomato label is mapped to dining -> 3 of 4 right
    assert out["labelled_rows"] == 4 and out["accuracy_vs_dataset_labels"] == 0.75


def test_real_data_label_without_label_map(tmp_path):
    from eval import real_data
    csv = tmp_path / "t.csv"
    csv.write_text("when,desc,amt,lab\n2025-01-01,NETFLIX.COM,10,fun\n")
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"date": "when", "description": "desc", "amount": "amt", "label": "lab"}))
    with pytest.raises(ValueError, match="label_map"):
        real_data.run(str(csv), str(mp), FakeLLM(lambda s, u: {}))


def test_render_md_with_ablation_and_prompt():
    from eval import run_eval
    assert "groceries" in ablation.ADVICE_SYSTEM and "txn_amount" in ablation.ADVICE_SYSTEM
    p = generate(seed=0)
    llm = FakeLLM(lambda s, u: {})
    import eval.run_eval as R, tempfile, pathlib
    out = pathlib.Path(tempfile.mkdtemp())
    old = R.OUT
    R.OUT = out
    try:
        res = R.main(["--no-llm"])
    finally:
        R.OUT = old
    res["ablation"] = ablation.run({"a": p}, FakeLLM(lambda s, u: {"next_month_net_flow": 0, "recommendations": []}))
    res["grounding"] = {"recs": 4, "first_pass_ok": 3, "first_pass_caught": 1, "first_pass_grounding_rate": 0.75, "final_verified": 4}
    md = R.render_md(res)
    assert "0.25 of 4 recs caught" in md and "flat-balance" in md and "(0 recs)" in md
