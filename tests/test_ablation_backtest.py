import json
import pytest
from eval import ablation, backtest
from eval.generate import generate
from tests.conftest import FakeLLM


def test_backtest_finds_real_savings():
    p = generate(seed=0)
    r = backtest.run(p, days=90)
    assert r["potential_savings"] > 0
    assert 0 <= r["pct_of_spend"] < 1
    assert r["cancel_precision"] == 1.0  # every cancelled sub is a planted unused one
    assert r["followed"]


def test_backtest_never_peeks_at_future():
    p = generate(seed=0)
    r = backtest.run(p, days=90)
    train_end = r["cutoff"]
    assert train_end < str(p.df["date"].max().date())


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
    assert r["net_flow_ape"] > 0


def test_baselines_survive_llm_unavailable():
    from finpilot.llm import LLMUnavailable
    p = generate(seed=0)

    def boom(s, u):
        raise LLMUnavailable("no key")

    llm = FakeLLM(boom)
    assert ablation.llm_only_categorization(p, llm) == {"accuracy": None, "macro_f1": None}
    assert ablation.llm_only_advice(p, llm)["hallucination_rate"] is None
    assert ablation.llm_only_forecast(p, llm)["net_flow_ape"] is None
    r = ablation.run({"a": p}, llm)
    assert r["categorization"]["llm_only"]["accuracy"] is None
    assert r["categorization"]["hybrid"]["accuracy"] is not None or r["categorization"]["hybrid"]["accuracy"] is None


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
