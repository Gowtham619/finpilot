import pytest
from eval import metrics as M


def test_accuracy_and_f1():
    y, p = ["a", "a", "b", "b"], ["a", "b", "b", "b"]
    assert M.accuracy(y, p) == 0.75
    # class a: P=1 R=.5 F1=2/3; class b: P=2/3 R=1 F1=.8 -> macro .7333
    assert M.macro_f1(y, p) == pytest.approx(0.7333, abs=1e-3)


def test_precision_recall():
    r = M.precision_recall({1, 2, 3}, {2, 3, 4})
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert M.precision_recall(set(), set()) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert M.precision_recall(set(), {1})["recall"] == 0.0


def test_brier():
    assert M.brier([1.0, 0.0], [1, 0]) == 0.0
    assert M.brier([0.5, 0.5], [1, 0]) == 0.25


def test_baseline_paths_tiny_series():
    import numpy as np
    from eval import run_eval
    p = run_eval.baseline_paths(100.0, [10, -20, 5], 4, 2.0)
    assert list(p["flat_balance"]) == [100.0] * 4
    assert list(p["seasonal_naive"]) == [110.0, 90.0, 95.0, 95.0]  # prev month replayed, zero-padded
    assert list(p["trailing_mean"]) == [102.0, 104.0, 106.0, 108.0]
    assert run_eval._mape(np.array([0.0]), np.array([0.0])) == 0.0  # zero divisor guarded


def test_run_eval_no_llm_smoke(tmp_path, monkeypatch):
    from eval import run_eval
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    res = run_eval.main(["--no-llm"])
    assert res["categorization"]["hybrid"] is None
    assert 0 < res["categorization"]["rules_only"]["accuracy"]["mean"] < 1
    assert res["detection"]["duplicate"]["recall"]["mean"] == 1.0
    fc = res["forecast"]
    assert fc["model"]["band_coverage"]["mean"] > 0.3
    for b in ("flat_balance", "seasonal_naive", "trailing_mean"):
        assert b in fc
    assert fc["model"]["balance_mape"]["mean"] < fc["flat_balance"]["balance_mape"]["mean"]
    assert fc["model"]["balance_mape"]["min"] <= fc["model"]["balance_mape"]["max"]
    assert len(fc["model"]["balance_mape"]["values"]) == res["forecast_meta"]["n_personas"]
    cal = res["calibration"]
    assert "base_brier" in cal and sum(b["count"] for b in cal["bins"]) == cal["n"]
    assert res["methodology"]
    md = (tmp_path / "results.md").read_text()
    assert "Methodology and limitations" in md and "trailing-30d-mean" in md
    assert (tmp_path / "results.json").exists()
    assert res.get("ablation") is None
    bt = res["backtest"]
    assert set(bt) == set(res["personas"]) and all(b["potential_savings"] > 0 for b in bt.values())
    assert "Advice backtest" in md and "n/a (--no-llm)" in md
