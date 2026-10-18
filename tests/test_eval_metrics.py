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


def test_run_eval_no_llm_smoke(tmp_path, monkeypatch):
    from eval import run_eval
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    res = run_eval.main(["--no-llm"])
    assert res["categorization"]["hybrid"] is None
    assert 0 < res["categorization"]["rules_only"]["accuracy"] < 1
    assert res["detection"]["duplicate"]["recall"] == 1.0
    assert res["forecast"]["model"]["band_coverage"] > 0.3
    assert (tmp_path / "results.json").exists() and (tmp_path / "results.md").exists()
