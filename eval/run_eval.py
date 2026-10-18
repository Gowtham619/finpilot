"""Self-evaluation. Run: python -m eval.run_eval [--no-llm]. Writes eval/results.json and eval/results.md."""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

from eval import metrics as M
from eval.generate import PERSONAS, load_persona
from finpilot import graph
from finpilot.agents.categorizer import categorize_df
from finpilot.engines import analyze as A
from finpilot.engines import forecast as F
from finpilot.engines.categorize import normalize_merchant
from finpilot.llm import LLM
from finpilot.trace import Trace

OUT = pathlib.Path(__file__).parent


def _with_truth(persona):
    df = persona.df.copy()
    df["merchant"] = df["description"].map(normalize_merchant)
    df["category"] = persona.truth.categories
    return df


def eval_categorization(persona, llm):
    truth = persona.truth.categories
    out = {}
    for name, use in (("rules_only", None), ("hybrid", llm)):
        if name == "hybrid" and llm is None:
            out[name] = None
            continue
        pred = categorize_df(persona.df, use)["category"]
        out[name] = {"accuracy": round(M.accuracy(truth, pred), 4), "macro_f1": round(M.macro_f1(truth, pred), 4),
                     "coverage": round(float((pred != "uncategorized").mean()), 4)}
    return out


def eval_detection(persona):
    df, t = _with_truth(persona), persona.truth
    rec = {r["merchant"] for r in A.detect_recurring(df)}
    subs = A.detect_subscriptions(df)
    return {
        "large_charge": M.precision_recall({f["txn_id"] for f in A.txn_outliers(df, rec)}, t.anomaly_ids),
        "duplicate": M.precision_recall({f["txn_id"] for f in A.duplicate_charges(df)}, t.duplicate_ids),
        "category_spike": M.precision_recall({(f["month"], f["category"]) for f in A.category_spikes(df)}, {t.spike}),
        "subscriptions": M.precision_recall({s["merchant"] for s in subs},
                                            {"NETFLIX", "HOTSTAR", "SPOTIFY", "CULT FIT GYM"}),
        "unused_subscriptions": M.precision_recall({u["merchant"] for u in A.unused_subscriptions(df, subs)}, t.unused_subs),
        "payday_trigger": M.precision_recall({f["category"] for f in A.behavior_triggers(df, rec)},
                                             {t.payday_trigger_category}),
    }


def _holdout(df, opening, cutoff):
    train = df[df["date"] < cutoff]
    end = (cutoff + pd.offsets.MonthEnd(0)).normalize()
    test = df[(df["date"] >= cutoff) & (df["date"] <= end)]
    model = F.fit(train)
    bal0 = F.current_balance(train, opening)
    days = pd.date_range(cutoff, end)
    actual = bal0 + test.groupby(test["date"].dt.normalize())["amount"].sum().reindex(days, fill_value=0.0).cumsum()
    return train, model, bal0, days, actual.to_numpy()


def eval_forecast(persona, origins=3):
    df = _with_truth(persona)
    last_month = df["date"].max().to_period("M")
    acc = {"model": {"ape": [], "mape": [], "cov": []}, "naive": {"ape": [], "mape": [], "cov": []}}
    for k in range(origins):
        cutoff = (last_month - k).to_timestamp()
        train, model, bal0, days, actual = _holdout(df, persona.opening_balance, cutoff)
        model.last_date = cutoff - pd.Timedelta(days=1)  # forecast starts on the cutoff day
        r = F.monte_carlo(model, bal0, len(days), 1000)
        true_net = actual[-1] - bal0
        acc["model"]["ape"].append(abs((r.p50[-1] - bal0) - true_net) / max(abs(true_net), 1.0))
        acc["model"]["mape"].append(float(np.mean(np.abs(r.p50 - actual) / np.abs(actual))))
        acc["model"]["cov"].append(float(np.mean((actual >= r.p10) & (actual <= r.p90))))
        last30 = train[train["date"] >= cutoff - pd.Timedelta(days=30)]["amount"].sum() / 30
        naive = bal0 + last30 * np.arange(1, len(days) + 1)
        acc["naive"]["ape"].append(abs((naive[-1] - bal0) - true_net) / max(abs(true_net), 1.0))
        acc["naive"]["mape"].append(float(np.mean(np.abs(naive - actual) / np.abs(actual))))
    return {n: {"net_flow_ape": round(float(np.mean(v["ape"])), 4), "balance_mape": round(float(np.mean(v["mape"])), 4),
                "band_coverage": round(float(np.mean(v["cov"])), 4) if v["cov"] else None}
            for n, v in acc.items()}


def eval_calibration(personas, horizon=30, drop=0.10):
    """Event: balance falls more than `drop` below its starting level within `horizon` days."""
    probs, outcomes = [], []
    for p in personas:
        df = _with_truth(p)
        for cutoff in pd.date_range(df["date"].min() + pd.Timedelta(days=100),
                                    df["date"].max() - pd.Timedelta(days=horizon), freq="7D"):
            train = df[df["date"] < cutoff]
            model, bal0 = F.fit(train), F.current_balance(train, p.opening_balance)
            model.last_date = cutoff - pd.Timedelta(days=1)
            r = F.monte_carlo(model, bal0, horizon, 500)
            probs.append(float((r.paths.min(axis=1) < bal0 * (1 - drop)).mean()))
            test = df[(df["date"] >= cutoff) & (df["date"] < cutoff + pd.Timedelta(days=horizon))]
            days = pd.date_range(cutoff, periods=horizon)
            path = bal0 + test.groupby(test["date"].dt.normalize())["amount"].sum().reindex(days, fill_value=0.0).cumsum()
            outcomes.append(int(path.min() < bal0 * (1 - drop)))
    return {"n": len(probs), "mean_pred": round(float(np.mean(probs)), 4), "observed": round(float(np.mean(outcomes)), 4),
            "brier": round(M.brier(probs, outcomes), 4)}


def eval_grounding(persona, llm):
    tr = Trace()
    g = graph.build_graph(llm, tr)
    out = g.invoke({"run_id": "eval", "profile": "eval", "request": {"type": "pipeline"}, "df": persona.df.copy(),
                    "opening": persona.opening_balance, "corrections": {}, "goals": []})
    kinds = [e["kind"] for e in tr.since("eval") if e["agent"] == "verifier"]
    ok, caught = kinds.count("ok"), kinds.count("caught")
    return {"recs": ok + caught, "first_pass_ok": ok, "first_pass_caught": caught,
            "first_pass_grounding_rate": round(ok / max(ok + caught, 1), 4),
            "final_verified": sum(r["verified"] for r in out["recs"])}


def _avg(dicts):
    if not dicts or dicts[0] is None:
        return None
    if isinstance(dicts[0], dict):
        return {k: _avg([d[k] for d in dicts]) for k in dicts[0]}
    return round(float(np.mean(dicts)), 4)


def render_md(res):
    c, f = res["categorization"], res["forecast"]
    d = res["detection"]
    rows = ["# Evaluation results", "", f"Personas: {', '.join(res['personas'])} (synthetic, known ground truth)", "",
            "## Categorization", "| Method | Accuracy | Macro-F1 | Coverage |", "|---|---|---|---|"]
    for name in ("rules_only", "hybrid"):
        m = c.get(name)
        rows.append(f"| {name} | " + (" | ".join(str(m[k]) for k in ("accuracy", "macro_f1", "coverage")) if m else "n/a | n/a | n/a") + " |")
    rows += ["", "## Detection (precision / recall)", "| Task | Precision | Recall |", "|---|---|---|"]
    rows += [f"| {k} | {v['precision']} | {v['recall']} |" for k, v in d.items()]
    rows += ["", "## Forecast (held-out months)", "| Method | Net-flow APE | Balance MAPE | p10-p90 coverage |", "|---|---|---|---|"]
    for n in ("model", "naive"):
        rows.append(f"| {n} | {f[n]['net_flow_ape']} | {f[n]['balance_mape']} | {'-' if f[n]['band_coverage'] is None else f[n]['band_coverage']} |")
    cal = res["calibration"]
    rows += ["", f"## Calibration\nn={cal['n']}, mean predicted {cal['mean_pred']}, observed {cal['observed']}, Brier {cal['brier']}", ""]
    g = res.get("grounding")
    rows.append("## Grounding\n" + (f"first-pass grounding rate {g['first_pass_grounding_rate']} "
                                    f"({g['first_pass_caught']} caught of {g['recs']}), final verified {g['final_verified']}" if g else "n/a (--no-llm)"))
    return "\n".join(rows)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    use_llm = "--no-llm" not in argv
    llm = LLM() if use_llm else None
    personas = {n: load_persona(n) for n in PERSONAS}
    res = {"personas": list(personas), "model": getattr(llm, "model", None)}
    cats = [eval_categorization(p, llm) for p in personas.values()]
    res["categorization"] = {"rules_only": _avg([c["rules_only"] for c in cats]),
                             "hybrid": _avg([c["hybrid"] for c in cats]) if use_llm else None}
    res["detection"] = _avg([eval_detection(p) for p in personas.values()])
    res["forecast"] = _avg([eval_forecast(p) for p in personas.values()])
    res["calibration"] = eval_calibration(list(personas.values()))
    res["grounding"] = eval_grounding(personas["priya"], llm) if use_llm else None
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    (OUT / "results.md").write_text(render_md(res))
    print(render_md(res))
    return res


if __name__ == "__main__":
    main()
