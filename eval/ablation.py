"""LLM-only baselines. They receive the same raw data summary but no engines, to test whether
the hybrid design is worth its complexity."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eval import metrics as M
from eval.run_eval import _with_truth, eval_categorization
from finpilot.agents.categorizer import _llm_map
from finpilot.agents.verifier import audit
from finpilot.engines.categorize import normalize_merchant
from finpilot.llm import LLMUnavailable


def llm_only_categorization(persona, llm):
    merchants = sorted({normalize_merchant(d) for d in persona.df["description"]})
    mapping = _llm_map(set(merchants), llm)
    if not mapping:
        return {"accuracy": None, "macro_f1": None}
    pred = [mapping.get(normalize_merchant(d), "uncategorized") for d in persona.df["description"]]
    truth = persona.truth.categories
    return {"accuracy": round(M.accuracy(truth, pred), 4), "macro_f1": round(M.macro_f1(truth, pred), 4)}


ADVICE_SYSTEM = ("You are a financial advisor. From the monthly spend by merchant, write up to 6 specific recommendations. "
                 "Return JSON {\"recommendations\":[{\"text\":str,\"claims\":[{\"kind\":\"monthly_cost\"|\"annual_cost\"|\"month_total\"|\"pct_change\","
                 "\"merchant\":str (for cost kinds),\"category\":str,\"month\":\"YYYY-MM\" (for month kinds),\"value\":number}]}]}")


def _merchant_table(df):
    s = df[df["amount"] < 0].copy()
    s["month"] = s["date"].dt.to_period("M").astype(str)
    t = (-s.groupby(["merchant", "month"])["amount"].sum()).round(0).unstack(fill_value=0)
    return t.to_csv()


def llm_only_advice(persona, llm):
    df = _with_truth(persona)
    try:
        res = llm.json(ADVICE_SYSTEM, _merchant_table(df))
    except (LLMUnavailable, ValueError):
        return {"recs": 0, "with_issues": 0, "hallucination_rate": None, "ungrounded_numbers": 0}
    recs = [r for r in res.get("recommendations", []) if "text" in r]
    with_issues, ungrounded = 0, 0
    for r in recs:
        rec = {"text": r["text"], "claims": r.get("claims") or [], "evidence": {}}
        issues = audit(rec, df)
        with_issues += bool(issues)
        ungrounded += sum(i["type"] == "ungrounded_number" for i in issues)
    return {"recs": len(recs), "with_issues": with_issues,
            "hallucination_rate": round(with_issues / len(recs), 4) if recs else None,
            "ungrounded_numbers": ungrounded}


def llm_only_forecast(persona, llm):
    df = _with_truth(persona)
    month = df["date"].max().to_period("M")
    train, test = df[df["date"].dt.to_period("M") < month], df[df["date"].dt.to_period("M") == month]
    hist = train.groupby(train["date"].dt.to_period("M").astype(str))["amount"].sum().round(0)
    try:
        res = llm.json("Given monthly net cash flow history, predict next month's net flow. "
                       "Return JSON {\"next_month_net_flow\": number}.", hist.to_csv())
        pred = float(res["next_month_net_flow"])
    except (LLMUnavailable, ValueError, KeyError, TypeError):
        return {"net_flow_ape": None}
    true = float(test["amount"].sum())
    return {"net_flow_ape": round(abs(pred - true) / max(abs(true), 1.0), 4)}


def _mean_dict(items, keys=("accuracy", "macro_f1")):
    items = [x for x in items if x and all(x.get(k) is not None for k in keys)]
    return {k: round(float(np.mean([x[k] for x in items])), 4) for k in keys} if items else {k: None for k in keys}


def run(personas, llm):
    ps = list(personas.values())
    cats = [eval_categorization(p, llm) for p in ps]
    adv = [llm_only_advice(p, llm) for p in ps]
    recs = sum(a["recs"] for a in adv)
    fa = [x for x in (llm_only_forecast(p, llm)["net_flow_ape"] for p in ps) if x is not None]
    return {"categorization": {"rules_only": _mean_dict([c["rules_only"] for c in cats]),
                               "llm_only": _mean_dict([llm_only_categorization(p, llm) for p in ps]),
                               "hybrid": _mean_dict([c["hybrid"] for c in cats])},
            "advice": {"llm_only": {"recs": recs, "with_issues": sum(a["with_issues"] for a in adv),
                                    "hallucination_rate": round(sum(a["with_issues"] for a in adv) / recs, 4) if recs else None}},
            "forecast": {"llm_only": {"net_flow_ape": round(float(np.mean(fa)), 4) if fa else None}},
            "meta": {"runs": 1, "temperature": 0, "n_personas": len(ps)}}
