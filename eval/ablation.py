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
from eval.run_eval import forecast_origins
from finpilot.config import CATEGORIES
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
                 "Spending categories: " + ", ".join(CATEGORIES) + ". Claim kinds the checker can verify: monthly_cost, annual_cost "
                 "(need merchant), month_total, pct_change (need category and month), txn_amount (needs txn_id). "
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


def llm_only_forecast(persona, llm, origins=3):
    """Same protocol as the model: at each of the shared monthly origins, give the LLM the monthly
    net flows before the origin and ask for that month's net flow."""
    df = _with_truth(persona)
    apes = []
    for cutoff in forecast_origins(df, origins):
        month = cutoff.to_period("M")
        train, test = df[df["date"] < cutoff], df[df["date"].dt.to_period("M") == month]
        hist = train.groupby(train["date"].dt.to_period("M").astype(str))["amount"].sum().round(0)
        try:
            res = llm.json("Given monthly net cash flow history, predict next month's net flow. "
                           "Return JSON {\"next_month_net_flow\": number}.", hist.to_csv())
            pred = float(res["next_month_net_flow"])
        except (LLMUnavailable, ValueError, KeyError, TypeError):
            continue
        true = float(test["amount"].sum())
        apes.append(abs(pred - true) / max(abs(true), 1.0))
    if not apes:
        return {"net_flow_ape": None, "n_used": 0}
    return {"net_flow_ape": round(float(np.mean(apes)), 4), "n_used": len(apes)}


def _mean_dict(items, keys=("accuracy", "macro_f1")):
    """Mean over personas that produced every key; n_used records how many did."""
    ok = [x for x in items if x and all(x.get(k) is not None for k in keys)]
    out = {k: round(float(np.mean([x[k] for x in ok])), 4) if ok else None for k in keys}
    out["n_used"] = len(ok)
    return out


def run(personas, llm):
    ps = list(personas.values())
    cats = [eval_categorization(p, llm) for p in ps]
    adv = [llm_only_advice(p, llm) for p in ps]
    recs = sum(a["recs"] for a in adv)
    fc = [llm_only_forecast(p, llm) for p in ps]
    used = [x for x in fc if x["net_flow_ape"] is not None]
    n_orig = sum(x["n_used"] for x in used)
    return {"categorization": {"rules_only": _mean_dict([c["rules_only"] for c in cats]),
                               "llm_only": _mean_dict([llm_only_categorization(p, llm) for p in ps]),
                               "hybrid": _mean_dict([c["hybrid"] for c in cats])},
            "advice": {"llm_only": {"recs": recs, "with_issues": sum(a["with_issues"] for a in adv),
                                    "hallucination_rate": round(sum(a["with_issues"] for a in adv) / recs, 4) if recs else None}},
            "forecast": {"llm_only": {"net_flow_ape": round(float(np.mean([x["net_flow_ape"] for x in used])), 4) if used else None,
                                      "n_used": n_orig, "n_personas_used": len(used)}},
            "meta": {"runs": 1, "temperature": 0, "n_personas": len(ps)}}
