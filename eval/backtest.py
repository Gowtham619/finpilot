"""Advice backtest: apply the system at a past cutoff, then measure what following its advice
would have saved over the following period. This is an upper bound that assumes the user
complied; it is NOT a prediction of behaviour."""
from __future__ import annotations

import pandas as pd

from eval.run_eval import _with_truth
from finpilot.agents.advisor import build_candidates
from finpilot.engines.analyze import analyze_all


def run(persona, days=90):
    df = _with_truth(persona)
    cutoff = df["date"].max().normalize() - pd.Timedelta(days=days)
    train, test = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    cands = build_candidates(analyze_all(train), train)
    followed, savings = [], 0.0
    cancelled = set()
    for c in cands:
        e = c["evidence"]
        if c["trigger_type"] == "unused_subscription":
            spent = float(-test[(test["merchant"] == e["merchant"]) & (test["amount"] < 0)]["amount"].sum())
            savings += spent
            cancelled.add(e["merchant"])
            followed.append({"action": c["action"], "saved": round(spent, 2)})
        elif c["trigger_type"] == "category_spike":
            t = test[(test["category"] == e["category"]) & (test["amount"] < 0)]
            monthly = (-t.groupby(t["date"].dt.to_period("M"))["amount"].sum())
            excess = float((monthly - e["baseline"]).clip(lower=0).sum())
            savings += excess
            followed.append({"action": c["action"], "saved": round(excess, 2)})
    total = float(-test[test["amount"] < 0]["amount"].sum())
    precision = (len(cancelled & persona.truth.unused_subs) / len(cancelled)) if cancelled else 1.0
    return {"cutoff": str(cutoff.date()), "followed": followed, "potential_savings": round(savings, 2),
            "pct_of_spend": round(savings / total, 4) if total else 0.0, "cancel_precision": round(precision, 3)}
