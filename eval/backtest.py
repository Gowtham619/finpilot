"""Advice backtest: apply the system at a past cutoff, then measure what following its advice
would have saved over the following period. Assumes the user complied with everything; it is an
optimistic upper bound, NOT a prediction of behaviour.

Two separate numbers are reported:
- cancel_savings (headline, deterministic): post-cutoff spend at the merchants flagged for cancellation.
- excess_savings (noisy upper bound): post-cutoff monthly category spend above the pre-cutoff baseline
  of each flagged spike category, prorated for partial months, net of cancelled merchants."""
from __future__ import annotations

import pandas as pd

from eval.run_eval import _with_truth
from finpilot.agents.advisor import build_candidates
from finpilot.engines.analyze import analyze_all


def split_at_cutoff(df, days=90):
    cutoff = df["date"].max().normalize() - pd.Timedelta(days=days)
    return cutoff, df[df["date"] <= cutoff], df[df["date"] > cutoff]


def candidates_at(df, cutoff):
    """Recommendations the system would make at `cutoff`, using only rows dated <= cutoff."""
    train = df[df["date"] <= cutoff]
    assert train["date"].max() <= cutoff
    return build_candidates(analyze_all(train), train)


def _days_present(month, test):
    start, end = month.start_time.normalize(), month.end_time.normalize()
    lo, hi = max(start, test["date"].min().normalize()), min(end, test["date"].max().normalize())
    return max((hi - lo).days + 1, 0), month.days_in_month


def run(persona, days=90):
    df = _with_truth(persona)
    cutoff, _, test = split_at_cutoff(df, days)
    cands = candidates_at(df, cutoff)
    spend = test[test["amount"] < 0]
    cancelled, followed, cancel_total = set(), [], 0.0
    for c in cands:
        if c["trigger_type"] == "unused_subscription":
            m = c["evidence"]["merchant"]
            spent = float(-spend[spend["merchant"] == m]["amount"].sum())
            cancel_total += spent
            cancelled.add(m)
            followed.append({"action": c["action"], "kind": "cancel", "saved": round(spent, 2)})
    cancelled_spend = spend[spend["merchant"].isin(cancelled)]
    excess_total = 0.0
    for c in cands:
        if c["trigger_type"] != "category_spike":
            continue
        e = c["evidence"]
        t = spend[spend["category"] == e["category"]]
        t = t[~t["merchant"].isin(cancelled)]  # avoid double counting with cancel savings
        monthly = -t.groupby(t["date"].dt.to_period("M"))["amount"].sum()
        ex = 0.0
        for month, v in monthly.items():
            present, dim = _days_present(month, test)
            ex += max(float(v) - e["baseline"] * present / dim, 0.0)
        excess_total += ex
        followed.append({"action": c["action"], "kind": "category_excess", "saved": round(ex, 2)})
    total = float(-spend["amount"].sum())
    truth = persona.truth.unused_subs
    k = len(cancelled & truth)
    return {"cutoff": str(cutoff.date()), "followed": followed,
            "cancel_savings": round(cancel_total, 2), "cancel_pct_of_spend": round(cancel_total / total, 4) if total else 0.0,
            "excess_savings": round(excess_total, 2), "excess_pct_of_spend": round(excess_total / total, 4) if total else 0.0,
            "n_cancelled": len(cancelled), "n_unused_truth": len(truth), "n_correct": k,
            "cancel_precision": round(k / len(cancelled), 3) if cancelled else None,
            "cancel_recall": round(k / len(truth), 3) if truth else None}
