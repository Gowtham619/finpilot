import pandas as pd

from ..engines.forecast import current_balance, fit, goal_status, monte_carlo
from .common import fmt_bal, grounded_or, narrate, pct_text, pct_values, risk_claim_consistent, risk_level

SYSTEM = ("Explain this cash-flow forecast in two plain sentences using only the numbers given. "
          "Return JSON {\"narrative\": str}.")


def strict_evidence(s, horizon, bal):
    """Only the numbers a forecast narration may state: money amounts (unsigned) and explicit percents."""
    return {"horizon": horizon, "balance": abs(bal),
            "money": [abs(s[k]) for k in ("end_p10", "end_p50", "end_p90", "floor_p50")],
            "p_overdraft_pct": pct_values(s["p_overdraft"])}


def run(df, opening, goals=(), llm=None, horizon=60):
    """Goals are validated individually: an invalid goal (not a dict, missing name/target/deadline,
    non-numeric target, unparseable deadline) yields {"name": ..., "error": reason} instead of raising.
    A deadline in the past is valid (goal horizon is clamped to 1 day)."""
    model = fit(df)
    bal = current_balance(df, opening)
    res = monte_carlo(model, bal, horizon)
    s = res.summary()  # risk (overdraft chance, floor, first risk day) always uses every real charge
    # The "typical" range and median leave out flagged one-off charges so a single large purchase does not keep
    # pulling every future month down. Rolling-origin backtest: lower error, bias near 0, coverage near nominal
    # (eval/forecast_experiments.md, post-hoc section).
    typical = monte_carlo(fit(df, robust_pool=True), bal, horizon)
    s.update(end_p10=round(typical.end_p10, 2), end_p50=round(typical.end_p50, 2), end_p90=round(typical.end_p90, 2))
    res = typical
    bands = {"dates": [str(d.date()) for d in res.dates], "p10": [round(float(x), 2) for x in res.p10],
             "p50": [round(float(x), 2) for x in res.p50], "p90": [round(float(x), 2) for x in res.p90]}
    gs = []
    for g in goals:
        try:
            name = g.get("name", "?")
            if not all(k in g for k in ("name", "target", "deadline")):
                raise KeyError("missing " + "/".join(k for k in ("name", "target", "deadline") if k not in g))
            target = float(g["target"])
            deadline = pd.Timestamp(g["deadline"])
            if pd.isna(deadline):
                raise ValueError("unparseable deadline")
            st = goal_status(model, bal, target, deadline)
            gs.append({"name": g["name"], "target": g["target"], "deadline": str(g["deadline"]), **st})
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as e:
            gs.append({"name": name if isinstance(g, dict) else "?", "error": f"invalid goal: {e}"})
    fallback = (f"In {horizon} days you will most likely have about {fmt_bal(s['end_p50'])}, "
                f"typically between {fmt_bal(s['end_p10'])} and {fmt_bal(s['end_p90'])}. "
                f"Chance of going overdrawn at some point: {pct_text(s['p_overdraft'])}.")
    out = narrate(llm, SYSTEM, {"summary": s, "horizon": horizon, "balance": bal}, None)
    text = out.get("narrative") if isinstance(out, dict) else None
    if not risk_claim_consistent(text, risk_level(s["p_overdraft"]), [s["p_overdraft"]]):
        text = None  # e.g. "zero chance of going overdrawn" while the computed risk is 2%
    narrative = grounded_or(text, strict_evidence(s, horizon, bal), fallback, strict=True)
    return {"balance": round(bal, 2), "horizon": horizon, "summary": s, "bands": bands,
            "goals": gs, "narrative": narrative, "model": model}
