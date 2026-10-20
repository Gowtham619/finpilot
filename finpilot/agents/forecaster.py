import pandas as pd

from ..engines.forecast import current_balance, fit, goal_status, monte_carlo
from .common import fmt_bal, grounded_or, inr, narrate

SYSTEM = ("Explain this cash-flow forecast in two plain sentences using only the numbers given. "
          "Return JSON {\"narrative\": str}.")


def run(df, opening, goals=(), llm=None, horizon=60):
    """Goals are validated individually: an invalid goal (not a dict, missing name/target/deadline,
    non-numeric target, unparseable deadline) yields {"name": ..., "error": reason} instead of raising.
    A deadline in the past is valid (goal horizon is clamped to 1 day)."""
    model = fit(df)
    bal = current_balance(df, opening)
    res = monte_carlo(model, bal, horizon)
    s = res.summary()
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
    fallback = (f"In {horizon} days your balance is projected at {fmt_bal(s['end_p50'])} "
                f"(likely range {fmt_bal(s['end_p10'])} to {fmt_bal(s['end_p90'])}). "
                f"Chance of going overdrawn: {s['p_overdraft']*100:.0f}%.")
    out = narrate(llm, SYSTEM, {"summary": s, "horizon": horizon, "balance": bal}, None)
    narrative = grounded_or(out.get("narrative") if isinstance(out, dict) else None,
                            {"s": s, "h": horizon, "b": bal}, fallback)
    return {"balance": round(bal, 2), "horizon": horizon, "summary": s, "bands": bands,
            "goals": gs, "narrative": narrative, "model": model}
