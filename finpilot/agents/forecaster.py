import pandas as pd

from ..engines.forecast import current_balance, fit, goal_status, monte_carlo
from .common import fmt_bal, grounded_or, inr, narrate

SYSTEM = ("Explain this cash-flow forecast in two plain sentences using only the numbers given. "
          "Return JSON {\"narrative\": str}.")


def run(df, opening, goals=(), llm=None, horizon=60):
    model = fit(df)
    bal = current_balance(df, opening)
    res = monte_carlo(model, bal, horizon)
    s = res.summary()
    bands = {"dates": [str(d.date()) for d in res.dates], "p10": [round(float(x), 2) for x in res.p10],
             "p50": [round(float(x), 2) for x in res.p50], "p90": [round(float(x), 2) for x in res.p90]}
    gs = []
    for g in goals:
        st = goal_status(model, bal, g["target"], pd.Timestamp(g["deadline"]))
        gs.append({"name": g["name"], "target": g["target"], "deadline": str(g["deadline"]), **st})
    fallback = (f"In {horizon} days your balance is projected at {fmt_bal(s['end_p50'])} "
                f"(likely range {fmt_bal(s['end_p10'])} to {fmt_bal(s['end_p90'])}). "
                f"Chance of going overdrawn: {s['p_overdraft']*100:.0f}%.")
    out = narrate(llm, SYSTEM, {"summary": s, "horizon": horizon, "balance": bal}, None)
    narrative = grounded_or(out.get("narrative") if isinstance(out, dict) else None,
                            {"s": s, "h": horizon, "b": bal}, fallback)
    return {"balance": round(bal, 2), "horizon": horizon, "summary": s, "bands": bands,
            "goals": gs, "narrative": narrative, "model": model}
