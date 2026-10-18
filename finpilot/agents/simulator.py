import pandas as pd

from ..engines import simulate as S
from .common import fmt_bal, grounded_or, inr, narrate

SYSTEM = ("You answer 'can I afford this?'. Use only the numbers given, in two or three sentences. "
          "Return JSON {\"narrative\": str}.")


def _verdict(p):
    return "safe" if p <= 0.05 else "risky" if p <= 0.25 else "unsafe"


def afford(model, balance, req, llm=None):
    """Raises ValueError if req["date"] is outside the forecast window; the API layer maps it to HTTP 400."""
    fm = int(req.get("financed_months") or 0)
    sim = S.simulate_purchase(model, balance, req["amount"], req["date"], financed_months=fm)
    flows = S.purchase_flows(req["amount"], req["date"], fm)
    start = pd.Timestamp(req["date"])
    cands = [start + pd.Timedelta(days=7 * k) for k in range(9)]
    best = S.best_date(model, balance, req["amount"], cands)[:3]
    fix = S.cheapest_fix(model, balance, flows)
    a, b = sim["with_purchase"], sim["baseline"]
    verdict = _verdict(a["p_overdraft"])
    text = (f"Buying {inr(req['amount'])} on {sim['date']} moves your overdraft risk from "
            f"{b['p_overdraft']*100:.0f}% to {a['p_overdraft']*100:.0f}% "
            f"(median balance in 60 days {fmt_bal(a['end_p50'])}). Verdict: {verdict}.")
    if best and best[0]["date"] != sim["date"]:
        text += f" Lowest-risk date tested: {best[0]['date']} ({best[0]['p_overdraft']*100:.0f}% risk)."
    if fix and fix["cut_pct"]:
        text += (f" To make it safe, cut {fix['category'].replace('_', ' ')} by {fix['cut_pct']}% "
                 f"(about {inr(fix['est_monthly_saving'])}/month).")
    evidence = {"sim": sim, "best": best, "fix": fix, "amount": req["amount"]}
    out = narrate(llm, SYSTEM, evidence, None)
    narrative = grounded_or(out.get("narrative") if isinstance(out, dict) else None, evidence, text)
    res = {"verdict": verdict, "sim": sim, "best_dates": best, "fix": fix, "narrative": narrative}
    n_ign = int(sim.get("ignored_flows") or 0)
    if n_ign > 0:  # deterministic, added after grounding so it cannot fail it
        res["notice"] = f"{n_ign} installment(s) fall outside the 60-day window and are not simulated"
        res["narrative"] = f"{narrative} ({res['notice']}.)"
    return res


def run_duel(model, balance, req, llm=None):
    def opt(o):
        return {"name": o["name"],
                "flows": S.purchase_flows(o.get("amount") or 0, o["date"], int(o.get("financed_months") or 0))
                if o.get("amount") else [],
                "cuts": o.get("cuts") or {}}
    res = S.duel(model, balance, opt(req["a"]), opt(req["b"]), horizon=int(req.get("horizon", 90)))
    out = narrate(llm, SYSTEM, {"duel": {k: res[k] for k in ("winner", "reason")},
                                "a": res["a"]["summary"], "b": res["b"]["summary"]}, None)
    res["narrative"] = grounded_or(out.get("narrative") if isinstance(out, dict) else None,
                                   {"a": res["a"]["summary"], "b": res["b"]["summary"], "h": res["horizon"]},
                                   res["reason"])
    if res.get("ignored_flows"):
        res["notice"] = (f"{res['ignored_flows']} installment(s) fall outside the "
                         f"{res['horizon']}-day window and are not simulated.")
    return res
