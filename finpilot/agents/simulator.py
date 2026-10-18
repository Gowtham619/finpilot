import re

import pandas as pd

from ..engines import simulate as S
from .common import fmt_bal, grounded_or, inr, narrate

SYSTEM = ("You answer 'can I afford this?'. Use only the numbers given, in two or three sentences. "
          "Return JSON {\"narrative\": str}.")


def _verdict(p):
    return "safe" if p <= 0.05 else "risky" if p <= 0.25 else "unsafe"


_VERDICT_WORD = re.compile(r"\b(safe|risky|unsafe)\b", re.I)
FLOOR_MARGIN = 1000.0  # an equal-risk date must raise the median lowest balance by at least this much


def verdict_consistent(text, verdict):
    """False when the text uses a verdict word (safe/risky/unsafe) other than the computed one."""
    return {w.lower() for w in _VERDICT_WORD.findall(str(text))} <= {verdict}


def afford_horizon(financed_months):
    """Long enough to cover the installments (up to ~11 months), capped at a year."""
    return min(365, max(60, 31 * financed_months + 31)) if financed_months else 60


def better_date(best, chosen):
    """The best tested date if it is really better than the chosen one, else None."""
    if not best or best[0]["date"] == chosen["date"]:
        return None
    b = best[0]
    if b["p_overdraft"] < chosen["p_overdraft"]:
        return b
    if (b["p_overdraft"] == chosen["p_overdraft"] and b["floor_p50"] >= 0
            and b["floor_p50"] >= chosen["floor_p50"] + FLOOR_MARGIN):
        return b
    return None


def afford(model, balance, req, llm=None):
    """Raises ValueError if req["date"] is outside the forecast window; the API layer maps it to HTTP 400."""
    fm = int(req.get("financed_months") or 0)
    hz = afford_horizon(fm)
    sim = S.simulate_purchase(model, balance, req["amount"], req["date"], horizon=hz, financed_months=fm)
    flows = S.purchase_flows(req["amount"], req["date"], fm)
    start = pd.Timestamp(req["date"])
    cands = [start + pd.Timedelta(days=7 * k) for k in range(9)]
    best = S.best_date(model, balance, req["amount"], cands, horizon=hz, financed_months=fm)[:3]
    fix = S.cheapest_fix(model, balance, flows, horizon=hz)
    a, b = sim["with_purchase"], sim["baseline"]
    verdict = _verdict(a["p_overdraft"])
    text = (f"Buying {inr(req['amount'])} on {sim['date']}"
            + (f" over {fm} monthly installments" if fm else "")
            + f" moves your overdraft risk over the next {hz} days from "
            f"{b['p_overdraft']*100:.0f}% to {a['p_overdraft']*100:.0f}% "
            f"(median balance in {hz} days {fmt_bal(a['end_p50'])}). Verdict: {verdict}.")
    alt = better_date(best, {"date": sim["date"], **a})
    if alt is not None:
        if alt["p_overdraft"] < a["p_overdraft"]:
            text += (f" Lower-risk date tested: {alt['date']} ({alt['p_overdraft']*100:.0f}% risk vs "
                     f"{a['p_overdraft']*100:.0f}%).")
        else:
            text += (f" Better date tested: {alt['date']} (same risk, median lowest balance "
                     f"{fmt_bal(alt['floor_p50'])} vs {fmt_bal(a['floor_p50'])}).")
    if fix and fix["cut_pct"]:
        text += (f" To make it safe, cut {fix['category'].replace('_', ' ')} by {fix['cut_pct']}% "
                 f"(about {inr(fix['est_monthly_saving'])}/month).")
    evidence = {"sim": sim, "best": best, "fix": fix, "amount": req["amount"], "verdict": verdict}
    out = narrate(llm, SYSTEM, evidence, None)
    llm_text = out.get("narrative") if isinstance(out, dict) else None
    if not verdict_consistent(llm_text, verdict):
        llm_text = None  # e.g. "completely safe" when the computed verdict is unsafe
    narrative = grounded_or(llm_text, evidence, text)
    res = {"verdict": verdict, "sim": sim, "best_dates": best, "fix": fix, "narrative": narrative}
    n_ign = int(sim.get("ignored_flows") or 0)
    if n_ign > 0:  # deterministic, added after grounding so it cannot fail it
        res["notice"] = f"{n_ign} installment(s) fall outside the {hz}-day window and are not simulated"
        res["narrative"] = f"{narrative} ({res['notice']}.)"
    return res


def duel_evidence(res):
    """Numbers a duel narrative may state; negative balances are worded 'overdrawn by ₹X'."""
    sides = [res["a"]["summary"], res["b"]["summary"]]
    over = [-v for sd in sides for k, v in sd.items()
            if k != "p_overdraft" and isinstance(v, (int, float)) and v < 0]
    return {"a": sides[0], "b": sides[1], "h": res["horizon"], "overdrawn_by": over}


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
                                   duel_evidence(res), res["reason"])
    if res.get("ignored_flows"):
        res["notice"] = (f"{res['ignored_flows']} installment(s) fall outside the "
                         f"{res['horizon']}-day window and are not simulated.")
    return res
