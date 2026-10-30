import re

import pandas as pd

from ..engines import simulate as S
from .common import (plural, fmt_bal, grounded_or, inr, narrate, pct_text, pct_values, risk_claim_consistent,
                     risk_level)

SYSTEM = ("You answer 'can I afford this?'. Use only the numbers given, in two or three sentences. "
          "Return JSON {\"narrative\": str}.")


def _verdict(p):
    return "safe" if p <= 0.05 else "risky" if p <= 0.25 else "unsafe"


_VERDICT_WORD = re.compile(r"\b(safe|risky|unsafe)\b", re.I)
SAFE_P = 0.05  # the "safe" verdict threshold; a suggested cut must reach it
FLOOR_MARGIN = 1000.0  # an equal-risk date must raise the median lowest balance by at least this much


def verdict_consistent(text, verdict):
    """False when the text uses a verdict word (safe/risky/unsafe) other than the computed one."""
    return {w.lower() for w in _VERDICT_WORD.findall(str(text))} <= {verdict}


def afford_horizon(financed_months):
    """Long enough to cover the installments of the latest candidate date (+56 days), capped at a year."""
    return min(365, max(60, 31 * financed_months + 31 + 56)) if financed_months else 60


def better_date(best, chosen):
    """The best tested date if it is really better than the chosen one, else None."""
    if not best or best[0]["date"] == chosen["date"]:
        return None
    b = best[0]
    if int(b.get("ignored_flows") or 0) > int(chosen.get("ignored_flows") or 0):
        return None  # looks safer only because installments fall outside the window
    if b["p_overdraft"] < chosen["p_overdraft"]:
        return b
    if (b["p_overdraft"] == chosen["p_overdraft"] and b["floor_p50"] >= 0
            and b["floor_p50"] >= chosen["floor_p50"] + FLOOR_MARGIN):
        return b
    return None


def _side_numbers(sm):
    """Strict evidence for one simulated summary: unsigned money amounts and explicit percents."""
    return {"money": [abs(sm[k]) for k in ("end_p10", "end_p50", "end_p90", "floor_p50", "avg_p50") if k in sm],
            "p_overdraft_pct": pct_values(sm["p_overdraft"])}


def afford_strict_evidence(sim, best, fix, amount, financed_months=0):
    """Only the numbers an affordability narration may state (no counts or flags that could ground a
    stray percent)."""
    ev = {"amount": abs(float(amount)), "horizon": sim["horizon"], "financed_months": financed_months,
          "baseline": _side_numbers(sim["baseline"]), "with_purchase": _side_numbers(sim["with_purchase"]),
          "best": [_side_numbers(b) for b in best or []], "target_pct": pct_values(SAFE_P)}
    if financed_months:
        ev["installment"] = abs(float(amount)) / financed_months
    if fix:
        ev["fix"] = {"p_overdraft_pct": pct_values(fix["p_overdraft"]), "cut_pct": fix["cut_pct"],
                     "monthly_saving": fix["est_monthly_saving"]}
    return ev


def afford(model, balance, req, llm=None):
    """Raises ValueError if req["date"] is outside the forecast window; the API layer maps it to HTTP 400."""
    fm = int(req.get("financed_months") or 0)
    hz = afford_horizon(fm)
    sim = S.simulate_purchase(model, balance, req["amount"], req["date"], horizon=hz, financed_months=fm)
    flows = S.purchase_flows(req["amount"], req["date"], fm)
    start = pd.Timestamp(req["date"])
    cands = [start + pd.Timedelta(days=7 * k) for k in range(9)]
    best = S.best_date(model, balance, req["amount"], cands, horizon=hz, financed_months=fm,
                       max_ignored=int(sim.get("ignored_flows") or 0))[:3]
    fix = S.cheapest_fix(model, balance, flows, target_p=SAFE_P, horizon=hz)
    a, b = sim["with_purchase"], sim["baseline"]
    verdict = _verdict(a["p_overdraft"])
    text = (f"Buying {inr(req['amount'])} on {sim['date']}"
            + (f" over {fm} monthly installments" if fm else "")
            + f" moves your overdraft risk over the next {hz} days from "
            f"{pct_text(b['p_overdraft'])} to {pct_text(a['p_overdraft'])}; in {hz} days the median balance "
            f"is {fmt_bal(a['end_p50'])}. Verdict: {verdict}.")
    alt = better_date(best, {"date": sim["date"], **a, "ignored_flows": sim.get("ignored_flows", 0)})
    if alt is not None:
        if alt["p_overdraft"] < a["p_overdraft"]:
            text += (f" Lower-risk date tested: {alt['date']} ({pct_text(alt['p_overdraft'])} risk vs "
                     f"{pct_text(a['p_overdraft'])}).")
        else:
            text += (f" Better date tested: {alt['date']} (same risk, median lowest balance "
                     f"{fmt_bal(alt['floor_p50'])} vs {fmt_bal(a['floor_p50'])}).")
    if fix and fix["cut_pct"]:
        text += (f" To bring overdraft risk down to about {pct_text(fix['p_overdraft'])}, cut "
                 f"{fix['category'].replace('_', ' ')} by {fix['cut_pct']}% "
                 f"(about {inr(fix['est_monthly_saving'])}/month).")
    elif fix is None and a["p_overdraft"] > SAFE_P:
        text += f" No single cut tested brings the risk under {SAFE_P*100:.0f}%."
    evidence = {"sim": sim, "best": best, "fix": fix, "amount": req["amount"], "verdict": verdict,
                "target_p": SAFE_P}
    out = narrate(llm, SYSTEM, evidence, None)
    llm_text = out.get("narrative") if isinstance(out, dict) else None
    if not verdict_consistent(llm_text, verdict):
        llm_text = None  # e.g. "completely safe" when the computed verdict is unsafe
    if not risk_claim_consistent(llm_text, risk_level(a["p_overdraft"]), [a["p_overdraft"]]):
        llm_text = None  # e.g. "go ahead, this will not hurt you" when the computed risk is real
    narrative = grounded_or(llm_text, afford_strict_evidence(sim, best, fix, req["amount"], fm), text, strict=True)
    res = {"verdict": verdict, "sim": sim, "best_dates": best, "fix": fix, "narrative": narrative}
    n_ign = int(sim.get("ignored_flows") or 0)
    if n_ign > 0:  # deterministic, added after grounding so it cannot fail it
        res["notice"] = f"{plural(n_ign, 'installment')} fall{'s' if n_ign == 1 else ''} outside the {hz}-day window and {'is' if n_ign == 1 else 'are'} not simulated"
        res["narrative"] = f"{narrative} ({res['notice']}.)"
    return res


def duel_evidence(res):
    """Numbers a duel narration may state (strict): unsigned money amounts per side, explicit percents and the
    horizon. Negative balances are worded 'overdrawn by ₹X', so their absolute values are listed."""
    return {"a": _side_numbers(res["a"]["summary"]), "b": _side_numbers(res["b"]["summary"]),
            "horizon": res["horizon"]}


_WIN_WORDS = r"(?:wins?|winner|winning|better|safer|best|preferable|smarter|wiser|stronger|comes out ahead|is ahead)"


def duel_text_consistent(text, res):
    """The narration must agree with the computed winner: it names the winner's option, never pairs the
    loser's name with win/better/safer/best wording, and in a tie declares no winner."""
    if not isinstance(text, str):
        return False
    low = text.lower()
    na, nb = res["a"]["name"].strip().lower(), res["b"]["name"].strip().lower()
    if res["winner"] == "tie":
        return not re.search(r"\b" + _WIN_WORDS + r"\b|\bprefer\w*|\bbeats?\b|\boutperform\w*", low)
    win, lose = (na, nb) if res["winner"] == "a" else (nb, na)
    if not win or win not in low:
        return False
    if lose and lose not in win:
        L = re.escape(lose)
        if re.search(L + r"\W+(?:\w+\W+){0,3}?" + _WIN_WORDS + r"\b", low):
            return False
        if re.search(r"\b(?:winner|choose|pick|prefer|go with|recommend\w*|opt for)\W+(?:\w+\W+){0,1}?" + L, low):
            return False
    return True


def run_duel(model, balance, req, llm=None):
    def opt(o):
        return {"name": o["name"],
                "flows": S.purchase_flows(o.get("amount") or 0, o["date"], int(o.get("financed_months") or 0))
                if o.get("amount") else [],
                "cuts": o.get("cuts") or {}}
    res = S.duel(model, balance, opt(req["a"]), opt(req["b"]), horizon=int(req.get("horizon", 90)))
    out = narrate(llm, SYSTEM, {"duel": {k: res[k] for k in ("winner", "reason")},
                                "a": res["a"]["summary"], "b": res["b"]["summary"]}, None)
    text = out.get("narrative") if isinstance(out, dict) else None
    ps = [res["a"]["summary"]["p_overdraft"], res["b"]["summary"]["p_overdraft"]]
    if not duel_text_consistent(text, res) or not risk_claim_consistent(text, risk_level(max(ps)), ps):
        text = None  # e.g. "Buy now wins clearly" when the computed winner is the other option
    res["narrative"] = grounded_or(text, duel_evidence(res), res["reason"], strict=True,
                                   ignore=(res["a"]["name"], res["b"]["name"]))
    if res.get("ignored_flows"):
        n = res["ignored_flows"]
        res["notice"] = (f"{plural(n, 'installment')} fall{'s' if n == 1 else ''} outside the "
                         f"{res['horizon']}-day window and {'is' if n == 1 else 'are'} not simulated.")
    return res
