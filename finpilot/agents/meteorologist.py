"""Meteorologist: narrates a weather.report(). Numbers come only from the engine; LLM prose is accepted
per field through grounded_or plus a label-consistency check, else a deterministic sentence is used."""
from __future__ import annotations

import math
import re

import pandas as pd

from ..engines import weather as W
from .common import grounded_or, inr, narrate, pct_text, plural, pct_values, risk_claim_consistent, risk_level

SYSTEM = ("You are a financial meteorologist. Using only the numbers in the given weather report, write "
          "three short plain sentences. Return JSON {\"headline_text\": str, \"autopsy_text\": str, "
          "\"umbrella_text\": str}. Do not invent numbers.")

SCENARIOS = ("none", "salary_skip", "purchase", "emergency")
DEFAULT_AMOUNT = {"purchase": 80000.0, "emergency": 40000.0}
MAX_AMOUNT = 1e9
_SEV = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3}
_STEM = re.compile(r"\b(storm\w*|rain\w*|cloud\w*|sunny|sunshine|sunlit|wet)\b", re.I)
_CLEAR = re.compile(r"\bclear\s+(?:skies|sky|weather|outlook|forecast|day|days)\b|\ball clear\b|"
                    r"\bskies\s+(?:look|are|stay|will be)\s+(?:fine|good|ok(?:ay)?|calm)\b", re.I)
_NO_RISK = re.compile(r"\bno (?:\w+ )?risk\b|\brisk[- ]free\b|\bfine\b|\bsafe\b|\bno breach\w*|"
                      r"\bnothing to worry\b|\blooks? fine\b|\ball good\b", re.I)


def check_amount(x):
    if isinstance(x, bool):
        raise ValueError("amount must be a number")
    try:
        a = float(x)
    except (TypeError, ValueError):
        raise ValueError("amount must be a number") from None
    if not (math.isfinite(a) and 0 < a <= MAX_AMOUNT):
        raise ValueError(f"amount must be greater than 0 and at most {MAX_AMOUNT:,.0f}")
    return a


def check_scenario(s):
    if s is None:
        return "none"
    if s not in SCENARIOS:
        raise ValueError(f"scenario must be one of {', '.join(SCENARIOS)}")
    return s


def default_amount(scenario, balance):
    """purchase: 95% of the balance, emergency: 70%, rounded to the nearest 1,000; 10,000 if balance <= 0."""
    if balance is None or not math.isfinite(float(balance)) or balance <= 0:
        return 10000.0
    frac = 0.95 if scenario == "purchase" else 0.70
    return float(max(1000, round(frac * float(balance) / 1000.0) * 1000))


def scenario_flows(model, scenario, amount=None, horizon=60, balance=None):
    """Extra (date, amount, label) flows for a stress scenario; pure function of the model (and balance
    for the default amounts)."""
    scenario = check_scenario(scenario)
    if scenario == "none":
        return []
    dates = pd.date_range(model.last_date + pd.Timedelta(days=1), periods=int(horizon))
    if scenario == "salary_skip":
        pos = [r for r in model.recurring if r["amount"] > 0]
        if not pos:
            return []
        r = max(pos, key=lambda r: (r["amount"], r["merchant"]))
        return [(str(d.date()), -float(r["amount"]), "salary skipped") for d in dates
                if d.day == min(r["day"], d.days_in_month)]
    amt = default_amount(scenario, balance) if amount is None else check_amount(amount)
    day = 10 if scenario == "purchase" else 15
    label = "planned purchase" if scenario == "purchase" else "emergency expense"
    return [(str(dates[min(day, len(dates)) - 1].date()), -amt, label)]


def _worst_label(rep):
    return W.label(rep["headline"]["worst_p"])


def _sev(word):
    w = word.lower()
    return 3 if w.startswith("storm") else 2 if w.startswith("rain") or w == "wet" else \
        1 if w.startswith("cloud") else 0


def label_consistent(text, worst):
    """Best-effort wording filter: weather words (storm*, rain*, cloud*, sunny, wet, 'clear skies') may not
    exceed the worst computed label, and a storm/rain outlook may not be called clear, fine, safe, risk-free
    or breach-free. 'clear' as a verb is not a weather word."""
    if not isinstance(text, str):
        return False
    sev = [_sev(w) for w in _STEM.findall(text)]
    if any(v > _SEV[worst] for v in sev):
        return False
    if _SEV[worst] >= _SEV["rain"] and (_CLEAR.search(text) or _NO_RISK.search(text)
                                        or any(v == 0 for v in sev)):
        return False
    return True


def strict_evidence(rep):
    """Only the numbers the narration may state (explicit *_pct values for percentages, money amounts)."""
    h, a = rep["headline"], rep["autopsy"]
    pc = lambda x: round(float(x) * 100, 1)  # noqa: E731
    return {"buffer": rep["buffer"], "horizon": rep["horizon"], "p_any_breach_pct": pc(h["p_any_breach"]),
            "worst_p_pct": pc(h["worst_p"]),
            "shown_pct": pct_values(h["p_any_breach"])[1:] + pct_values(h["worst_p"])[1:],
            "n_storm_days": h["n_storm_days"], "n_storms": len(rep["storms"]),
            "storms": [{"peak_pct": pc(x["peak_p"]), "days": x["days"]} for x in rep["storms"]],
            "umbrellas": [{"cut_pct": u["cut_pct"], "before_pct": pc(u["p_any_breach_before"]),
                           "after_pct": pc(u["p_any_breach_after"]), "reduction_pct": pc(u["risk_reduction"]),
                           "monthly_sacrifice": u["monthly_sacrifice"]} for u in rep["umbrellas"]],
            "autopsy": {"n_failing": a["n_failing"], "n_paths": a["n_paths"],
                        "causes": [{"excess_per_path": c["excess_per_path"], "share_pct": pc(c["share_of_gap"])}
                                   for c in a["causes"]],
                        "bills": [abs(b["amount"]) for b in a["bills"]]},
            "scenario_amount": rep.get("scenario_amount"),
            "scenario_flow_amounts": [abs(f["amount"]) for f in rep.get("scenario_flows") or []],
            "day_of_month": sorted({pd.Timestamp(d).day for d in _text_dates(rep)})}


def _text_dates(rep):
    """Dates the narration may write as 'Oct 10' (their day numbers become allowed bare numbers)."""
    a = rep["autopsy"]
    ds = [f["date"] for f in rep.get("scenario_flows") or []] + [b["date"] for b in a["bills"]]
    ds += [a["first_breach"]["date"]] if a["first_breach"].get("date") else []
    ds += [rep["headline"]["worst_day"]] + [x for st in rep["storms"] for x in (st["start"], st["end"], st["peak_date"])]
    return [d for d in ds if d]


def _md(iso):
    try:
        t = pd.Timestamp(iso)
    except (ValueError, TypeError):
        return str(iso)
    return f"{t.strftime('%b')} {t.day}"


def _join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def autopsy_text(rep):
    """Deterministic autopsy sentence. With a scenario, it says first that every simulated future contains the
    scenario flow (a purchase or emergency expense, or a missing salary, which is never called an outflow),
    then what separates the failing futures. Names the LARGEST recurring bill before the first dip."""
    a, sc = rep["autopsy"], rep.get("scenario")
    flows = rep.get("scenario_flows") or []
    n, N = a["n_failing"], a["n_paths"]
    causes = a["causes"]
    share = lambda c: f"{c['share_of_gap']*100:.0f}% of the gap"  # noqa: E731
    cat = lambda c: c["category"].replace("_", " ")  # noqa: E731
    cause_said = False
    if sc == "salary_skip" and flows:
        aut = (f"Salary of {inr(abs(flows[0]['amount']))} does not arrive on "
               f"{_join([_md(f['date']) for f in flows])}")
        if n == 0:
            return aut + "; even so, none of the simulated futures dips below the buffer."
        aut += f"; {n} of {N} simulated futures then dip below the buffer."
    elif sc in ("purchase", "emergency") and flows:
        f = flows[0]
        what = (f"planned {inr(abs(f['amount']))} purchase" if sc == "purchase"
                else f"{inr(abs(f['amount']))} emergency expense")
        aut = f"Every simulated future includes the {what} on {_md(f['date'])}"
        if n == 0:
            return aut + "; none of them dips below the buffer."
        if causes:
            aut += f"; among them, the failing ones also spent more on {cat(causes[0])} ({share(causes[0])})."
            cause_said = True
        elif n >= N:
            aut += "; every one of them dips below the buffer, so there is no non-failing group to compare."
            cause_said = True
        else:
            aut += f"; {n} of the {N} dip below the buffer."
    else:
        if n == 0:
            return "No failing futures, so there is nothing to explain."
        aut = f"{n} of {N} simulated futures fail."
    fb = a["first_breach"]
    if fb.get("date"):
        aut += f" Most first dip below the buffer around {_md(fb['date'])}."
    if not cause_said:
        if causes:
            c = causes[0]
            aut += (f" The biggest driver is {cat(c)}: {inr(c['excess_per_path'])} more spend than in the "
                    f"non-failing futures ({share(c)}).")
        elif a["insufficient_failing"]:
            aut += f" Too few failing futures ({n}) to attribute causes reliably."
        elif n >= N:
            aut += " Every simulated future falls below the buffer, so there is no non-failing group to compare."
    bills = [b for b in a["bills"] if not str(b["merchant"]).lower().startswith("planned")]
    if bills:
        b = max(bills, key=lambda b: (abs(b["amount"]), [-ord(ch) for ch in b["date"]]))
        aut += f" Largest bill before then: {b['merchant']} {inr(abs(b['amount']))} on {_md(b['date'])}."
    return aut


def paydays(model, horizon=60, skip=()):
    """ISO dates of recurring income (positive recurring flows) inside the horizon, minus skipped dates."""
    dates = pd.date_range(model.last_date + pd.Timedelta(days=1), periods=int(horizon))
    skip = set(skip)
    out = {str(d.date()) for r in model.recurring if r["amount"] > 0 for d in dates
           if d.day == min(r["day"], d.days_in_month)}
    return sorted(out - skip)


def llm_payload(rep):
    """What the narrating model sees: the strict numbers plus labels from fixed vocabularies (worst label,
    scenario, category names). Never merchant names or other free text from the statement."""
    return {**strict_evidence(rep), "worst_label": _worst_label(rep), "scenario": rep.get("scenario"),
            "cause_categories": [c["category"] for c in rep["autopsy"]["causes"]],
            "umbrella_categories": [u["category"] for u in rep["umbrellas"]]}


def fallback_texts(rep):
    h, a = rep["headline"], rep["autopsy"]
    worst = _worst_label(rep)
    if h["p_any_breach"] <= 0:
        head = (f"Clear skies: none of the simulated futures dips below your {inr(rep['buffer'])} safety buffer "
                f"in the next {rep['horizon']} days.")
    else:
        head = (f"{pct_text(h['p_any_breach'])} of simulated futures dip below your {inr(rep['buffer'])} safety "
                f"buffer within {rep['horizon']} days. The riskiest day is {_md(h['worst_day'])}, with a "
                f"{pct_text(h['worst_p'])} chance ({worst}).")
        n_s, n_w = h["n_storm_days"], len(rep["storms"])
        head += (f" Storm level is reached on {plural(n_s, 'day')}, in {plural(n_w, 'separate stretch', 'separate stretches')}."
                 if rep["storms"] else " No day reaches storm level.")
    aut = autopsy_text(rep)
    if rep["umbrellas"]:
        u = rep["umbrellas"][0]
        b4, af = u["p_any_breach_before"] * 100, u["p_any_breach_after"] * 100
        fmt = ".1f" if f"{b4:.0f}" == f"{af:.0f}" else ".0f"
        umb = (f"Best umbrella: cut {u['category'].replace('_', ' ')} by {u['cut_pct']}%, which lowers the "
               f"chance of breaching the buffer from {b4:{fmt}}% to {af:{fmt}}% and costs about "
               f"{inr(u['monthly_sacrifice'])} a month.")
        if b4 - af < 1:
            umb += " That is only a marginal effect."
    elif h["p_any_breach"] <= 0:
        umb = "No umbrella needed: the buffer is not breached in any simulated future."
    else:
        umb = "No single spending cut tested reduces the risk."
    return {"headline_text": head, "autopsy_text": aut, "umbrella_text": umb}


def run(model, balance, params, llm=None):
    params = dict(params or {})
    scenario = check_scenario(params.get("scenario"))
    horizon = params.get("horizon", 60)
    flows = scenario_flows(model, scenario, params.get("amount"), horizon, balance)
    rep = W.report(model, balance, horizon=horizon, buffer=params.get("buffer"),
                   lifestyle=params.get("lifestyle", 1.0), cuts=params.get("cuts"), extra_flows=flows)
    rep["scenario"] = scenario
    rep["scenario_flows"] = [{"date": d, "amount": round(float(a), 2), "label": lab} for d, a, lab in flows]
    rep["scenario_amount"] = (abs(flows[0][1]) if flows and scenario in ("purchase", "emergency") else None)
    if scenario == "salary_skip":  # a missing salary is not an outflow: keep it out of the bills list
        rep["autopsy"]["bills"] = [b for b in rep["autopsy"]["bills"]
                                   if not str(b["merchant"]).startswith("planned: salary")]
    rep["paydays"] = paydays(model, horizon, skip=[d for d, _, _ in flows] if scenario == "salary_skip" else ())
    if scenario == "salary_skip":
        rep["scenario_note"] = (f"{plural(len(flows), 'pay date')} removed inside the horizon" if flows else
                                "no recurring income found, so there is no salary to skip")
    elif scenario != "none" and not flows:
        rep["scenario_note"] = "scenario produced no flow"
    rep.update(narrate_report(rep, llm))
    return rep


def narrate_report(rep, llm=None):
    """Deterministic texts, replaced per field by LLM prose only if it is label- and risk-consistent and
    strictly grounded in strict_evidence(rep)."""
    texts = fallback_texts(rep)
    out = narrate(llm, SYSTEM, llm_payload(rep), None)
    if isinstance(out, dict):
        worst, ev = _worst_label(rep), strict_evidence(rep)
        p = rep["headline"]["p_any_breach"]
        for k in texts:
            v = out.get(k)
            if (isinstance(v, str) and label_consistent(v, worst)
                    and risk_claim_consistent(v, risk_level(p), [p])):
                texts[k] = grounded_or(v, ev, texts[k], strict=True)
    return texts
