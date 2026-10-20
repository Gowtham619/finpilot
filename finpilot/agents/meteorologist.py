"""Meteorologist: narrates a weather.report(). Numbers come only from the engine; LLM prose is accepted
per field through grounded_or plus a label-consistency check, else a deterministic sentence is used."""
from __future__ import annotations

import math
import re

import pandas as pd

from ..engines import weather as W
from .common import grounded_or, inr, narrate

SYSTEM = ("You are a financial meteorologist. Using only the numbers in the given weather report, write "
          "three short plain sentences. Return JSON {\"headline_text\": str, \"autopsy_text\": str, "
          "\"umbrella_text\": str}. Do not invent numbers.")

SCENARIOS = ("none", "salary_skip", "purchase", "emergency")
DEFAULT_AMOUNT = {"purchase": 80000.0, "emergency": 40000.0}
MAX_AMOUNT = 1e9
_SEV = {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3}
_STEM = re.compile(r"\b(storm\w*|rain\w*|cloud\w*|sunny|sunshine|sunlit|wet)\b", re.I)
_CLEAR = re.compile(r"\bclear\s+(?:skies|sky|weather|outlook|forecast|day|days)\b|\ball clear\b", re.I)
_NO_RISK = re.compile(r"\bno (?:\w+ )?risk\b|\brisk[- ]free\b", re.I)


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
    """Weather words (storm*, rain*, cloud*, sunny, wet, 'clear skies') may not exceed the worst label, and
    a storm/rain outlook cannot be called clear or risk-free. 'clear' as a verb is not a weather word."""
    if not isinstance(text, str):
        return False
    sev = [_sev(w) for w in _STEM.findall(text)]
    if any(v > _SEV[worst] for v in sev):
        return False
    if _SEV[worst] >= _SEV["rain"] and (_CLEAR.search(text) or _NO_RISK.search(text)
                                        or any(v == 0 for v in sev)):
        return False
    return True


def _evidence(rep):
    return {k: rep[k] for k in ("buffer", "lifestyle", "horizon", "headline", "storms", "autopsy", "umbrellas")}


def strict_evidence(rep):
    """Only the numbers the narration may state (explicit *_pct values for percentages, money amounts)."""
    h, a = rep["headline"], rep["autopsy"]
    pc = lambda x: round(float(x) * 100, 1)  # noqa: E731
    return {"buffer": rep["buffer"], "horizon": rep["horizon"], "p_any_breach_pct": pc(h["p_any_breach"]),
            "worst_p_pct": pc(h["worst_p"]), "n_storm_days": h["n_storm_days"], "n_storms": len(rep["storms"]),
            "storms": [{"peak_pct": pc(x["peak_p"]), "days": x["days"]} for x in rep["storms"]],
            "umbrellas": [{"cut_pct": u["cut_pct"], "before_pct": pc(u["p_any_breach_before"]),
                           "after_pct": pc(u["p_any_breach_after"]), "reduction_pct": pc(u["risk_reduction"]),
                           "monthly_sacrifice": u["monthly_sacrifice"]} for u in rep["umbrellas"]],
            "autopsy": {"n_failing": a["n_failing"], "n_paths": a["n_paths"],
                        "causes": [{"excess_per_path": c["excess_per_path"], "share_pct": pc(c["share_of_gap"])}
                                   for c in a["causes"]],
                        "bills": [abs(b["amount"]) for b in a["bills"]]},
            "scenario_amount": rep.get("scenario_amount")}


def fallback_texts(rep):
    h, a = rep["headline"], rep["autopsy"]
    worst = _worst_label(rep)
    if h["p_any_breach"] <= 0:
        head = (f"Clear skies: none of the simulated futures dips below your {inr(rep['buffer'])} safety buffer "
                f"in the next {rep['horizon']} days.")
    else:
        head = (f"{h['p_any_breach']*100:.0f}% of simulated futures dip below your {inr(rep['buffer'])} safety "
                f"buffer within {rep['horizon']} days. Worst day {h['worst_day']} ({worst}, "
                f"{h['worst_p']*100:.0f}% chance)")
        head += (f"; {h['n_storm_days']} storm day(s) in {len(rep['storms'])} storm(s)." if rep["storms"]
                 else "; no storm days.")
    if a["n_failing"] == 0:
        aut = "No failing futures, so there is nothing to explain."
    else:
        aut = f"{a['n_failing']} of {a['n_paths']} simulated futures fail."
        fb = a["first_breach"]
        if fb["date"]:
            aut += f" Most first dip below the buffer around {fb['date']}."
        if a["causes"]:
            c = a["causes"][0]
            aut += (f" The biggest driver is {c['category'].replace('_', ' ')}: {inr(c['excess_per_path'])} "
                    f"more spend than in the safe futures ({c['share_of_gap']*100:.0f}% of the gap).")
        elif a["insufficient_failing"]:
            aut += f" Too few failing futures ({a['n_failing']}) to attribute causes reliably."
        elif a["n_failing"] >= a["n_paths"]:
            aut += (" Every simulated future falls below the buffer, so there is no non-failing group to compare.")
        if a["bills"]:
            b = a["bills"][0]
            aut += f" Earliest outflow before then: {b['merchant']} {inr(abs(b['amount']))} on {b['date']}."
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
    if scenario == "salary_skip":
        rep["scenario_note"] = (f"{len(flows)} pay date(s) removed inside the horizon" if flows else
                                "no recurring income found, so there is no salary to skip")
    elif scenario != "none" and not flows:
        rep["scenario_note"] = "scenario produced no flow"
    texts = fallback_texts(rep)
    out = narrate(llm, SYSTEM, _evidence(rep), None)
    if isinstance(out, dict):
        worst, ev = _worst_label(rep), strict_evidence(rep)
        for k in texts:
            v = out.get(k)
            if isinstance(v, str) and label_consistent(v, worst):
                texts[k] = grounded_or(v, ev, texts[k], strict=True)
    rep.update(texts)
    return rep
