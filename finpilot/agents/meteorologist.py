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
_WORD = re.compile(r"\b(clear|cloudy|rain|storm)s?\b", re.I)
_NO_RISK = re.compile(r"\bno (?:\w+ )?risk\b|\ball clear\b|\brisk[- ]free\b", re.I)


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


def scenario_flows(model, scenario, amount=None, horizon=60):
    """Extra (date, amount, label) flows for a stress scenario; pure function of the model."""
    scenario = check_scenario(scenario)
    if scenario == "none":
        return []
    dates = pd.date_range(model.last_date + pd.Timedelta(days=1), periods=int(horizon))
    if scenario == "salary_skip":
        pos = [r for r in model.recurring if r["amount"] > 0]
        if not pos:
            return []
        r = max(pos, key=lambda r: (r["amount"], r["merchant"]))
        for d in dates:
            if d.day == min(r["day"], d.days_in_month):
                return [(str(d.date()), -float(r["amount"]), "salary skipped")]
        return []
    amt = DEFAULT_AMOUNT[scenario] if amount is None else check_amount(amount)
    day = 10 if scenario == "purchase" else 15
    label = "planned purchase" if scenario == "purchase" else "emergency expense"
    return [(str(dates[min(day, len(dates)) - 1].date()), -amt, label)]


def _worst_label(rep):
    return W.label(rep["headline"]["worst_p"])


def label_consistent(text, worst):
    """Weather words in the text may not exceed the worst label, and a storm/rain outlook cannot be
    called clear or risk-free."""
    if not isinstance(text, str):
        return False
    words = {w.lower() for w in _WORD.findall(text)}
    if any(_SEV[w] > _SEV[worst] for w in words):
        return False
    if _SEV[worst] >= _SEV["rain"] and ("clear" in words or _NO_RISK.search(text)):
        return False
    return True


def _evidence(rep):
    return {k: rep[k] for k in ("buffer", "lifestyle", "horizon", "headline", "storms", "autopsy", "umbrellas")}


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
            aut += " Too few failing futures to attribute causes reliably."
        if a["bills"]:
            b = a["bills"][0]
            aut += f" Earliest outflow before then: {b['merchant']} {inr(abs(b['amount']))} on {b['date']}."
    if rep["umbrellas"]:
        u = rep["umbrellas"][0]
        umb = (f"Best umbrella: cut {u['category'].replace('_', ' ')} by {u['cut_pct']}%, which lowers the "
               f"chance of breaching the buffer from {u['p_any_breach_before']*100:.0f}% to "
               f"{u['p_any_breach_after']*100:.0f}% and costs about {inr(u['monthly_sacrifice'])} a month.")
    elif h["p_any_breach"] <= 0:
        umb = "No umbrella needed: the buffer is not breached in any simulated future."
    else:
        umb = "No single spending cut tested reduces the risk."
    return {"headline_text": head, "autopsy_text": aut, "umbrella_text": umb}


def run(model, balance, params, llm=None):
    params = dict(params or {})
    scenario = check_scenario(params.get("scenario"))
    horizon = params.get("horizon", 60)
    flows = scenario_flows(model, scenario, params.get("amount"), horizon)
    rep = W.report(model, balance, horizon=horizon, buffer=params.get("buffer"),
                   lifestyle=params.get("lifestyle", 1.0), cuts=params.get("cuts"), extra_flows=flows)
    rep["scenario"] = scenario
    rep["scenario_flows"] = [{"date": d, "amount": round(float(a), 2), "label": lab} for d, a, lab in flows]
    if scenario != "none" and not flows:
        rep["scenario_note"] = ("no recurring income found, so there is no salary to skip"
                                if scenario == "salary_skip" else "scenario produced no flow")
    texts = fallback_texts(rep)
    out = narrate(llm, SYSTEM, _evidence(rep), None)
    if isinstance(out, dict):
        worst, ev = _worst_label(rep), _evidence(rep)
        for k in texts:
            v = out.get(k)
            if isinstance(v, str) and label_consistent(v, worst):
                texts[k] = grounded_or(v, ev, texts[k])
    rep.update(texts)
    return rep
