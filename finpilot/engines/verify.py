"""Claim recomputation and text auditing.

Independence caveat: the aggregation claims (`month_total`, `pct_change`) reuse the spike engine's
aggregation, and the weather claims (`breach_probability`, `storm_peak`, `storm_peak_after_cut`) re-fit the
model from the raw transactions and re-run the SAME Monte Carlo engine (seed 0, n 1000). They are
consistency checks (the number in the advice is exactly what the engine yields from the raw data and the
opening balance), not an independent implementation.

Weather claim values are percents (0..100). Tolerance: money kinds max(1, 2% of value); `pct_change`
3 points; weather percent kinds 0.5 point absolute. Weather claims need `ctx = {"opening": float}`
(balance = opening + sum of transaction amounts); without it they are unverifiable.
"""
from __future__ import annotations

import math
import re

import pandas as pd

from .analyze import category_month_stats

WEATHER_KINDS = ("breach_probability", "storm_peak", "storm_peak_after_cut")
WEATHER_HORIZON = 60

ALLOWED_CONSTANTS = {12, 24, 30, 60, 90, 100}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _weather_setup(df, ctx):
    """(model, balance, buffer) or None. Buffer is the UNROUNDED default buffer."""
    from . import forecast as F
    from . import weather as W
    if not isinstance(ctx, dict) or not _finite(ctx.get("opening")):
        return None
    model = F.fit(df)
    return model, F.current_balance(df, ctx["opening"]), W.default_buffer(model)


def _recompute_weather(claim, df, ctx):
    from . import forecast as F
    from . import weather as W
    setup = _weather_setup(df, ctx)
    if setup is None:
        return None
    model, bal, buf = setup
    k = claim["kind"]
    if k == "breach_probability":
        h = claim["horizon"]
        if not _finite(h) or h != int(h) or not 1 <= h <= 365:
            return None
        res = F.monte_carlo(model, bal, int(h), 1000, 0)
        return round(float((res.paths < buf).any(axis=1).mean()) * 100, 1)
    date = str(pd.Timestamp(claim["date"]).date())
    cal = W.risk_calendar(model, bal, WEATHER_HORIZON, 1000, 0, buffer=buf)
    storm = next((s for s in W.storms(cal) if s["start"] <= date <= s["end"]), None)
    if storm is None:
        return None
    if k == "storm_peak":
        return round(storm["peak_p"] * 100, 1)
    cut = claim["cut_pct"]
    if not _finite(cut) or not 0 <= cut <= 100 or claim["category"] not in model.cats:
        return None
    cal2 = W.risk_calendar(model, bal, WEATHER_HORIZON, 1000, 0, buffer=buf,
                           cuts={claim["category"]: cut / 100.0})
    return round(max(c["p_below_buffer"] for c in cal2 if storm["start"] <= c["date"] <= storm["end"]) * 100, 1)


def _recompute(claim, df, ctx=None):
    k = claim["kind"]
    if k in WEATHER_KINDS:
        return _recompute_weather(claim, df, ctx)
    if k in ("monthly_cost", "annual_cost"):
        g = df[(df["merchant"] == claim["merchant"]) & (df["amount"] < 0)].sort_values("date").tail(3)
        if g.empty:
            return None
        v = float(abs(g["amount"].mean()))
        return v * 12 if k == "annual_cost" else v
    if k in ("month_total", "pct_change"):
        # same aggregation as the spike engine (flagged large charges excluded, zero months counted)
        st = category_month_stats(df, claim["category"], str(claim["month"]))
        return None if st is None else st[k]
    if k == "txn_amount":
        try:
            return float(abs(df.loc[int(claim["txn_id"]), "amount"]))
        except (KeyError, ValueError):
            return None
    return None


_NEEDS = {"monthly_cost": ("merchant",), "annual_cost": ("merchant",), "month_total": ("category", "month"),
          "pct_change": ("category", "month"), "txn_amount": ("txn_id",),
          "breach_probability": ("horizon",), "storm_peak": ("date",),
          "storm_peak_after_cut": ("date", "category", "cut_pct")}


def _well_formed(claim):
    if not isinstance(claim, dict) or claim.get("kind") not in _NEEDS:
        return False
    return all(claim.get(k) is not None for k in _NEEDS[claim["kind"]])


def recompute(claim, df, ctx=None):
    """Recompute a claim from transactions; None for anything malformed or unrecomputable."""
    if not _well_formed(claim):
        return None
    try:
        v = _recompute(claim, df, ctx)
        if v is None or isinstance(v, (bool, complex)) or not math.isfinite(float(v)):
            return None
        return float(v)
    except Exception:
        return None


def _numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_claim(claim, df, ctx=None):
    actual = recompute(claim, df, ctx)
    if actual is None or not _numeric(claim.get("value")):
        return {"ok": False, "actual": actual if actual is not None and _well_formed(claim) else None}
    if claim["kind"] == "pct_change":
        tol = 3.0
    elif claim["kind"] in WEATHER_KINDS:
        tol = 0.5
    else:
        tol = max(1.0, 0.02 * abs(actual))
    return {"ok": abs(float(claim["value"]) - actual) <= tol, "actual": actual}


def _flatten_numbers(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten_numbers(v)


_SUFFIX = {"k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
           "crore": 1e7, "crores": 1e7, "cr": 1e7}
_TOKEN = re.compile(r"(?P<cur>₹|\$|(?<![A-Za-z])(?:Rs\.?|INR|USD))?\s*(?P<num>\d[\d,]*\.?\d*)\s*"
                    r"(?P<suf>(?:lakhs?|lacs?|crores?|cr|k)\b)?(?P<pct>\s*(?:%|(?:percentage\s+points?|per\s*cent|percent|pct)\b))?", re.I)
_DATE = re.compile(r"(?<!\d)(19|20)\d{2}-\d{2}(-\d{2})?(?!\d)")


def text_numbers(text):
    """Numbers stated in text, ISO dates removed. Yields dicts {value, currency, suffix, percent, token}."""
    text = _DATE.sub(" ", str(text))  # ISO dates and months are not claims
    for m in _TOKEN.finditer(text):
        tok = m.group("num").rstrip(".,")
        raw = tok.replace(",", "")
        if not raw:
            continue
        suf = m.group("suf")
        try:
            n = float(raw) * (_SUFFIX[suf.lower()] if suf else 1)
        except ValueError:
            continue
        yield {"value": n, "currency": bool(m.group("cur")), "suffix": bool(suf),
               "percent": bool(m.group("pct")), "token": tok}


def matches(n, a):
    """Shared tolerance: within 1% of a or within 1 in absolute terms, whichever is larger."""
    return abs(n - a) <= max(1.0, 0.01 * abs(a))


_PCT_KEY = re.compile(r"pct|percent|share|rate", re.I)


def _keyed_numbers(obj, key=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield key, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _keyed_numbers(v, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _keyed_numbers(v, key)


def ungrounded_numbers(text, claims, evidence):
    """Numbers in text that match no claim value or evidence number. Exempt (not checked): ISO dates,
    bare 4-digit years 1990-2100, and BARE numbers under 10 or in ALLOWED_CONSTANTS. A number followed
    by '%' (or 'percent', 'per cent', 'pct', 'percentage points') or preceded by a currency token (₹, Rs, INR, $, USD) is always checked, and a percentage
    must match a percent-like value: a probability/fraction in [0, 1] times 100, a pct_change claim, or
    an evidence value whose key names a percentage (pct, percent, share, rate)."""
    vals = [c.get("value") for c in claims if isinstance(c, dict)]
    allowed = list(_flatten_numbers(vals)) + list(_flatten_numbers(evidence))
    allowed += [a * 100 for a in allowed if 0 < abs(a) <= 1]  # probabilities quoted as percents
    pct_ok = [a * 100 for a in allowed if abs(a) <= 1]
    pct_ok += [float(c["value"]) for c in claims if isinstance(c, dict) and c.get("kind") in ("pct_change", *WEATHER_KINDS)
               and _numeric(c.get("value"))]
    pct_ok += [v for k, v in _keyed_numbers(evidence) if _PCT_KEY.search(k)]
    out = []
    for t in text_numbers(text):
        n = t["value"]
        if t["percent"]:
            if not any(matches(n, a) for a in pct_ok):
                out.append(n)
            continue
        strict = t["currency"] or t["percent"]
        if not strict and not t["suffix"]:
            if re.fullmatch(r"\d{4}", t["token"]) and 1990 <= n <= 2100:
                continue
            if n in ALLOWED_CONSTANTS:
                continue
        if n < 10 and not strict:
            continue
        if not any(matches(n, a) for a in allowed):
            out.append(n)
    return out


def claims_missing_from_text(text, claims):
    """Claims whose value is not stated in the text (same tolerance; a pct_change is matched as the
    number before '%'). Catches advice that omits or contradicts its own verified claims."""
    nums = [t["value"] for t in text_numbers(text)]
    out = []
    for c in claims:
        if not isinstance(c, dict) or not _numeric(c.get("value")):
            continue
        v = abs(float(c["value"]))
        if not any(matches(n, v) for n in nums):
            out.append(c)
    return out


def weather_text_issues(text, claims):
    """Role binding for weather percent claims. Each must be stated as a percent token (a bare number does
    not count), and when both storm_peak and storm_peak_after_cut are claimed (and differ) the text must state
    the peak before the after-cut value, so swapped numbers cannot pass. Returns [(type, claim)]."""
    toks = [t["value"] for t in text_numbers(text) if t["percent"]]
    first, out = {}, []
    for c in claims:
        if not isinstance(c, dict) or c.get("kind") not in WEATHER_KINDS or not _numeric(c.get("value")):
            continue
        v = abs(float(c["value"]))
        idx = next((i for i, n in enumerate(toks) if matches(n, v)), None)
        if idx is None:
            out.append(("claim_not_percent", c))
        else:
            first[c["kind"]] = (idx, v)
    if "storm_peak" in first and "storm_peak_after_cut" in first:
        (ip, vp), (ia, va) = first["storm_peak"], first["storm_peak_after_cut"]
        if not matches(vp, va) and ia < ip:
            out.append(("claim_order", next(c for c in claims if isinstance(c, dict)
                                            and c.get("kind") == "storm_peak_after_cut")))
    return out
