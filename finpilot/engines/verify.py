from __future__ import annotations

import math
import re

import pandas as pd

from .analyze import category_month_stats

ALLOWED_CONSTANTS = {12, 24, 30, 60, 90, 100}


def _recompute(claim, df):
    k = claim["kind"]
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
          "pct_change": ("category", "month"), "txn_amount": ("txn_id",)}


def _well_formed(claim):
    if not isinstance(claim, dict) or claim.get("kind") not in _NEEDS:
        return False
    return all(claim.get(k) is not None for k in _NEEDS[claim["kind"]])


def recompute(claim, df):
    """Recompute a claim from transactions; None for anything malformed or unrecomputable."""
    if not _well_formed(claim):
        return None
    try:
        v = _recompute(claim, df)
        if v is None or isinstance(v, (bool, complex)) or not math.isfinite(float(v)):
            return None
        return float(v)
    except Exception:
        return None


def _numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_claim(claim, df):
    actual = recompute(claim, df)
    if actual is None or not _numeric(claim.get("value")):
        return {"ok": False, "actual": actual if actual is not None and _well_formed(claim) else None}
    tol = 3.0 if claim["kind"] == "pct_change" else max(1.0, 0.02 * abs(actual))
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
                    r"(?P<suf>(?:lakhs?|lacs?|crores?|cr|k)\b)?(?P<pct>\s*%)?", re.I)
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
    by '%' or preceded by a currency token (₹, Rs, INR, $, USD) is always checked, and a percentage
    must match a percent-like value: a probability/fraction in [0, 1] times 100, a pct_change claim, or
    an evidence value whose key names a percentage (pct, percent, share, rate)."""
    vals = [c.get("value") for c in claims if isinstance(c, dict)]
    allowed = list(_flatten_numbers(vals)) + list(_flatten_numbers(evidence))
    allowed += [a * 100 for a in allowed if 0 < abs(a) <= 1]  # probabilities quoted as percents
    pct_ok = [a * 100 for a in allowed if abs(a) <= 1]
    pct_ok += [float(c["value"]) for c in claims if isinstance(c, dict) and c.get("kind") == "pct_change"
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
