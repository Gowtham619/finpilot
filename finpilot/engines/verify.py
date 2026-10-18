from __future__ import annotations

import math
import re

import pandas as pd

ALLOWED_CONSTANTS = {12, 24, 30, 60, 90, 100}


def _month(df):
    return df["date"].dt.to_period("M").astype(str)


def _recompute(claim, df):
    k = claim["kind"]
    if k in ("monthly_cost", "annual_cost"):
        g = df[(df["merchant"] == claim["merchant"]) & (df["amount"] < 0)].sort_values("date").tail(3)
        if g.empty:
            return None
        v = float(abs(g["amount"].mean()))
        return v * 12 if k == "annual_cost" else v
    if k == "month_total":
        m = df[(df["category"] == claim["category"]) & (df["amount"] < 0) & (_month(df) == claim["month"])]
        return None if m.empty else float(-m["amount"].sum())
    if k == "pct_change":
        s = df[(df["category"] == claim["category"]) & (df["amount"] < 0)]
        tot = (-s["amount"]).groupby(_month(s)).sum()
        if claim["month"] not in tot.index or len(tot) < 3:
            return None
        return float((tot[claim["month"]] / tot.median() - 1) * 100)
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
_TOKEN = re.compile(r"(?P<cur>₹|Rs\.?|INR)?\s*(?P<num>\d[\d,]*\.?\d*)\s*"
                    r"(?P<suf>(?:lakhs?|lacs?|crores?|cr|k)\b)?", re.I)
_DATE = re.compile(r"(?<!\d)(19|20)\d{2}-\d{2}(-\d{2})?(?!\d)")


def ungrounded_numbers(text, claims, evidence):
    vals = [c.get("value") for c in claims if isinstance(c, dict)]
    allowed = list(_flatten_numbers(vals)) + list(_flatten_numbers(evidence))
    allowed += [a * 100 for a in allowed if 0 < abs(a) <= 1]  # probabilities quoted as percents
    text = _DATE.sub(" ", str(text))  # ISO dates and months are not claims
    out = []
    for m in _TOKEN.finditer(text):
        tok = m.group("num").rstrip(".,")
        raw = tok.replace(",", "")
        if not raw:
            continue
        cur, suf = m.group("cur"), m.group("suf")
        n = float(raw) * (_SUFFIX[suf.lower()] if suf else 1)
        if not cur and not suf:
            if re.fullmatch(r"\d{4}", tok) and 1990 <= n <= 2100:
                continue
            if n in ALLOWED_CONSTANTS:
                continue
        if n < 10:
            continue
        if not any(abs(n - a) <= max(1.0, 0.01 * abs(a)) for a in allowed):
            out.append(n)
    return out
