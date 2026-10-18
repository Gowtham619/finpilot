from __future__ import annotations

import re

import pandas as pd

ALLOWED_CONSTANTS = {12, 24, 30, 60, 90, 100}


def _month(df):
    return df["date"].dt.to_period("M").astype(str)


def recompute(claim, df):
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


def check_claim(claim, df):
    actual = recompute(claim, df)
    if actual is None:
        return {"ok": False, "actual": None}
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


def ungrounded_numbers(text, claims, evidence):
    allowed = list(_flatten_numbers([c.get("value") for c in claims])) + list(_flatten_numbers(evidence))
    allowed += [a * 100 for a in allowed if 0 < abs(a) <= 1]  # probabilities quoted as percents
    text = re.sub(r"\d{4}-\d{2}(-\d{2})?", " ", text)  # ISO dates and months are not claims
    out = []
    for raw in re.findall(r"\d[\d,]*\.?\d*", text):
        raw = raw.replace(",", "").rstrip(".")
        if not raw:
            continue
        n = float(raw)
        if n < 10 or 1990 <= n <= 2100 or n in ALLOWED_CONSTANTS:
            continue
        if not any(abs(n - a) <= max(1.0, 0.01 * abs(a)) for a in allowed):
            out.append(n)
    return out
