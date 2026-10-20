"""Money Weather: a daily risk calendar, storm runs, a pre-mortem of failing paths and umbrellas.

Definitions (shared across the product):
  buffer      7 days of average total spend (mean daily discretionary spend over the model window
              + |recurring outflows| per month / 30.4). User-overridable.
  lifestyle   multiplier on discretionary spend (0..2), separate from per-category `cuts`.
  day risk    fraction of simulated paths whose balance on day d is below `buffer`.
  label       clear p<0.05, cloudy <0.15, rain <0.35, storm otherwise (judged on the 3-dp value shown).
  storm       maximal run of consecutive days with p >= 0.25.
  failing     a path below `buffer` on at least one day.

Everything rides on forecast.monte_carlo, whose RNG stream is independent of cuts / spend_scale /
extra_flows, so every comparison here (lifestyle, umbrellas) uses common random numbers.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import forecast as F

STORM_P = 0.25
MAX_LIFESTYLE = 2.0
CUT_STEPS = (10, 20, 30)


def label(p: float) -> str:
    if p < 0.05:
        return "clear"
    if p < 0.15:
        return "cloudy"
    if p < 0.35:
        return "rain"
    return "storm"


def default_buffer(model) -> float:
    daily = float(model.daily.sum() / len(model.daily)) if model.daily.size and len(model.daily) else 0.0
    bills = sum(abs(r["amount"]) for r in model.recurring if r["amount"] < 0)
    return float(7.0 * (daily + bills / 30.4))


def _lifestyle(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        raise ValueError("lifestyle must be a number in 0..2") from None
    if not (math.isfinite(x) and 0.0 <= x <= MAX_LIFESTYLE):
        raise ValueError("lifestyle must be finite and in 0..2")
    return x


def _buffer(model, buffer) -> float:
    if buffer is None:
        return default_buffer(model)
    b = float(buffer)
    if not math.isfinite(b) or b < 0:
        raise ValueError("buffer must be finite and >= 0")
    return b


def _run(model, balance, horizon, n, seed, lifestyle, cuts, extra_flows, detail=False):
    if int(horizon) < 1 or int(n) < 1:
        raise ValueError("horizon and n must be >= 1")
    return F.monte_carlo(model, balance, int(horizon), int(n), seed, extra_flows=extra_flows,
                         cuts=cuts, spend_scale=_lifestyle(lifestyle), detail=detail)


def _calendar(res, buf):
    p = np.round((res.paths < buf).mean(axis=0), 3)
    q = np.round(np.percentile(res.paths, [10, 50, 90], axis=0), 2)
    return [{"date": str(d.date()), "p_below_buffer": float(p[i]), "label": label(float(p[i])),
             "p10": float(q[0, i]), "p50": float(q[1, i]), "p90": float(q[2, i])}
            for i, d in enumerate(res.dates)]


def risk_calendar(model, balance, horizon=60, n=1000, seed=0, buffer=None, lifestyle=1.0,
                  cuts=None, extra_flows=()):
    buf = _buffer(model, buffer)
    return _calendar(_run(model, balance, horizon, n, seed, lifestyle, cuts, extra_flows), buf)


def storms(calendar):
    out, cur = [], []

    def close():
        if cur:
            pk = max(cur, key=lambda c: c["p_below_buffer"])  # first day of the maximum
            out.append({"start": cur[0]["date"], "end": cur[-1]["date"], "days": len(cur),
                        "peak_p": round(float(pk["p_below_buffer"]), 3), "peak_date": pk["date"],
                        "label": label(pk["p_below_buffer"])})

    for c in calendar:
        if c["p_below_buffer"] >= STORM_P:
            cur.append(c)
        else:
            close()
            cur = []
    close()
    return sorted(out, key=lambda s: (-s["peak_p"], s["start"]))


def _autopsy(model, res, buf, horizon):
    breach = res.paths < buf
    failing = breach.any(axis=1)
    n_paths, n_fail = int(len(failing)), int(failing.sum())
    out = {"p_any_breach": round(n_fail / n_paths, 3), "n_failing": n_fail, "n_paths": n_paths,
           "causes": [], "bills": [], "first_breach": {"date": None, "share": 0.0}}
    if n_fail == 0:
        return out
    first = np.argmax(breach[failing], axis=1)
    counts = np.bincount(first, minlength=breach.shape[1])
    mode = int(np.argmax(counts))  # ties -> earliest day
    out["first_breach"] = {"date": str(res.dates[mode].date()), "share": round(float(counts[mode] / n_fail), 3)}
    if n_fail < n_paths and model.cats and res.disc_by_cat is not None:
        fm = res.disc_by_cat[failing].mean(axis=0)
        om = res.disc_by_cat[~failing].mean(axis=0)
        ex = fm - om
        total = float(ex[ex > 0].sum())
        rows = [{"category": model.cats[i], "excess_per_path": round(float(ex[i]), 2),
                 "failing_mean": round(float(fm[i]), 2), "other_mean": round(float(om[i]), 2),
                 "share_of_gap": round(float(ex[i] / total), 3)}
                for i in np.argsort(-ex, kind="stable") if ex[i] > 0]
        out["causes"] = rows[:5]
    # recurring outflows landing on or before the median first-breach day
    med = res.dates[int(np.sort(first)[(len(first) - 1) // 2])]
    bills = []
    for r in model.recurring:
        if r["amount"] >= 0:
            continue
        for d in res.dates:
            if d <= med and d.day == min(r["day"], d.days_in_month):
                bills.append({"merchant": r["merchant"], "date": str(d.date()), "amount": round(float(r["amount"]), 2)})
    out["bills"] = sorted(bills, key=lambda b: (b["date"], b["merchant"]))
    return out


def autopsy(model, balance, horizon=60, n=1000, seed=0, buffer=None, lifestyle=1.0, cuts=None, extra_flows=()):
    buf = _buffer(model, buffer)
    res = _run(model, balance, horizon, n, seed, lifestyle, cuts, extra_flows, detail=True)
    return _autopsy(model, res, buf, horizon)


def _p_any(res, buf):
    return float((res.paths < buf).any(axis=1).mean())


def _umbrellas(model, balance, horizon, n, seed, buf, lifestyle, cuts, extra_flows, base_p):
    if base_p <= 0 or not model.cats:
        return []
    cuts = dict(cuts or {})
    life = _lifestyle(lifestyle)
    mean_daily = model.daily.mean(axis=0)
    top = sorted(range(len(model.cats)), key=lambda i: (-mean_daily[i], model.cats[i]))[:4]
    cands = []
    for i in top:
        cat = model.cats[i]
        if mean_daily[i] <= 0:
            continue
        have = min(max(float(cuts.get(cat, 0.0)), 0.0), 1.0)
        for pct in CUT_STEPS:
            c = dict(cuts)
            c[cat] = 1.0 - (1.0 - have) * (1.0 - pct / 100.0)  # stack on any existing cut
            after = _p_any(_run(model, balance, horizon, n, seed, life, c, extra_flows), buf)
            red = base_p - after
            sac = float(mean_daily[i]) * life * (1.0 - have) * 30 * pct / 100.0
            if red > 0 and sac > 0:
                cands.append({"category": cat, "cut_pct": pct, "p_any_breach_before": round(base_p, 3),
                              "p_any_breach_after": round(after, 3), "risk_reduction": round(red, 3),
                              "monthly_sacrifice": round(sac, 2),
                              "reduction_per_1000": round(red / (sac / 1000.0), 3)})
    cands.sort(key=lambda c: (-c["reduction_per_1000"], c["category"], c["cut_pct"]))
    return cands[:3]


def umbrellas(model, balance, horizon=60, n=1000, seed=0, buffer=None, lifestyle=1.0, extra_flows=()):
    buf = _buffer(model, buffer)
    base = _p_any(_run(model, balance, horizon, n, seed, lifestyle, None, extra_flows), buf)
    return _umbrellas(model, balance, horizon, n, seed, buf, lifestyle, None, extra_flows, base)


def report(model, balance, horizon=60, n=1000, seed=0, buffer=None, lifestyle=1.0, cuts=None, extra_flows=()):
    buf = _buffer(model, buffer)
    life = _lifestyle(lifestyle)
    res = _run(model, balance, horizon, n, seed, life, cuts, extra_flows, detail=True)
    cal = _calendar(res, buf)
    st = storms(cal)
    aut = _autopsy(model, res, buf, horizon)
    # umbrellas stack on top of whatever `cuts` the map already applies
    umb = _umbrellas(model, balance, horizon, n, seed, buf, life, cuts, extra_flows, _p_any(res, buf))
    worst = max(cal, key=lambda c: c["p_below_buffer"])
    return {"buffer": round(buf, 2), "lifestyle": life, "horizon": int(horizon), "calendar": cal,
            "storms": st, "autopsy": aut, "umbrellas": umb,
            "headline": {"p_any_breach": aut["p_any_breach"], "worst_day": worst["date"],
                         "worst_p": worst["p_below_buffer"], "n_storm_days": int(sum(s["days"] for s in st))}}
