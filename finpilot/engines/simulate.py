from __future__ import annotations

import pandas as pd

from .forecast import monte_carlo

STEPS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)


def purchase_flows(amount, on, financed_months=0):
    on = pd.Timestamp(on).normalize()
    if not financed_months:
        return [(on, -float(amount))]
    part = float(amount) / financed_months
    return [(on + pd.DateOffset(months=k), -part) for k in range(financed_months)]


def _window(model, horizon):
    start = model.last_date + pd.Timedelta(days=1)
    return start, start + pd.Timedelta(days=horizon - 1)


def _count_ignored(model, horizon, flows):
    start, end = _window(model, horizon)
    return sum(1 for d, _ in flows if not (start <= pd.Timestamp(d).normalize() <= end))


def simulate_purchase(model, balance, amount, on, horizon=60, n=1000, seed=0, financed_months=0):
    flows = purchase_flows(amount, on, financed_months)
    start, end = _window(model, horizon)
    if not (start <= flows[0][0] <= end):
        d = flows[0][0].date()
        raise ValueError(f"purchase date {d} is outside the forecast window "
                         f"{start.date()}..{end.date()}")
    base = monte_carlo(model, balance, horizon, n, seed)
    alt = monte_carlo(model, balance, horizon, n, seed, extra_flows=flows)
    b, a = base.summary(), alt.summary()
    return {"amount": float(amount), "date": str(pd.Timestamp(on).date()),
            "baseline": b, "with_purchase": a,
            "ignored_flows": _count_ignored(model, horizon, flows),
            "delta_p_overdraft": round(a["p_overdraft"] - b["p_overdraft"], 3),
            "delta_end_p50": round(a["end_p50"] - b["end_p50"], 2)}


def best_date(model, balance, amount, candidates, horizon=60, n=1000, seed=0):
    out = []
    start, end = _window(model, horizon)
    for c in candidates:
        if not (start <= pd.Timestamp(c).normalize() <= end):
            continue
        r = monte_carlo(model, balance, horizon, n, seed, extra_flows=purchase_flows(amount, c))
        out.append({"date": str(pd.Timestamp(c).date()), **r.summary()})
    return sorted(out, key=lambda x: (x["p_overdraft"], -x["floor_p50"]))


def cheapest_fix(model, balance, extra_flows, target_p=0.10, horizon=60, n=1000, seed=0):
    """Smallest discretionary cut (single top category, or uniform) that brings
    P(overdraft) under target_p. Cheapest = lowest monthly saving required.

    Returns None when no option reaches the target or there are no
    discretionary categories to cut."""
    base = monte_carlo(model, balance, horizon, n, seed, extra_flows=extra_flows)
    if base.p_overdraft <= target_p:
        return {"category": None, "cut_pct": 0, "p_overdraft": round(base.p_overdraft, 3),
                "est_monthly_saving": 0.0}
    if not model.cats:
        return None
    mean_daily = dict(zip(model.cats, model.daily.mean(axis=0)))
    top = sorted(mean_daily, key=mean_daily.get, reverse=True)[:3]
    options = [(c, [c]) for c in top] + [("all", list(model.cats))]
    best = None
    for name, cats in options:
        for s in STEPS:
            r = monte_carlo(model, balance, horizon, n, seed, extra_flows=extra_flows,
                            cuts={c: s for c in cats})
            if r.p_overdraft <= target_p:
                saving = float(sum(mean_daily[c] for c in cats) * 30 * s)
                cand = {"category": name, "cut_pct": int(round(s * 100)),
                        "p_overdraft": round(r.p_overdraft, 3), "est_monthly_saving": round(saving, 2)}
                if best is None or saving < best["est_monthly_saving"]:
                    best = cand
                break
    return best


def _bands(res):
    return {"dates": [str(d.date()) for d in res.dates],
            "p10": [round(float(x), 2) for x in res.p10],
            "p50": [round(float(x), 2) for x in res.p50],
            "p90": [round(float(x), 2) for x in res.p90]}


def duel(model, balance, a, b, horizon=90, n=1000, seed=0):
    ra = monte_carlo(model, balance, horizon, n, seed, extra_flows=a["flows"], cuts=a.get("cuts"))
    rb = monte_carlo(model, balance, horizon, n, seed, extra_flows=b["flows"], cuts=b.get("cuts"))
    sa, sb = ra.summary(), rb.summary()
    if abs(sa["p_overdraft"] - sb["p_overdraft"]) > 0.02:
        winner, basis = ("a" if sa["p_overdraft"] < sb["p_overdraft"] else "b"), "overdraft risk"
    elif sa["end_p50"] != sb["end_p50"]:
        winner, basis = ("a" if sa["end_p50"] > sb["end_p50"] else "b"), "end balance"
    else:
        winner, basis = "tie", "nothing"
    if winner == "tie":
        reason = "Both options end in the same place."
    else:
        w, l = (sa, sb) if winner == "a" else (sb, sa)
        wn = a["name"] if winner == "a" else b["name"]
        reason = (f"{wn} wins on {basis}: overdraft risk {w['p_overdraft']*100:.0f}% vs "
                  f"{l['p_overdraft']*100:.0f}%, median balance in {horizon} days "
                  f"₹{w['end_p50']:,.0f} vs ₹{l['end_p50']:,.0f}.")
    return {"a": {"name": a["name"], "summary": sa, "bands": _bands(ra)},
            "b": {"name": b["name"], "summary": sb, "bands": _bands(rb)},
            "winner": winner, "reason": reason, "horizon": horizon,
            "ignored_flows": (_count_ignored(model, horizon, a["flows"])
                              + _count_ignored(model, horizon, b["flows"]))}
