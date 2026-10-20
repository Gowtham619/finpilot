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
    return {"amount": float(amount), "date": str(pd.Timestamp(on).date()), "horizon": int(horizon),
            "baseline": b, "with_purchase": a,
            "ignored_flows": _count_ignored(model, horizon, flows),
            "delta_p_overdraft": round(a["p_overdraft"] - b["p_overdraft"], 3),
            "delta_end_p50": round(a["end_p50"] - b["end_p50"], 2)}


def best_date(model, balance, amount, candidates, horizon=60, n=1000, seed=0, financed_months=0,
              max_ignored=None):
    """Candidate start dates ranked by risk; a financed purchase is simulated as its installments.
    Each entry carries ignored_flows (installments past the window). With max_ignored, candidates that
    drop more installments than that are excluded: dropping a payment would make a date look safer."""
    out = []
    start, end = _window(model, horizon)
    for c in candidates:
        if not (start <= pd.Timestamp(c).normalize() <= end):
            continue
        flows = purchase_flows(amount, c, financed_months)
        ign = _count_ignored(model, horizon, flows)
        if max_ignored is not None and ign > max_ignored:
            continue
        r = monte_carlo(model, balance, horizon, n, seed, extra_flows=flows)
        out.append({"date": str(pd.Timestamp(c).date()), "ignored_flows": ign, **r.summary()})
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


def _bal(x):
    """Unsigned wording (the grounding check compares unsigned figures)."""
    return f"₹{x:,.0f}" if x >= 0 else f"overdrawn by ₹{-x:,.0f}"


def duel(model, balance, a, b, horizon=90, n=1000, seed=0):
    """Winner: lower overdraft risk (if more than 2 points apart), else higher median end balance,
    else higher median lowest balance, else higher average median balance; tie only if all are equal."""
    ra = monte_carlo(model, balance, horizon, n, seed, extra_flows=a["flows"], cuts=a.get("cuts"))
    rb = monte_carlo(model, balance, horizon, n, seed, extra_flows=b["flows"], cuts=b.get("cuts"))
    sa = {**ra.summary(), "avg_p50": round(float(ra.p50.mean()), 2)}
    sb = {**rb.summary(), "avg_p50": round(float(rb.p50.mean()), 2)}
    if abs(sa["p_overdraft"] - sb["p_overdraft"]) > 0.02:
        winner, basis = ("a" if sa["p_overdraft"] < sb["p_overdraft"] else "b"), "overdraft risk"
    else:
        winner, basis = "tie", None
        for key, label in (("end_p50", "end balance"), ("floor_p50", "lowest balance"),
                           ("avg_p50", "average balance")):
            if sa[key] != sb[key]:
                winner, basis = ("a" if sa[key] > sb[key] else "b"), label
                break
    if winner == "tie":
        reason = (f"Tie: both options have overdraft risk {sa['p_overdraft']*100:.0f}%, median balance in "
                  f"{horizon} days {_bal(sa['end_p50'])}, median lowest balance {_bal(sa['floor_p50'])} and "
                  f"average median balance {_bal(sa['avg_p50'])}.")
    else:
        w, l = (sa, sb) if winner == "a" else (sb, sa)
        wn = a["name"] if winner == "a" else b["name"]
        reason = (f"{wn} wins on {basis}: overdraft risk {w['p_overdraft']*100:.0f}% vs "
                  f"{l['p_overdraft']*100:.0f}%, median balance in {horizon} days "
                  f"{_bal(w['end_p50'])} vs {_bal(l['end_p50'])}")
        if basis in ("lowest balance", "average balance"):
            reason += f", median lowest balance {_bal(w['floor_p50'])} vs {_bal(l['floor_p50'])}"
        if basis == "average balance":
            reason += f", average median balance {_bal(w['avg_p50'])} vs {_bal(l['avg_p50'])}"
        reason += "."
    return {"a": {"name": a["name"], "summary": sa, "bands": _bands(ra)},
            "b": {"name": b["name"], "summary": sb, "bands": _bands(rb)},
            "winner": winner, "reason": reason, "horizon": horizon,
            "ignored_flows": (_count_ignored(model, horizon, a["flows"])
                              + _count_ignored(model, horizon, b["flows"]))}
