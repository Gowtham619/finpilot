from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analyze import detect_recurring


@dataclass
class Model:
    recurring: list
    cats: list
    daily: np.ndarray       # days x cats, positive discretionary spend
    weekdays: np.ndarray    # weekday of each row in daily
    last_date: pd.Timestamp


def current_balance(df, opening):
    return float(opening + df["amount"].sum())


def fit(df, window_days=90) -> Model:
    if df is None or len(df) == 0:
        raise ValueError("need at least 1 day of history")
    rec = detect_recurring(df)
    rec_m = {r["merchant"] for r in rec}
    end = df["date"].max().normalize()
    start = max(end - pd.Timedelta(days=window_days - 1), df["date"].min().normalize())
    idx = pd.date_range(start, end)
    disc = df[(~df["merchant"].isin(rec_m)) & (df["amount"] < 0) & (df["date"] >= idx[0])]
    cats = sorted(disc["category"].unique())
    if cats:
        mat = (disc.assign(d=disc["date"].dt.normalize(), s=-disc["amount"])
               .pivot_table(index="d", columns="category", values="s", aggfunc="sum")
               .reindex(idx).fillna(0.0).reindex(columns=cats, fill_value=0.0))
        daily = mat.to_numpy()
    else:
        daily = np.zeros((len(idx), 0))
    return Model(rec, cats, daily, np.asarray(idx.weekday), end)


def recurring_flows(model, dates):
    flows = np.zeros(len(dates))
    for r in model.recurring:
        for i, d in enumerate(dates):
            if d.day == min(r["day"], d.days_in_month):
                flows[i] += r["amount"]
    return flows


@dataclass
class Result:
    dates: pd.DatetimeIndex
    paths: np.ndarray

    @property
    def p10(self): return np.percentile(self.paths, 10, axis=0)
    @property
    def p50(self): return np.percentile(self.paths, 50, axis=0)
    @property
    def p90(self): return np.percentile(self.paths, 90, axis=0)
    @property
    def p_overdraft(self): return float((self.paths.min(axis=1) < 0).mean())
    @property
    def floor_p50(self): return float(np.median(self.paths.min(axis=1)))
    @property
    def end_p10(self): return float(self.p10[-1])
    @property
    def end_p50(self): return float(self.p50[-1])
    @property
    def end_p90(self): return float(self.p90[-1])

    @property
    def risk_day(self):
        frac = (self.paths < 0).mean(axis=0)
        hit = np.where(frac > 0.10)[0]
        return str(self.dates[hit[0]].date()) if len(hit) else None

    def summary(self):
        return {"p_overdraft": round(self.p_overdraft, 3), "floor_p50": round(self.floor_p50, 2),
                "end_p10": round(self.end_p10, 2), "end_p50": round(self.end_p50, 2),
                "end_p90": round(self.end_p90, 2), "risk_day": self.risk_day}


def monte_carlo(model, balance, horizon=60, n=1000, seed=0, extra_flows=(), cuts=None) -> Result:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(model.last_date + pd.Timedelta(days=1), periods=horizon)
    fixed = recurring_flows(model, dates)
    for d, a in extra_flows:
        d = pd.Timestamp(d).normalize()
        if d in dates:
            fixed[dates.get_loc(d)] += a
    cuts = cuts or {}
    scale = np.clip(np.array([1.0 - float(cuts.get(c, 0.0)) for c in model.cats], dtype=float), 0.0, 1.0)
    by_wd = {}
    for w in range(7):
        b = np.where(model.weekdays == w)[0]
        by_wd[w] = b if len(b) else np.arange(len(model.weekdays))  # sparse history fallback
    rows = np.empty((n, horizon), dtype=int)
    for j, d in enumerate(dates):  # weekday-matched bootstrap of observed days
        rows[:, j] = rng.choice(by_wd[d.weekday()], size=n)
    disc = (model.daily[rows] * scale).sum(axis=2) if model.cats else np.zeros((n, horizon))
    paths = balance + np.cumsum(fixed[None, :] - disc, axis=1)
    return Result(dates, paths)


def goal_status(model, balance, target, deadline, n=1000, seed=0, max_horizon=365):
    deadline = pd.Timestamp(deadline).normalize()
    horizon = max(1, (deadline - model.last_date).days)
    eff = min(horizon, max_horizon)
    r = monte_carlo(model, balance, eff, n, seed)
    p_met = float((r.paths[:, -1] >= target).mean())
    far = monte_carlo(model, balance, max_horizon, n, seed)
    hit = np.where(far.p50 >= target)[0]
    return {"p_met": round(p_met, 3), "shortfall_p50": round(float(target - r.end_p50), 2),
            "eta": str(far.dates[hit[0]].date()) if len(hit) else None, "horizon": int(eff),
            "capped": bool(horizon > max_horizon)}
