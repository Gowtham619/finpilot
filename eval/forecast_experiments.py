"""Forecast refinement experiments (v2 task 5). Run: python -m eval.forecast_experiments
Writes eval/forecast_experiments.md. Takes about a minute.

Each candidate wraps forecast.fit / forecast.monte_carlo as seen by the committed harness
(eval.run_eval.eval_forecast): same generators, same 3 monthly origins per persona, same metric code.
Nothing here changes finpilot/engines/forecast.py; this is an experiment log, not shipped behaviour.

Tuned knobs (k, w, v, window) are chosen ONLY by inner backtests on the training slice: for a held-out
cutoff, the inner origins are the first days of the last 3 complete months before the cutoff (each with
>= 60 days of earlier history); the model is refit on data before each inner origin and scored on that
inner month. No held-out month is used to choose anything. The fixed-window rows (60/120/180 days) are a
sensitivity analysis whose value would have to be picked by looking at held-out results, so they are
NOT eligible for adoption whatever they score.

Adoption rule (fixed before any candidate was scored): adopt only if, averaged over 3 personas x 3
origins on EACH generator, month-end MAE / monthly spend improves by >= 0.01 on each generator OR band
coverage moves toward 0.80 by >= 0.03 on each generator, AND balance MAPE worsens by no more than 0.002
(original) / 0.01 (OOD) AND coverage moves away from 0.80 by no more than 0.02 on either generator.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from eval import run_eval as R
from eval.generate import PERSONAS, load_persona
from eval.generate_messy import OOD_PERSONAS, load_messy
from finpilot.engines import forecast as F

OUT = pathlib.Path(__file__).parent / "forecast_experiments.md"
BASE_FIT, BASE_MC = F.fit, F.monte_carlo
NOMINAL = 0.80
KEYS = ("balance_mape", "mae_vs_monthly_spend", "band_coverage")


def score(gens):
    out = {}
    for g, ps in gens.items():
        model = [R.eval_forecast(p)["model"] for p in ps]
        out[g] = {k: round(float(np.mean([x[k] for x in model])), 4) for k in KEYS}
    return out


# ---------------- candidate mechanics ----------------
def _window_dates(model):
    return pd.date_range(model.last_date - pd.Timedelta(days=len(model.daily) - 1), model.last_date)


def _rows(model, dates, n, seed):
    """The base engine's weekday-matched row draws, regenerated from the same seed (identical draws)."""
    rng = np.random.default_rng(seed)
    by_wd = {}
    for wd in range(7):
        b = np.where(model.weekdays == wd)[0]
        by_wd[wd] = b if len(b) else np.arange(len(model.weekdays))
    rows = np.empty((n, len(dates)), dtype=int)
    for j, d in enumerate(dates):
        rows[:, j] = rng.choice(by_wd[d.weekday()], size=n)
    return rows


def mc_variant(model, balance, horizon=60, n=1000, seed=0, extra_flows=(), cuts=None, spend_scale=1.0, detail=False):
    o = getattr(model, "_opt", {})
    r = BASE_MC(model, balance, horizon, n, seed, extra_flows, cuts, spend_scale, detail)
    paths, dates = r.paths, r.dates
    w, v = o.get("w", 0.0), o.get("v", 0.0)
    if w or v:
        fixed = F.recurring_flows(model, dates)
        disc = -(np.diff(np.concatenate([np.full((n, 1), balance), paths], axis=1), axis=1) - fixed[None, :])
        s = pd.Series(model.daily.sum(axis=1), index=_window_dates(model))
        if w:  # (c) last month's same-day discretionary spend, blended at weight w
            prev = np.array([s.get(d - pd.DateOffset(months=1), np.nan) for d in dates])
            prev = np.where(np.isnan(prev), disc.mean(axis=0), prev)
            disc = (1 - w) * disc + w * prev[None, :]
        if v:  # (d) day-of-month mean discretionary profile, blended at weight v
            dom = s.groupby(s.index.day).mean()
            prof = np.array([dom.get(d.day, s.mean()) for d in dates])
            disc = (1 - v) * disc + v * prof[None, :]
        paths = balance + np.cumsum(fixed[None, :] - disc, axis=1)
    if o.get("credits") is not None:  # (b2) non-recurring credits resampled on the same rows (CRN kept)
        paths = paths + np.cumsum(o["credits"][_rows(model, dates, n, seed)], axis=1)
    k = o.get("k", 1.0)
    if k != 1.0:  # (a) spread around the per-day median scaled by k
        med = np.median(paths, axis=0)
        paths = med + k * (paths - med)
    return F.Result(dates, paths, r.disc_by_cat, r.disc_cum_by_cat)


def _credits(df, model):
    rec_m = {x["merchant"] for x in model.recurring}
    idx = _window_dates(model)
    c = df[(~df["merchant"].isin(rec_m)) & (df["amount"] > 0) & (df["date"] >= idx[0])]
    return c.groupby(c["date"].dt.normalize())["amount"].sum().reindex(idx, fill_value=0.0).to_numpy()


def make_model(df, opt):
    win = opt.get("win", 90)
    m = BASE_FIT(df, win if isinstance(win, int) else 90)
    m._opt = {}
    if opt.get("last_amount"):  # (b1) last observed amount for low-cv recurring items
        for x in m.recurring:
            if x["cv"] <= 0.1:
                x["amount"] = float(df[df["merchant"] == x["merchant"]].sort_values("date")["amount"].iloc[-1])
    if opt.get("credits"):
        m._opt["credits"] = _credits(df, m)
    for key in ("k", "w", "v"):
        if isinstance(opt.get(key), (int, float)):
            m._opt[key] = opt[key]
    return m


def _fixed(opt):
    return {k: v for k, v in opt.items() if v != "tune"}


def inner_score(train, opt, metric):
    vals = []
    last = train["date"].max().to_period("M")
    for j in range(3):
        o = (last - j).to_timestamp()
        if (o - train["date"].min()).days < 60:
            continue
        end = (o + pd.offsets.MonthEnd(0)).normalize()
        sub = train[train["date"] < o]
        m = make_model(sub, opt)
        m.last_date = o - pd.Timedelta(days=1)
        bal0 = 0.0  # coverage and month-end error are shift invariant
        actual = bal0 + np.cumsum(R._daily_net(train, o, end))
        vals.append(metric(mc_variant(m, bal0, len(actual), 500), actual))
    return float(np.mean(vals)) if vals else 0.0


def _cov_gap(r, actual):
    return abs(float(np.mean((actual >= r.p10) & (actual <= r.p90))) - NOMINAL)


def _end_err(r, actual):
    return abs(r.p50[-1] - actual[-1])


GRIDS = {"k": ([0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5], _cov_gap),
         "w": ([0.0, 0.25, 0.5, 0.75, 1.0], _end_err),
         "v": ([0.0, 0.25, 0.5, 0.75, 1.0], _end_err),
         "win": ([60, 90, 120, 180], _end_err)}


def make_fit(opt, chosen):
    def fit(df, window_days=90):
        o = _fixed(opt)
        for key, (grid, metric) in GRIDS.items():
            if opt.get(key) == "tune":
                o[key] = min(grid, key=lambda x: inner_score(df, {**o, key: x}, metric))
                chosen.setdefault(key, []).append(o[key])
        return make_model(df, o)
    return fit


CANDIDATES = [  # (name, options, eligible for adoption)
    ("baseline (committed engine)", {}, False),
    ("a: band calibration, k tuned in-sample", {"k": "tune"}, True),
    ("b1: last amount for recurring items with cv<=0.1", {"last_amount": True}, True),
    ("b2: resample non-recurring credits (refunds etc.)", {"credits": True}, True),
    ("c: seasonal blend, w tuned in-sample", {"w": "tune"}, True),
    ("d: day-of-month profile blend, v tuned in-sample", {"v": "tune"}, True),
    ("b2+c combined", {"credits": True, "w": "tune"}, True),
    ("e: fit window tuned in-sample (60/90/120/180)", {"win": "tune"}, True),
    ("sensitivity: fixed 60-day window", {"win": 60}, False),
    ("sensitivity: fixed 120-day window", {"win": 120}, False),
    ("sensitivity: fixed 180-day window", {"win": 180}, False),
]


def verdict(base, new):
    gap = lambda c: abs(c - NOMINAL)  # noqa: E731
    gains = all(base[g]["mae_vs_monthly_spend"] - new[g]["mae_vs_monthly_spend"] >= 0.01 for g in base) or \
        all(gap(base[g]["band_coverage"]) - gap(new[g]["band_coverage"]) >= 0.03 for g in base)
    guard = (new["original"]["balance_mape"] - base["original"]["balance_mape"] <= 0.002
             and new["ood"]["balance_mape"] - base["ood"]["balance_mape"] <= 0.01
             and all(gap(new[g]["band_coverage"]) - gap(base[g]["band_coverage"]) <= 0.02 for g in base))
    return gains and guard


def run_all():
    gens = {"original": [load_persona(n) for n in PERSONAS], "ood": [load_messy(n) for n in OOD_PERSONAS]}
    rows, base = [], None
    try:
        for name, opt, eligible in CANDIDATES:
            chosen = {}
            R.F.fit, R.F.monte_carlo = make_fit(opt, chosen), mc_variant
            s = score(gens)
            base = base or s
            rows.append((name, s, eligible, verdict(base, s), {k: sorted(set(v)) for k, v in chosen.items()}))
        R.F.fit = BASE_FIT
        for sd in (1, 2, 3, 4):  # noise floor: the committed engine with a different Monte Carlo seed
            R.F.monte_carlo = lambda m, b, h=60, n=1000, seed=0, *a, _sd=sd, **kw: BASE_MC(m, b, h, n, _sd, *a, **kw)
            rows.append((f"noise floor: committed engine, MC seed {sd}", score(gens), False, None, {}))
    finally:
        R.F.fit, R.F.monte_carlo = BASE_FIT, BASE_MC
    return rows


def render(rows):
    out = ["# Forecast refinement experiments", "", __doc__.split("\n\n", 1)[1].strip(), "",
           "| Candidate | Orig MAPE | Orig month-end MAE/spend | Orig coverage | OOD MAPE | OOD month-end MAE/spend | OOD coverage | Meets rule | Knob values chosen |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, s, eligible, ok, chosen in rows:
        o, d = s["original"], s["ood"]
        mark = "-" if ok is None or name.startswith("baseline") else (("yes" if ok else "no") + ("" if eligible else " (not eligible)"))
        ch = "; ".join(f"{k} in {v}" for k, v in chosen.items()) or "-"
        out.append(f"| {name} | {o['balance_mape']} | {o['mae_vs_monthly_spend']} | {o['band_coverage']} | "
                   f"{d['balance_mape']} | {d['mae_vs_monthly_spend']} | {d['band_coverage']} | {mark} | {ch} |")
    return "\n".join(out) + "\n"


def main():
    md = render(run_all())
    OUT.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
